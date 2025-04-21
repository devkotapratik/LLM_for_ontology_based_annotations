import re
import json
import torch
import numpy as np
import torch.multiprocessing as mp
import xml.etree.ElementTree as ETree

from pathlib import Path
from tqdm.auto import tqdm
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


BASE_DIR = Path(".").resolve().parent
DATA_DIR = BASE_DIR.joinpath("data")
CRAFT_DIR = sorted(DATA_DIR.joinpath("CRAFT").glob("v*/"), key=lambda x: [int(i) for i in x.name[1:].split(".")], reverse=True)[0]
JSON_DIR = CRAFT_DIR.joinpath("articles", "json")
ANNOT_DIR = JSON_DIR.parent.joinpath("json_with_annotations")
ANNOT_DIR.mkdir(exist_ok=True, parents=True)


def get_annotations_by_id(source_id: str, data_dir: str | Path):
    """Returns a list of annotations given a source_id as input
    Input params:
    source_id (str): 8 digit number as source
    data_dir (str | Path): directory to search for files by source_id

    Return params:
    annotations (dict): list of annotations with start index, end index, spanned text
    GO ID and GO concept.
    """
    reqd_files = [i for i in Path(data_dir).rglob("*.xml") if re.search(f"{source_id}", i.stem)]
    annotations = defaultdict(dict)
    for xml_file in reqd_files:
        if xml_file.suffix == ".xml" and "extension" not in str(xml_file):
            root = ETree.parse(xml_file).getroot()
            assert root.attrib.get("textSource") == source_id + ".txt"
            for child in root:
                if child.tag == "annotation":
                    span_children = child.findall("span")
                    spans = defaultdict(list)
                    for span_child in span_children:
                        span = span_child.attrib
                        spans["span"].append((int(span["start"]), int(span["end"])))
                        spans["spanned_text"] = child.find("spannedText").text
                    annotations[
                        child.find("mention").attrib.get("id")].update(spans)
                if child.tag == "classMention":
                    mention_class = child.find("mentionClass")
                    annotations[child.attrib.get("id")].update({
                        "id": mention_class.get("id"),
                        "concept": mention_class.text
                    })
    return sorted(annotations.values(), key=lambda x: x.get("span")[0])


def save_annotations_json(gpu_id, source_ids, start_idx, complete_path, lock):
    processed = []
    torch.cuda.set_device(gpu_id)
    device = f"cuda:{gpu_id}"
    print(f"[GPU {gpu_id}]: Loading model on {device}...")
    model = SentenceTransformer("Salesforce/SFR-Embedding-Mistral", device=device)
    pbar = tqdm(total=len(source_ids), desc=f"[GPU {gpu_id}: Adding annotations to JSON", position=gpu_id)
    for source_id in source_ids:
        pbar.set_description(f"[GPU {gpu_id}]: Adding annotations to {source_id}.json")
        with open(Path.joinpath(JSON_DIR.parent, "txt", f"{source_id}.txt"), "r") as f:
            article_txt = f.read()
        annotations = get_annotations_by_id(source_id, CRAFT_DIR)
        ### Create new passages using only txt and xml files
        passages_txt = []
        for passage in re.finditer(r"(.+?)(?:[\n]+|$)", article_txt):
            pass_span = passage.span(0)
            passages_txt.append(dict(
                span=passage.span(1),
                text=passage.group(1),
                next=passage.group(0).replace(passage.group(1), ""),
                annotations=[i for i in annotations if i["span"][0][0] >= pass_span[0] and
                    i["span"][-1][1] <= pass_span[1]]
            ))
        embeddings_for_craft_text = [i.flatten().reshape(1, -1) for i in model.encode(
            [i.get("text") for i in passages_txt])
        ]
        ### Read json file from Pubmed using BioC url
        with open(Path.joinpath(JSON_DIR, f"{source_id}.json"), "r", encoding="utf-8") as f:
            article_json = json.load(f)
        passages_json = article_json[0]["documents"][0]["passages"]
        new_passages, removed_passages = [], []
        abbr_pos = -1
        for idx, passage_json in enumerate(passages_json):
            remove = False
            offset, infons, annots = [passage_json[i] for i in ["offset", "infons", "annotations"]]
            original_text = passage_json.pop("text")
            formatted_text = original_text
            section_type, style_type = [infons[i] for i in ["section_type", "type"]]
            # For abbreviation, the first one is the abbreviation and the next
            # passage is its full form - join them with " - "
            if section_type == "ABBR" and style_type == "paragraph":
                abbr_pos += 1
                if abbr_pos%2 == 0:
                    next_pass = passages_json[idx+1]
                    if next_pass["infons"]["section_type"] == "ABBR" and next_pass["infons"]["type"] == "paragraph":
                        formatted_text = f"{formatted_text} - {passages_json[idx + 1]['text']}"
                else:
                    continue
            
            # Get embedding for string from json file to compare with all strings from txt file
            # Here a string is everything seperated by one or more newlines.
            emb = model.encode(formatted_text).flatten().reshape(1, -1)
            similarity = [j for i in embeddings_for_craft_text for j in cosine_similarity(emb, i)[0]]
            if max(similarity) > 0.95: # Only consider a match is max similarity > 0.95.
                reqd_idx = np.argmax(similarity)
                reqd_passage = passages_txt.pop(reqd_idx)
                embeddings_for_craft_text.pop(reqd_idx)
                original_text = reqd_passage.pop("text")
            else:
                reqd_passage = passage_json.copy()
            reqd_passage["json_text"] = formatted_text
            reqd_passage["original_text"] = original_text
            passage_json.update(reqd_passage)
            
            if section_type == "TABLE" and infons.get("xml"):
                remove = True
            else:
                if section_type in ["REF", "COMP_INT"] and not len(annots):
                    remove = True
            if not remove:
                if style_type in ["abstract_title_1", "title", "title_1", "title_2", "title_3"] and not len(annots):
                    remove = True
            if not remove:
                new_passages.append(passage_json)
            else:
                if len(passage_json.get("annotations")):
                    new_passages.append(passage_json)
                else:
                    removed_passages.append(passage_json)
        # article_json[0]["documents"][0]["passages"] = new_passages
        annots_txt = [i for i in passages_txt if i.get("annotations")]
        n_txt = [j for i in annots_txt for j in i.get("annotations")]
        annots_json = [i for i in new_passages if i.get("annotations")]
        n_json = [j for i in annots_json for j in i.get("annotations")]
        # try:
        if len(n_txt) != 0: # If we still have annotations that are not added to json file
            # raise AssertionError(f"{n_txt} are not in JSON with PubMed ID: {source_id}")
            for miss_idx, passage_txt in enumerate(annots_txt):
                original_text = passage_txt.pop("text")
                formatted_text = original_text
                # Text in passage from Pubmed data using BioC does not have citations.
                # To compare and match this text with text from CRAFT corpus that has
                # citations, we remove the citations in this step.
                # Find the citations and reverse the order so that during the removal of
                # citation, it happens from the right to the left
                citations = list(re.finditer(r"\s*\[[0-9, -]+\]\s*", formatted_text))[::-1]
                # Get the number of whitespaces in each citation, this is important so that
                # during the removal we know how many whitespaces are required to replace
                # the citation.
                # E.g. if formatted_text = "this is first citation [1, 2] and second citation [3]."
                # We calculate whitespaces before and after citation and replace with that
                # many - 1 whitespaces. We use 2-1=1 whitespace in the first citation and
                # 1-1=0 whitespace in the second citation.
                ws_in_citations = [len(re.findall(r"\s", i.group()))-1 for i in citations]
                spans = [i.span() for i in citations]
                for idx, span in enumerate(spans):
                    formatted_text = formatted_text[:span[0]] + ws_in_citations[idx]*" " + formatted_text[span[1]:]

                passage_json = dict(
                    offset=-1,
                    infons=dict(
                        section_type="MISSING",
                        type="missing"
                    ),
                    json_text="",
                    formatted_text=formatted_text,
                    original_text=original_text,
                    sentences=[],
                    annotations=[],
                    relations=[]
                )
                passage_json.update(passage_txt)
                new_passages.append(passage_json)
            annots_json = [i for i in new_passages if i.get("annotations")]
            n_json = [j for i in annots_json for j in i.get("annotations")]
        if len(annotations) != len(n_json):
            raise AssertionError(f"\n# Original_annotations: {len(annotations)}\n# Annotations in JSON: {len(n_json)}")
        with open(Path.joinpath(ANNOT_DIR, f"{source_id}.json"), "w") as f:
            article_json[0]["documents"][0]["passages"] = new_passages
            json.dump(article_json, f, indent=4)
        # Write safely to shared file
        with lock:
            with open(complete_path, "a") as f:
                f.write(f"{source_id}\n")
        torch.cuda.empty_cache()
        processed.append(source_id)
        pbar.update(1)
    pbar.close()
    return processed


def main(source_ids, complete_path):
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No GPUs available.")
    print(f"Found {num_gpus} GPUs. Distributing {len(source_ids)} articles...")
    
    # Evenly distribute articles across GPUs
    source_ids_split = [[] for _ in range(num_gpus)]
    start_indices = [0] * num_gpus
    for idx, source_id in enumerate(source_ids):
        gpu_id = idx % num_gpus
        if len(source_ids_split[gpu_id]) == 0:
            start_indices[gpu_id] = idx
        source_ids_split[gpu_id].append(source_id)
    
    # Start a process per GPU
    with mp.Manager() as manager:
        lock = manager.Lock()
        processes = []
        for gpu_id in range(num_gpus):
            p = mp.Process(
                target=save_annotations_json,
                args=(gpu_id, source_ids_split[gpu_id], start_indices[gpu_id], complete_path, lock)
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()



if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)  # Ensures compatibility across OSes
    source_ids = sorted(set(re.match("[0-9]{8}", i.stem).group() for i in CRAFT_DIR.rglob("*") if (
        i.is_file() and # look only for files
        i.suffix.lower() == ".txt" and # file should be of '.txt' extension
        re.match("[0-9]{8}", i.stem) # files are saved with 8 digit number as filename
    )))
    print(f"{len(source_ids)} articles found in JSON format.")
    print("Adding annotation from xml files to json...")
    complete_path = Path.joinpath(ANNOT_DIR, "completed")
    if complete_path.exists():
        with open(complete_path, "r") as f:
            completed = [i.strip() for i in f.readlines()]
    else:
        completed = []
        with open(complete_path, "w") as f:
            for i in completed:
                f.write("\n")
    if len(completed):
        print(f"{len(completed)} json files already exists with annotations added, skipping those files...")
    main(list(set(source_ids) - set(completed)), complete_path)
    print("Done!")
