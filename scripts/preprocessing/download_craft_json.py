import re
import sys
import json
import shutil
import zipfile
import aiohttp
import asyncio
import requests
import argparse
import numpy as np
import pandas as pd

from git import Repo
from pathlib import Path
from tqdm.auto import tqdm
from bs4 import BeautifulSoup


async def fetch_and_save(session, index, url, filename, filetype):
    try:
        async with session.get(url) as response:
            if response.status == 200:
                content = await response.read()
                try:
                    if not str(filename).endswith(filetype):
                        filename = f"{filename}.{filetype}"
                    if filetype == "json":
                        content = json.loads(content.decode("utf-8"))
                        with open(filename, "w") as f:
                            json.dump(content, f, indent=4)
                    elif filetype == "csv":
                        with open(filename, "wb") as f:
                            f.write(content)
                except json.JSONDecodeError:
                    print(f"Error decoding JSON from index: {index}")
                except Exception as ex:
                    print(f"Error saving response: {ex}")
    except aiohttp.ClientError as client_err:
        print(f"Error fetching {url}: {client_err}")


async def download_multiple_files(data: list, filetype: str, progress_description: str, progress_unit: str):
    async with aiohttp.ClientSession() as session:
        tasks = []
        pbar = tqdm(
            total=len(data), unit=progress_unit,
            desc=progress_description)
        for index, (filename, file_url) in enumerate(data):
            # fetch_task = fetch_url(session, file_url)
            task = await fetch_and_save(session, index, file_url, filename, filetype)
            # save_task = save_response(await fetch_task, index, filename, filetype)
            tasks.append(task)
            pbar.update(1)
        # await asyncio.gather(*tasks)
        pbar.close()


def main(args):
    DATA_DIR = Path(args.data_dir).resolve()
    DATA_DIR.mkdir(exist_ok=True, parents=True)
    CRAFT_DIR = Path(args.craft_dir) if args.craft_dir else DATA_DIR.joinpath("CRAFT")
    CRAFT_DIR.mkdir(exist_ok=True, parents=True)
    bioc_url = "https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_{format}/{ID}/{encoding}"
    craft_url = "UCDenver-ccp/CRAFT"
    latest_release = f"https://api.github.com/repos/{craft_url}/releases/latest"

    ### Get CRAFT latest version number from github
    print(f"Getting latest version id from github: {craft_url} ...")
    response = requests.get(latest_release)
    json_data = response.json()
    latest_version = json_data.get("tag_name", "")
    print(f"Latest version on github: {latest_version}.")

    zip_file_url = json_data.get("zipball_url")
    zip_file_path = CRAFT_DIR.joinpath(f"{zip_file_url.split('/')[-1]}.zip")
    extract_path = zip_file_path.with_suffix('')
    if len(list(extract_path.glob("*"))) and args.replace_if_exists:
        shutil.rmtree(extract_path)
    response = requests.get(zip_file_url, stream=True)
    total = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 KB

    with open(zip_file_path, 'wb') as file, tqdm(desc=f"Downloading CRAFT repo: {zip_file_path}",
        total=total, unit='iB', unit_scale=True, unit_divisor=1024) as bar:
        for data in response.iter_content(block_size):
            file.write(data)
            bar.update(len(data))

    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        pbar = tqdm(total=len(zip_ref.infolist()), desc="Extracting files:")
        for member in zip_ref.infolist():
            path_parts = Path(member.filename).parts
            pbar.set_description(f"Extracting file: {Path(*path_parts[1:])}")
            pbar.update(1)
            if len(path_parts) < 2: continue
            # Reconstruct path, dropping the first part (top-level dir)
            relative_path = Path(*path_parts[1:])
            target_path = Path(extract_path) / relative_path
            # Make sure target directory exists
            if member.is_dir():
                target_path.mkdir(parents=True, exist_ok=True)
            else:
                target_path.parent.mkdir(parents=True, exist_ok=True)
                with zip_ref.open(member) as source, open(target_path, "wb") as target:
                    target.write(source.read())
        pbar.close()

    zip_file_path.unlink()
    CRAFT_DIR = CRAFT_DIR.joinpath(latest_version)
    all_files = [i for i in CRAFT_DIR.rglob("*") if not i.stem.startswith(".")]
    keep = "(articles/txt)|(concept\-annotation/GO_BP)|(concept\-annotation/GO_CC)|(concept\-annotation/GO_MF)"
    reqd_files = [i for i in all_files if i.suffix.lower() in [".txt", ".xml", ".md"] and bool(re.search(keep, str(i)))]
    not_reqd = sorted(set(all_files) - set(reqd_files), key=lambda x: len(x.parts), reverse=True)
    print(f"Removing unnecessary files and directories...")
    test = [i.unlink() if i.is_file() else (i.rmdir() if not len(list(i.iterdir())) else None) for i in sorted(
            not_reqd, key=lambda x: len(x.parts), reverse=True)]
    ##### Get a list of all article names
    # Each article and its corresponding annotations are identified by a unique 8 digit number
    print(f"Getting the PMIDs of article for json download...")
    source_ids = sorted(set(re.match("[0-9]{8}", i.stem).group() for i in CRAFT_DIR.rglob("*") if (
        i.is_file() and # look only for files
        i.suffix.lower() == ".txt" and # file should be of '.txt' extension
        re.match("[0-9]{8}", i.stem) # files are saved with 8 digit number as filename
    )))
    print(f"{len(source_ids)} articles found. Downloading json files...")

    JSON_DIR = CRAFT_DIR.joinpath("articles", "json")
    JSON_DIR.mkdir(exist_ok=True, parents=True)
    bioc_urls = [(
        Path.joinpath(JSON_DIR, f"{pmid}.json"), bioc_url.format(format="json", ID=pmid, encoding="unicode")
    ) for pmid in source_ids]

    asyncio.run(download_multiple_files(
        data=bioc_urls,
        filetype="json",
        progress_description="Downloading Pubmed articles in JSON format...",
        progress_unit="JSON"
    ))
    print(f"{len(source_ids)} articles downloaded in json format. Download directory: {JSON_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download CRAFT dataset from github.")
    parser.add_argument('--data_dir', type=str, required=True, help='Path to download data to.')
    parser.add_argument('--craft_dir', type=str, required=False, help='Path to download CRAFT dataset.')
    parser.add_argument('--replace_if_exists', action='store_true', help="Enable this to replace existing files.")
    args = parser.parse_args()
    main(args)
    print("Done!")