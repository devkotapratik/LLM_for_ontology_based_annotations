# 📚 Optimizing Large Language Models for Ontology-Based Annotation: A Study on Gene Ontology in Biomedical Texts

This repository contains the code, datasets, and evaluation scripts used in our study: **"Leveraging Large Language Models for Automated Ontology Annotation of Biomedical Text"**.

## 🧠 Abstract

Automated ontology annotation of scientific literature plays a crucial role in knowledge management, especially in biology and biomedicine. Traditional methods like Recurrent Neural Networks (RNNs) and Bidirectional Gated Recurrent Units (Bi-GRUs) have shown effectiveness but struggle with complex biomedical terminology and semantics.

In this project, we explore the potential of fine-tuned Large Language Models (LLMs) — including **MPT-7B**, **Phi**, **BiomedLM**, and **Meditron** — for annotating scientific literature with **Gene Ontology (GO)** concepts. We benchmark these models on the **CRAFT dataset**, evaluating their performance using metrics such as:

- F1 Score  
- Semantic Similarity  
- Memory Usage  
- Inference Speed  

We also investigate techniques such as **Parameter-Efficient Fine-Tuning (PEFT)** and **advanced prompting** to address computational efficiency.


## 🗂️ Project Structure
```
.
├── configs/              # Fine-tuning configurations for LLMs
├── data/                 # Preprocessed CRAFT dataset and ontology mappings
├── model_output/         # Saved model checkpoints, logs, and generated annotations
├── notebooks/            # Jupyter notebooks for exploratory analysis
├── results/              # Evaluation results, metrics, and figures
├── scripts/              # Preprocessing, training, evaluation, and inference scripts
├── utils/                # Helper functions for parsing, evaluation, and visualization
├── LICENSE               # Project license information
├── README.md             # Project documentation
└── requirements.txt      # Python dependencies
```

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/devkotapratik/LLM_for_ontology_based_annotations.git
cd LLM_for_ontology_based_annotations
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the CRAFT dataset
Follow the instructions in ```data/README.md``` to download and preprocess the dataset.

### 4. Fine-tune a model
Example for fine-tuning BiomedLM:

```bash
python scripts/train.py --model biomedlm --dataset craft --epochs 5
```

### 5. Run evaluation
```bash
python scripts/evaluate.py --model biomedlm --metrics all
```

## ⚙️ Features

* Fine-tuning scripts for multiple LLMs
* Support for parameter-efficient fine-tuning (LoRA/PEFT)
* Ontology-aware semantic evaluation
* Integration with Hugging Face Transformers and Datasets

## 📄 License
MIT License. See ```LICENSE``` file for details.