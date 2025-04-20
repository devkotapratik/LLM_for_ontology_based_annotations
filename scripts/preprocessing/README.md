# 📥 CRAFT Dataset Downloader and Preprocessor

This Python script automates downloading the latest release of the [CRAFT Corpus](https://github.com/UCDenver-ccp/CRAFT), extracts the necessary files, and fetches corresponding PubMed articles in BioC JSON format using the NCBI BioNLP RESTful API.

---

## 🚀 Features

- 📌 Automatically fetches the **latest version** of the CRAFT dataset from GitHub.
- 📦 Downloads and **extracts** only relevant files:
  - Article `.txt` files
  - GO annotations from `GO_BP`, `GO_CC`, and `GO_MF`
- 🧹 **Removes unnecessary files and directories**
- 🌐 Downloads **PubMed articles** in BioC JSON format using extracted PMIDs
- 🗂 Organizes outputs in a clean directory structure

---

## 📂 Directory Structure
```
├── data/
|   └── CRAFT/
|       └── vX.Y.Z/                   # Latest version from GitHub
|           ├── articles/
|           |   ├── json/             # BioC JSON articles
|           |   └── txt/              # Raw article texts
│           ├── concept-annotation/
|           |   ├── GO_BP/
|           |   ├── GO_CC/
|           |   └── GO_MF/
|           └── .gitignore
└── ...
```

---

## ⚙️ Usage

### Command

```bash
python download_craft.py --data_dir /path/to/output [--craft_dir /custom/craft/path] [--replace_if_exists]
```

### Arguments

| Argument             | Required | Type | Description                                                                 |
|----------------------|----------|------|-----------------------------------------------------------------------------|
| `--data_dir`         | ✅ Yes   | str  | Path to the main directory where all data (CRAFT and articles) will be saved. |
| `--craft_dir`        | ❌ No    | str  | Optional path to save the CRAFT dataset. Defaults to `<data_dir>/CRAFT`.     |
| `--replace_if_exists`| ❌ No    | flag | If set, replaces existing downloaded and extracted CRAFT data.               |


## 🧪 What the Script Does

1. 📡 Fetches metadata of the latest CRAFT release from GitHub.

2. ⬇️ Downloads the zipball of the release.
3. 📂 Extracts and filters only the needed files.
4. 🧬 Identifies PMIDs from article .txt filenames.
5. 🌍 Downloads the corresponding articles from NCBI in BioC JSON format.
6. ✅ Stores everything in a neatly organized structure.