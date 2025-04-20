# 🧠 Prompts

This directory contains reference prompt templates used during the development of the ontology annotation pipeline. These are not used directly in the training scripts but serve as documentation for prompt engineering decisions.

## 📄 Files

- **initial_prompt.md**  
  Describes the baseline prompt template used before fine-tuning.

- **final_prompt.md**  
  Contains the refined prompt structure used after iterative testing and optimization. This prompt typically yields improved semantic performance and aligns closely with the task-specific fine-tuning objectives.
  ✨ **Note**:  
  - This file is formatted with line breaks and indentation for **human readability only**.  
  - In the actual training script, the prompt (including expected response format) is collapsed into a **single-line string** with no extra spacing or indentation.  
  - This distinction helps ensure clarity during development while maintaining compatibility with script-based training pipelines.

## 📌 Purpose

These prompt files are intended for:
- Quick reference and documentation
- Comparing prompt versions and structures
- Understanding how prompting evolved during model development

They are **not** directly imported or parsed by the training scripts.

---
