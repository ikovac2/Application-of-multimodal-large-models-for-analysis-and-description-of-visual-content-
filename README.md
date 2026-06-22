# Multimodal Large Language Models for Leather Defect Detection

**Course:** Introduction to Digital Image Processing (UDOS)  
**Author:** Iman Kovač  
**Institution:** Faculty of Electrical Engineering, University of Sarajevo  
**Year:** 2026

---

## Overview

This project investigates the application of multimodal large language 
models (MLLMs) for automated defect detection and description on tanned 
leather surfaces. Three open-source models are evaluated in a zero-shot 
setting: CLIP (contrastive classification), Moondream (lightweight VQA), 
and LLaVA-7B (generative VQA).

The evaluation is performed on a subset of 70 manually annotated samples 
from a custom leather defect dataset developed as part of a prior 
Bachelor's thesis [see Dataset section below].

---

## Results Summary

| Model | Accuracy | Precision | Recall | F1 | Avg. time/image |
|-------|----------|-----------|--------|----|-----------------|
| CLIP ViT-B/32 | 97.1% | 97.1% | 100.0% | 98.6% | ~0.5s |
| Moondream | 12.9% | 100.0% | 10.3% | 18.7% | ~5s |
| LLaVA-7B-v1.6 | 17.1% | 100.0% | 14.7% | 25.6% | ~64s |

Key finding: CLIP significantly outperforms both generative models across 
all metrics, while being over 100x faster. Generative models fail 
primarily due to domain gap — they were not trained on industrial leather 
inspection data.

---

## Repository Structure
├── notebook/

│   └── leather_defect_analysis.ipynb   # Main evaluation notebook

├── gui/

│   └── leather_defect_gui.py           # Desktop GUI demo application

├── results/

│   ├── labels_ground_truth.csv         # Ground truth annotations

│   ├── clip_results.csv                # CLIP predictions

│   ├── llava_moondream_results.csv     # Moondream predictions

│   ├── llava7b_results.csv             # LLaVA-7B predictions

│   ├── full_analysis.csv               # Combined results

│   └── final_results_table.csv         # Summary metrics table

├── figures/

│   ├── clip_confusion_matrix.png

│   ├── clip_distribution.png

│   ├── model_comparison.png

│   └── final_comparison_3models.png

├── requirements.txt

└── README.md

---

## Dataset

This project uses a subset of the leather defect database developed in:

> I. Kovač, "Integrated-Database-for-Systematic-Analysis-of-Defects-in-Tanned-Leather," B.S. thesis, Faculty of Electrical 
> Engineering, University of Sarajevo, 2025.

**Dataset repository:** [Integrated-Database-for-Systematic-Analysis-of-Defects-in-Tanned-Leather](https://github.com/ikovac2/Integrated-Database-for-Systematic-Analysis-of-Defects-in-Tanned-Leather)

The evaluation subset consists of:
- 50 samples of **Bold** leather type
- 20 samples of **Elba** leather type
- 70 samples total, all manually annotated using the 
  [DefectDetect](https://github.com/arapov1c/DefectDetect-Application) 
  annotation tool

---

## Setup and Usage

### Prerequisites

- Python 3.9+
- [Anaconda](https://www.anaconda.com/download) (recommended)
- [Ollama](https://ollama.com/download) (for Moondream and LLaVA models)

### Installation

```bash
# 1. Clone this repository
git clone https://github.com/ikovac2/Application-of-multimodal-large-models-for-analysis-and-description-of-visual-content-.git
cd Application-of-multimodal-large-models-for-analysis-and-description-of-visual-content-

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Pull Ollama models
ollama pull moondream
ollama pull llava:7b-v1.6
```

### Running the Notebook

Open `notebook/leather_defect_analysis.ipynb` in Jupyter and run cells 
sequentially. Make sure Ollama is running in the background before 
executing LLaVA/Moondream cells.

### Running the GUI Demo

```bash
python gui/leather_defect_gui.py
```

Load any leather sample image and click the model buttons to compare 
outputs side by side.

---

## Models Used

| Model | Source | License |
|-------|--------|---------|
| CLIP ViT-B/32 | [OpenAI](https://github.com/openai/CLIP) | MIT |
| Moondream | [Ollama](https://ollama.com/library/moondream) | Apache 2.0 |
| LLaVA-7B-v1.6 | [Ollama](https://ollama.com/library/llava) | Apache 2.0 |

---

## References

This project is based on the following key works:

- CLIP: Radford et al., [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020), ICML 2021
- LLaVA: Liu et al., [Visual Instruction Tuning](https://arxiv.org/abs/2304.08485), NeurIPS 2023
- LLaVA-1.5: Liu et al., [Improved Baselines with Visual Instruction Tuning](https://arxiv.org/abs/2310.03744), CVPR 2024

