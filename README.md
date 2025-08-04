# TSS-Align: Temporal-Semantic-Static Alignment for Functional Brain Network Analysis

This repository provides the official implementation of our paper:

**TSS-Align: Temporal-Semantic-Static Alignment for Functional Brain Network Analysis**  
[Paper PDF](#) | [Project Page](#) (if applicable)

**Authors:** [Your names]  
**Affiliation:** [Your institution or lab name]

---

## 🧠 Overview

**TSS-Align** is a unified framework that models brain functional networks by aligning static (sFC), dynamic (dFC), and semantic (knowledge-driven) connectivity patterns for neuroimaging-based disease classification. It consists of:

### Key Components:
1. **Network Construction**  
   Builds three graphs from fMRI and textual knowledge:
   - Static sFC graph via Pearson correlation.
   - Dynamic dFC graphs via sliding window and GAT.
   - Semantic graph via medical-language modeling (BERT + GPT-4o + RAG).
2. **Shallow Alignment**  
   Injects semantic priors into sFC and dFC representations.
3. **Deep Alignment**  
   Uses prototype-driven contrastive alignment to refine multi-view consistency.
4. **Classification**  
   Performs downstream disease prediction based on aligned embeddings.

<p align="center"> 
  <img src="assets/overview.png" width="800"/> 
</p>

---

## 🔧 Installation

The codebase is implemented using **PyTorch**.

### Dependencies
Install required packages:
```bash
pip install -r requirements.txt
```

Key packages:
- `torch`
- `transformers`
- `scikit-learn`
- `numpy`
- `matplotlib`

---

## 📁 Project Structure

```bash
TSSA-Net/
├── data/                 # Dataset files (.mat or .npz) and semantic resources
├── models/               # Core model components (GAT, Transformers, DEC, etc.)
├── utils/                # Data loader, metrics, visualization, etc.
├── scripts/              # Training and evaluation scripts
├── configs/              # Configuration files
├── assets/               # Diagrams, model figures
├── Information_fusion.pdf  # Final version of the paper
├── main.py               # Entry point
└── README.md
```

---

## 📊 Dataset

We support **ABIDE**, **ADHD-200**, and custom datasets. For ABIDE:
1. Download preprocessed fMRI data from [here](#).
2. Use the **AAL atlas** to extract ROI time series.
3. Semantic knowledge comes from curated RAG-based descriptions (details in paper Section 3.1).

---

## 🚀 Training & Evaluation

### 1. Preprocess Data
Ensure your dataset is formatted as ROI × Time matrix (.mat or .npz), e.g.:
```python
X: shape [L, N]  # L time points, N brain regions
```

### 2. Train the Model
```bash
python main.py --config configs/abide.yaml
```

### 3. Evaluate
```bash
python main.py --mode eval --checkpoint path_to_model.pth
```

---

## 🧩 Key Components
| Module              | Description                                        |
|---------------------|----------------------------------------------------|
| **NetworkConstructor** | Builds sFC, dFC, and semantic graphs             |
| **ShallowAligner**     | Adds semantic positional embeddings               |
| **DeepAligner**        | DEC-based prototype clustering + contrastive loss |
| **Classifier**         | Fully connected or MLP head for prediction        |

---

## 📝 Citation

If you find this work helpful, please cite:
```bibtex
@article{your2024tssalign,
  title={TSS-Align: Temporal-Semantic-Static Alignment for Functional Brain Network Analysis},
  author={Your, Authors},
  journal={Information Fusion},
  year={2024}
}
```

---

## 📬 Contact

For any questions, feel free to open an issue or contact [your-email@domain.com](mailto:your-email@domain.com).

---
Feel free to suggest improvements or report issues!
