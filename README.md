# MolBridge: Atom-Level Joint Graph Refinement for Robust Drug-Drug Interaction Prediction

## Introduction

Drug combinations offer therapeutic benefits but also carry the risk of adverse drug–drug interactions (DDIs), especially under complex molecular structures. Accurate DDI event prediction requires capturing fine-grained inter-drug relationships, which are critical for modeling metabolic mechanisms such as _enzyme-mediated competition_. However, existing models typically rely on isolated drug representations and are limited to modeling atom-level cross-molecular interactions explicitly. To address this gap, we propose MolBridge, a novel atom-level joint graph refinement framework for robust DDI event prediction. MolBridge constructs a unified graph that integrates atomic structures of drug pairs, enabling direct modeling of inter-drug associations. To further capture long-range association while mitigating over-smoothing, we introduce a graph residual network that iteratively refines node features and preserves global structural context. This joint design allows MolBridge to effectively learn both local and global interaction patterns, yielding robust representations across both frequent and rare DDI types. Extensive experiments on two benchmark datasets show that MolBridge consistently outperforms state-of-the-art baselines, with at least 2.19\% and 2.49\% improvements in Macro-F1 and Macro-Recall, respectively. These results demonstrate the advantages of fine-grained graph refinement in improving the accuracy, robustness, and mechanistic interpretability of DDI prediction.

Paper link: XXXXXXXXX

![Framework](./Supplymentary/Framework.png)


## Installation and Dependencies

### Requirements

- Python 3.9
- PyTorch 2.4.0 (with CUDA 11.8 support)
- PyTorch Geometric 2.6.1
- RDKit 2024.9.6
- scikit-learn 1.6.1
- numpy
- pandas
- tqdm
- PyYAML
- matplotlib (for visualization)

### Setup

```bash
# Clone the repository

# Create and activate virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install PyTorch with CUDA support
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu118

# Install PyTorch Geometric and its dependencies
pip install torch-geometric==2.6.1
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.4.0+cu118.html

# Install other dependencies
pip install rdkit scikit-learn numpy pandas tqdm pyyaml matplotlib
```

## Usage

### Model Configuration

The model parameters can be configured in the `config.yml` file:

```yaml
# Model selection
model_name: 'MolBridge'

# Experiment name
experiment_name: '3d2-num3'

# Cross-validation settings
folds: [0, 1, 2, 3, 4]

# Device configuration
device: 'cuda:0'

# Training parameters
train:
  batch_size: 512
  lr: 0.005
  epochs: 500
  loss_fn: 'CrossEntropyLoss'
  optimizer:
    name: 'AdamW'
    weight_decay: 0.01
  dropout: 0.1
  num_workers: 4
  seed: 42

# Model architecture parameters
model:
  block:
    num: 3
    ffn:
      hidden_dim: 2048
  mlp:
    hidden_dim: 256
  attn_heads: 4
  is_a_: true
  is_rg: true
  alpha: 0.0
  is_alpha_learn: true
  is_eye: true
  is_joint: true

# Data parameters
data:
  source: "Deng"
  num_classes: 65
  atom_dim: 75
  atom_hid_dim: 256
  clip_num_atom: 50
```

### Data Preparation

The model expects data in CSV format with drug SMILES strings and interaction labels:

```python
# Dataset format example
"""
smile_1,smile_2,label
CC1=CC=C(C=C1)C(=O)OCCN(C)C,CC(=O)C1=CC=C(N)C=C1,23
CC(C)CC1=CC=C(C=C1)C(C)C(=O)O,CCN(CC)CC,7
...
"""
```

### Training the Model

```python
# Run training with specified configuration
python main.py
```

The training process includes:

1. Loading and preprocessing molecular data
2. Converting SMILES to graph representations
3. Training the GRN-DDI model with cross-validation
4. Evaluating model performance on validation and test sets

### Using the Trained Model for Prediction

```python
import torch
from src.models.mymodel import MyModel
from src.datasets.dataloader.feature_encoding import smile_to_graph

# Load trained model
model_path = "best_model.pth"
model = MyModel()
model.load_state_dict(torch.load(model_path, weights_only=True))
model.eval()

# Prepare drug pair data
def prepare_drug_pair(smile1, smile2, device):
    # Convert SMILES to graph representations
    # (Implementation details as per src/datasets/dataloader/collate_fn.py)
    # ...
    return embeds, adjs, masks, cnn_masks

# Make prediction
drug1_smiles = "CC1=CC=C(C=C1)C(=O)OCCN(C)C"
drug2_smiles = "CC(=O)C1=CC=C(N)C=C1"
inputs = prepare_drug_pair(drug1_smiles, drug2_smiles, device="cuda:0")
with torch.no_grad():
    scores, _ = model(*inputs, torch.zeros(1))
    predicted_class = torch.argmax(scores, dim=1).item()
    print(f"Predicted interaction class: {predicted_class}")
```

## Evaluation Metrics

The model's performance is evaluated using the following metrics:

1. **Accuracy**: The proportion of correctly predicted interactions.

   ```
   accuracy = correct_predictions / total_predictions
   ```

2. **Macro Precision**: The average precision across all classes.

   ```
   macro_precision = sum(precision_per_class) / num_classes
   ```

3. **Macro Recall**: The average recall across all classes.

   ```
   macro_recall = sum(recall_per_class) / num_classes
   ```

4. **Macro F1 Score**: The harmonic mean of precision and recall, averaged across all classes.

   ```
   macro_f1 = 2 * (macro_precision * macro_recall) / (macro_precision + macro_recall)
   ```

These metrics are computed for both validation and test sets during model training, with the best model checkpoint saved based on combined F1 score and accuracy performance.

## Citation

If you use this code in your research, please cite:

```
XXXXXXXX
```
 
## Atom Feature Encoding
 
- Purpose: provide the complete atom feature schema for molecular graphs.
- Feature vector per atom has 75 dimensions composed of:
  - Atomic symbol (44): one-hot over [C, N, O, S, F, Si, P, Cl, Br, Mg, Na, Ca, Fe, As, Al, I, B, V, K, Tl, Yb, Sb, Sn, Ag, Pd, Co, Se, Ti, Zn, H, Li, Ge, Cu, Au, Ni, Cd, In, Mn, Zr, Cr, Pt, Hg, Pb, Unknown]
  - Atomic degree (11): one-hot over [0–10]
  - Implicit valence (7): one-hot over [0–6]
  - Formal charge (1): integer scalar (typically −3 to +3)
  - Radical electrons (1): integer scalar (typically 0–2)
  - Hybridization (5): one-hot over [SP, SP2, SP3, SP3D, SP3D2]
  - Aromaticity (1): binary indicator (0/1)
  - Total hydrogens (5): one-hot over [0–4]
 
## Implementation Details
 
- Baselines: DeepDDI, R-GCN, GoGNN, TrimNet‑DDI, SSI‑DDI, MUFFIN, MRCGNN（reported）；DSN‑DDI、CSSE‑DDI、TIGER（re‑implemented per official configs）。
- Data split and metrics: 5‑fold CV with 7:1:2 train/val/test；report Accuracy、Macro‑F1、Macro‑Recall、Macro‑Precision。
- Training config: batch size 512、learning rate 0.005、seed 42、3 GFormer layers；optimizer AdamW；early stopping on validation；average over folds。
- Environment: Intel Xeon Platinum‑8457C、NVIDIA L20 (48GB)、Ubuntu 22.04.1 LTS、PyTorch 2.4.0 (CUDA 11.8)、RDKit 2024.9.6。
 
## Ablation Study
 
- Variants on Deng’s dataset:
  - w/o SCM：replace Structure Consistency Module with standard GCN layers
  - w/o A'：remove chemical bond adjacency matrix
  - w/o Ar：remove attention‑based graph reconstruction
  - w/o Joint：no joint construction；model drugs independently then fuse
- Conclusion: all components contribute；SCM is most critical；joint construction and the combination of A' and Ar are both necessary。
- Figure: [Supplymentary/ablation/ablation_study_deng.pdf](./Supplymentary/ablation/ablation_study_deng.pdf)
 
## Hyperparameter Sensitivity Analysis
 
- Swept hyperparameters on Deng：learning rate、batch size、attention heads、SCM layers。
- Best configuration：lr=0.005、batch=512、heads=4、SCM layers=3。
- Figures：
  - Learning rate: [Supplymentary/hparam/lr.pdf](./Supplymentary/hparam/lr.pdf)
  - Batch size: [Supplymentary/hparam/Batch.pdf](./Supplymentary/hparam/Batch.pdf)
  - Attention heads: [Supplymentary/hparam/head.pdf](./Supplymentary/hparam/head.pdf)
  - SCM layers: [Supplymentary/hparam/num-of-GRU.pdf](./Supplymentary/hparam/num-of-GRU.pdf)

## Complete Inductive Evaluation (S1/S2)

This section provides the complete inductive evaluation results on DrugBank and TWOSIDES under the S1 and S2 settings. Methods marked with "*" use external knowledge graphs or heterogeneous networks.

### S1 Results (DrugBank and TWOSIDES)

| Methods     | DrugBank (S1) F1 | DrugBank Acc | DrugBank Kappa | TWOSIDES (S1) PR-AUC | TWOSIDES ROC-AUC | TWOSIDES Acc |
|-------------|------------------:|-------------:|---------------:|----------------------:|-----------------:|-------------:|
| MLP         | 21.1 ± 0.8        | 46.6 ± 2.1   | 33.4 ± 2.5     | 81.5 ± 1.5            | 81.2 ± 1.9       | 76.0 ± 2.1   |
| Similarity  | 43.0 ± 5.0        | 51.3 ± 3.5   | 44.8 ± 3.8     | 56.2 ± 0.5            | 55.7 ± 0.6       | 53.9 ± 0.4   |
| CSMDD       | 45.5 ± 1.8        | 62.6 ± 2.8   | 55.0 ± 3.2     | 73.2 ± 2.6            | 74.2 ± 2.9       | 69.9 ± 2.2   |
| STNN-DDI*   | 39.7 ± 1.8        | 56.7 ± 2.6   | 46.5 ± 3.4     | 68.9 ± 2.0            | 68.3 ± 2.6       | 65.3 ± 1.8   |
| HIN-DDI*    | 37.3 ± 2.9        | 58.9 ± 1.4   | 47.6 ± 1.8     | 81.9 ± 0.6            | 83.8 ± 0.9       | 79.3 ± 1.1   |
| MSTE*       | 7.0 ± 0.7         | 51.4 ± 1.8   | 37.4 ± 2.2     | 64.1 ± 1.1            | 62.3 ± 1.1       | 58.7 ± 0.7   |
| KG-DDI*     | 26.1 ± 0.9        | 46.7 ± 1.9   | 35.2 ± 2.5     | 79.1 ± 0.9            | 77.7 ± 1.0       | 60.2 ± 2.2   |
| CompGCN*    | 26.8 ± 2.2        | 48.7 ± 3.0   | 37.6 ± 2.8     | 80.3 ± 3.2            | 79.4 ± 4.0       | 71.4 ± 3.1   |
| Decagon*    | 24.3 ± 4.5        | 47.4 ± 4.9   | 35.8 ± 5.9     | 79.0 ± 2.0            | 78.5 ± 2.3       | 69.7 ± 2.4   |
| KGNN*       | 23.1 ± 3.4        | 51.4 ± 1.9   | 40.3 ± 2.7     | 78.5 ± 0.5            | 79.8 ± 0.6       | 72.3 ± 0.7   |
| SumGNN*     | 35.0 ± 4.3        | 48.8 ± 8.2   | 41.1 ± 4.7     | 80.3 ± 1.1            | 81.4 ± 1.0       | 73.0 ± 1.4   |
| DeepLGF*    | 39.7 ± 2.3        | 60.7 ± 2.4   | 51.0 ± 2.6     | 81.4 ± 2.1            | 82.2 ± 2.6       | 72.8 ± 2.8   |
| MolBridge   | 53.8 ± 5.0        | 64.4 ± 2.8   | 56.9 ± 3.7     | 66.6 ± 1.3            | 70.0 ± 1.3       | 63.5 ± 1.8   |

### S2 Results (DrugBank and TWOSIDES)

| Methods    | DrugBank (S2) F1 | DrugBank Acc | DrugBank Kappa | TWOSIDES (S2) PR-AUC | TWOSIDES ROC-AUC | TWOSIDES Acc |
|------------|------------------:|-------------:|---------------:|----------------------:|-----------------:|-------------:|
| CSMDD      | 19.8 ± 3.1        | 37.3 ± 4.8   | 22.0 ± 4.9     | 55.8 ± 4.9            | 57.0 ± 6.1       | 55.1 ± 5.2   |
| HIN-DDI*   | 8.8 ± 1.0         | 27.6 ± 2.4   | 13.8 ± 2.4     | 64.8 ± 2.3            | 58.5 ± 1.6       | 59.8 ± 1.4   |
| KG-DDI*    | 1.1 ± 0.1         | 32.2 ± 3.6   | --             | 53.9 ± 3.9            | 47.0 ± 5.5       | 50.0 ± 0.0   |
| DeepLGF*   | 4.8 ± 1.9         | 31.9 ± 3.7   | 8.2 ± 2.3      | 59.4 ± 8.7            | 54.7 ± 5.9       | 54.0 ± 6.2   |
| MolBridge  | 20.5 ± 1.3        | 40.3 ± 2.7   | 25.9 ± 3.0     | 56.5 ± 2.5            | 56.4 ± 3.9       | 54.5 ± 3.8   |

Note: Methods with "*" use external knowledge graphs or heterogeneous networks.

## Figure 9: t-SNE of 4 Most-Frequent Events (Deng)

We provide full-resolution t-SNE visualizations for the four most-frequent DDI events on the Deng dataset. Each method shows two rows: the upper row is the initial pairwise feature encoding, and the lower row is the learned embedding.

- TrimNet-DDI: [PDF](./Supplymentary/tsne/t_sne_TrimNet_duo_.pdf)
- SSI-DDI: [PDF](./Supplymentary/tsne/t_sne_SSI_duo_.pdf)
- MRCGNN: [PDF](./Supplymentary/tsne/t_sne_MRCGNN_duo_.pdf)
- DSN-DDI: [PDF](./Supplymentary/tsne/t_sne_DSN_duo_.pdf)
- MolBridge (Ours): [PDF](./Supplymentary/tsne/t_sne_2d_duo_.pdf)

An index is also provided at `code/Supplymentary/tsne/frequent/` for these frequent-event t-SNE assets.

## Full ATC Code Analysis (Clusters 0–6)

Complete ATC codes for the representative drugs in each identified cluster.

| Cluster | Drug    | ATC Code | Level 1 | Level 2 | Level 3 | Level 4 | Level 5 |
|--------:|---------|----------|---------|---------|---------|---------|---------|
| 0 | DB00559 | C02KX01 | C - Cardiovascular system | 02 - Antihypertensives | K - Other antihypertensives | X - Pulmonary hypertension drugs | 01 - Bosentan |
| 1 | DB04839 | G03HA01 | G - Genito-urinary system and sex hormones | 03 - Sex hormones and modulators of the genital system | H - Antiandrogens | A - Antiandrogens, plain | 01 - Cyproterone |
| 2 | DB09280 | R07AX30 | R - Respiratory system | 07 - Other respiratory system products | A - Other respiratory system drugs | X - Other respiratory system products | 30 - Ivacaftor/Lumacaftor |
| 3 | DB00648 | L01XX23 | L - Antineoplastic and immunomodulating agents | 01 - Antineoplastic agents | X - Other antineoplastic agents | X - Other antineoplastic agents | 23 - Mitotane |
| 4 | DB11901 | L02BB05 | L - Antineoplastic and immunomodulating agents | 02 - Endocrine therapy | B - Hormones and related agents | B - Antiandrogens | 05 - Enzalutamide |
| 4 | DB08899 | L02BB04 | L - Antineoplastic and immunomodulating agents | 02 - Endocrine therapy | B - Hormones and related agents | B - Antiandrogens | 04 - Enzalutamide |
| 5 | DB08912 | L01EC02 | L - Antineoplastic and immunomodulating agents | 01 - Antineoplastic agents | E - Protein kinase inhibitors | C - BRAF serine/threonine kinase inhibitors | 02 - Dabrafenib |
| 6 | DB01320 | N03AB05 | N - Nervous system | 03 - Antiepileptics | A - Antiepileptics | B - Hydantoin derivatives | 05 - Phenytoin |
| 6 | DB00564 | N03AF01 | N - Nervous system | 03 - Antiepileptics | A - Antiepileptics | F - Carboxamide derivatives | 01 - Carbamazepine |
