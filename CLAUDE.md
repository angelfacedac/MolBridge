# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MolBridge is an atom-level joint graph refinement framework for robust Drug-Drug Interaction (DDI) event prediction. The project implements a novel approach that constructs unified graphs integrating atomic structures of drug pairs to model inter-drug associations for DDI prediction.

## Core Architecture

### Main Components

- **Models** (`src/models/`): Core neural network implementations
  - `mymodel.py`: Main MolBridge model with both standard and PyG variants
  - `mylayers.py`: Custom graph neural network layers (GRU, GAT, GIN)
  - `GCN.py`: Graph Convolutional Network implementation
  - `myloss.py`: Custom loss functions

- **Data Pipeline** (`src/datasets/`): Molecular data processing
  - `dataset.py`: Basic dataset wrapper for CSV files
  - `get_dataloader.py`: DataLoader creation with fold-based cross-validation
  - `dataloader/feature_encoding.py`: SMILES to graph conversion using RDKit
  - `dataloader/collate_fn.py`: Batch processing with molecular graph caching
  - `dataloader/utils.py`: Graph normalization utilities

- **Experiments** (`src/experiments/`): Training and evaluation
  - `train.py`: Training loop implementation
  - `valida.py`: Validation evaluation
  - `test.py`: Test set evaluation
  - `move_data_to_device.py`: GPU memory management

- **Utils** (`src/utils/`): Supporting functionality
  - `manager.py`: Experiment management with TensorBoard logging
  - `metrics.py`: Evaluation metrics (accuracy, macro F1, precision, recall)
  - `backup.py`: Project backup utilities
  - `set_seed.py`: Reproducibility utilities

### Data Structure

The project supports two main datasets:
- **Deng**: 65 DDI classes
- **Ryu**: 86 DDI classes

Data is organized in cross-validation folds (0-4) with train/val/test splits. Each dataset contains:
- Drug pairs with SMILES representations
- DDI event labels
- Molecular graph features (75-dimensional atom features)

## Development Commands

### Running Experiments

```bash
# Main training command
python main.py

# Specific analyses
python ablation_study.py    # Ablation study experiments
python CaseStudy.py         # Case study analysis
python cluster_analyse.py   # Clustering analysis
python Long-dis.py         # Long-distance interaction analysis
```

### Configuration

Edit `config.yml` to modify:
- Model architecture parameters (num blocks, hidden dimensions, attention heads)
- Training hyperparameters (batch size, learning rate, epochs)
- Dataset selection ('Deng' or 'Ryu')
- Cross-validation folds
- Device configuration

Key configuration options:
```yaml
model_name: 'MulBridge-woJoint&RG'  # Model variant
data.source: "Ryu"                   # Dataset selection
data.num_classes: 86                 # Must match dataset
train.batch_size: 1024              # Training batch size
model.block.num: 4                   # Number of graph blocks
model.is_joint: true                 # Enable joint processing
```

### Model Variants

The codebase supports multiple model configurations controlled by config flags:
- `is_joint`: Enable/disable joint graph processing
- `is_rg`: Enable/disable residual connections
- `is_gcn/is_gat/is_gin`: Switch between GCN, GAT, or GIN layers
- `is_pyg`: Use PyTorch Geometric implementation

## Key Technical Details

### Molecular Processing
- SMILES strings are converted to molecular graphs using RDKit
- Canonical SMILES normalization ensures consistent representation
- Atom features include atomic properties (75-dimensional vectors)
- Graphs are clipped to maximum 50 atoms for consistent batch processing

### Training Process
- Cross-validation with configurable folds
- AdamW optimizer with weight decay
- CrossEntropyLoss for multi-class classification
- Early stopping based on validation performance
- Model checkpointing and TensorBoard logging

### Memory Optimization
- Molecular graph caching in `collate_fn.py` to avoid recomputation
- Efficient batch processing with padding masks
- GPU memory management in data loading

### Evaluation
- Metrics: Accuracy, Macro F1, Macro Precision, Macro Recall
- Performance tracking across training/validation/test sets
- Best model selection based on combined F1 and accuracy

## Data Dependencies

- **RDKit**: Molecular processing and SMILES parsing
- **PyTorch Geometric**: Graph neural network operations
- **scikit-learn**: Evaluation metrics
- **pandas**: Data manipulation
- **PyYAML**: Configuration management

## Important Notes

- The project uses deterministic seeding for reproducibility
- Experiments automatically backup the codebase before running
- TensorBoard logs are saved in `logs/{dataset}/{model}/{experiment}/`
- Model checkpoints include both standard PyTorch and PyG variants
- Data preprocessing includes atom ordering normalization for consistency