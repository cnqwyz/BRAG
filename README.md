# BRAG: Backbone-Aware Graph for Polymer Property Prediction

A graph neural network framework for predicting polymer glass transition temperature (Tg), featuring role-aware inductive bias to separate backbone and side-chain representations.

## Installation

```bash
pip install -r requirements.txt
```

**Requirements:**
- Python >= 3.8
- PyTorch >= 2.0.0
- PyTorch Geometric >= 2.3.0
- RDKit >= 2023.3.1

## Project Structure

```
BRAG2.0/
├── models/              # Model architectures
│   ├── brag.py         # Main BRAG model
│   ├── baselines.py    # Baseline models (VanillaGNN, AtomAttentionGNN)
│   └── brag_ablations.py  # Ablation variants
├── data/               # Dataset and preprocessing
│   ├── dataset.py      # PolymerDataset
│   └── openpoly.csv    # Polymer Tg dataset
├── pooling/            # Role-aware pooling methods
├── loss/               # Loss functions (MSE, contrastive)
├── experiments/        # Experiment scripts
│   └── table_generator.py  # Generate paper tables
├── train/              # Training scripts
├── scripts/            # Utility scripts
└── checkpoints/        # Trained model weights
```

## Quick Start

### 1. Generate Predictions

Train BRAG and generate predictions:

```bash
python scripts/generate_predictions.py
```

### 2. Generate Experiment Tables

Reproduce paper results (Tables 1-6):

```bash
# Main comparison tables
python experiments/table_generator.py --table 1 --epochs 200 --seeds 10

# All tables
for $t in 1 2 3 4 5 6 { python experiments/table_generator.py --table $t --epochs 200 --seeds 10 }
```

Available tables:
| Table | Description |
|-------|-------------|
| 1 | Main model vs baselines |
| 2 | High-Tg bias analysis |
| 3 | Ablation study |
| 4 | Pooling/interaction sensitivity |
| 5 | Aggregation methods |
| 6 | Contrastive learning |
| 101 | Dataset statistics |
| 102 | Hyperparameters |

## Models

| Model | Description |
|-------|-------------|
| **BRAG** | Main model with backbone/side-chain interaction |
| VanillaGNN | Standard GNN baseline |
| AtomAttentionGNN | Node-level attention baseline |
| BRAGOnlyBackbone | Ablation: backbone only |
| BRAGOnlySidechain | Ablation: side-chain only |
| ContrastiveRoleGNN | Contrastive learning variant |

## Usage Example

```python
from models.brag import BRAG
from models.gnn_backbone import GNNEncoder
from data.dataset import PolymerDataset

# Load dataset
dataset = PolymerDataset(root='data/', csv_path='data/openpoly.csv', target_column='Tg_K')

# Initialize model
encoder = GNNEncoder(in_dim=..., hidden_dim=128, num_layers=3, gnn_type="gcn")
model = BRAG(encoder, hidden_dim=128, pool="mean", interaction="abs_diff")

# Train and predict
# See scripts/generate_predictions.py for full training loop
```

## Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--epochs` | 300 | Training epochs |
| `--seeds` | 10 | Number of random seeds |
| `--device` | cuda | Device (cuda/cpu) |
| `--hidden_dim` | 128 | Hidden dimension |
| `--batch_size` | 32 | Batch size |

## Results

Results are saved to `results/` directory as JSON files.

## License

MIT License
