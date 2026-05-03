"""
PyTorch Geometric Dataset for polymers with role information.
"""

import torch
import pandas as pd
from typing import Optional

from torch_geometric.data import Data, Dataset, InMemoryDataset
from torch_geometric.utils import from_smiles

from chemistry.polymer_role_parser import identify_backbone_sidechain


class PolymerDataset(InMemoryDataset):
    """
    Polymer property prediction dataset with role annotations.

    Each graph contains:
    - x: Node features
    - edge_index: Edge indices
    - node_roles: Node roles (0=backbone, 1=side-chain)
    - y: Target property
    """

    def __init__(
        self,
        root: str,
        csv_path: str,
        target_column: str = "Tg_K",
        transform=None,
        pre_transform=None,
        smiles_column: str = "PSMILES"
    ):
        """
        Args:
            root: Root directory for dataset
            csv_path: Path to CSV file
            target_column: Name of target property column
            transform: Transform function
            pre_transform: Pre-transform function
            smiles_column: Name of SMILES column
        """
        self.csv_path = csv_path
        self.target_column = target_column
        self.smiles_column = smiles_column
        super().__init__(root, transform, pre_transform)
        self.load(self.processed_paths[0])
    
    @property
    def raw_file_names(self):
        return ["openpoly.csv"]
    
    @property
    def processed_file_names(self):
        return ["data.pt"]
    
    def process(self):
        """
        Process raw CSV into PyG Data objects with role information.

        Statistics tracked:
        - role_mismatch_count: Number of samples where role count != node count
        - total_processed: Total number of samples processed
        """
        df = pd.read_csv(self.raw_paths[0])
        data_list = []

        # Statistics for reviewer sanity check
        role_mismatch_count = 0
        total_processed = 0
        skipped_count = 0

        for idx, row in df.iterrows():
            smiles = row[self.smiles_column]

            # Skip if SMILES is empty
            if pd.isna(smiles) or smiles == "":
                skipped_count += 1
                continue

            # Convert SMILES to graph
            try:
                mol = from_smiles(smiles)
                if mol is None:
                    skipped_count += 1
                    continue
            except Exception as e:
                print(f"Error processing SMILES {smiles}: {e}")
                skipped_count += 1
                continue

            # Identify backbone/side-chain roles
            node_roles = identify_backbone_sidechain(smiles)
            if len(node_roles) != mol.num_nodes:
                # Log role mismatch for reviewer inspection
                print(f"[WARNING] Role mismatch for SMILES {smiles[:50]}...: "
                      f"expected {mol.num_nodes} nodes, got {len(node_roles)} roles")
                role_mismatch_count += 1
                # Fallback: assign all as backbone
                node_roles = [0] * mol.num_nodes
            
            mol.node_roles = torch.tensor(node_roles, dtype=torch.long)

            # Ensure node features are float32
            if mol.x is not None:
                mol.x = mol.x.float()
            if mol.edge_index is not None:
                mol.edge_index = mol.edge_index.long()

            # CRITICAL: Only add graphs with valid data
            # PyG will automatically drop graphs with no nodes during batching
            # This causes mismatch between batch.num_graphs and batch.y.shape[0]
            if mol.num_nodes == 0:
                continue

            if mol.x is None or mol.x.size(0) == 0:
                continue

            if mol.edge_index is None or mol.edge_index.size(1) == 0:
                continue

            # Extract target value
            target = row[self.target_column]
            if pd.isna(target):
                continue

            mol.y = torch.tensor([float(target)], dtype=torch.float32)  # shape: [1]

            data_list.append(mol)
            total_processed += 1

        # Print statistics for reviewer
        print("\n" + "=" * 60)
        print("Dataset Processing Statistics (Reviewer Sanity Check)")
        print("=" * 60)
        print(f"Total samples in CSV: {len(df)}")
        print(f"Successfully processed: {total_processed}")
        print(f"Skipped (invalid SMILES/empty): {skipped_count}")
        print(f"Role label mismatches: {role_mismatch_count} "
              f"({role_mismatch_count/total_processed*100:.2f}%)")
        print("=" * 60 + "\n")

        # Save to disk
        self.save(data_list, self.processed_paths[0])
    

