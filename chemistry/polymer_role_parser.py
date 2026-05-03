"""
Polymer role parser: identify backbone vs side-chain atoms in polymer SMILES.
Backbone: shortest path between two wildcard [*] atoms
Side-chain: all other atoms
"""

from typing import List
from collections import deque

try:
    from rdkit import Chem
except ImportError:
    Chem = None


def bfs_shortest_path(mol: "Chem.Mol", start_idx: int, end_idx: int) -> List[int]:
    """
    Find shortest path between two atoms using BFS.
    
    Args:
        mol: RDKit molecule object
        start_idx: starting atom index
        end_idx: ending atom index
        
    Returns:
        List of atom indices forming the shortest path
    """
    queue = deque([(start_idx, [start_idx])])
    visited = {start_idx}
    
    while queue:
        (current, path) = queue.popleft()
        
        if current == end_idx:
            return path
        
        atom = mol.GetAtomWithIdx(current)
        for neighbor in atom.GetNeighbors():
            neighbor_idx = neighbor.GetIdx()
            if neighbor_idx not in visited:
                visited.add(neighbor_idx)
                queue.append((neighbor_idx, path + [neighbor_idx]))
    
    return [start_idx, end_idx]


def identify_backbone_sidechain(psmiles: str) -> List[int]:
    """
    Identify backbone vs side-chain roles for each atom in polymer SMILES.

    Backbone: shortest path between two wildcard [*] atoms
    Side-chain: all other atoms

    Args:
        psmiles: Polymer SMILES string with wildcard [*] atoms

    Returns:
        List of integers: 0 for backbone, 1 for side-chain

    Note: For polymers without explicit repeat unit markers (fewer than 2 wildcard
    atoms), all atoms are treated as backbone. This is documented in the paper as:
    "For polymers without explicit repeat unit markers, all atoms are treated as backbone."
    """
    if Chem is None:
        raise ImportError("RDKit is required. Install with: pip install rdkit")
    
    mol = Chem.MolFromSmiles(psmiles, sanitize=False)
    if mol is None:
        return []
    
    num_atoms = mol.GetNumAtoms()
    
    # Find wildcard atoms (atomic number = 0)
    wildcard_idx = [
        atom.GetIdx() for atom in mol.GetAtoms()
        if atom.GetAtomicNum() == 0
    ]
    
    # Fallback: if fewer than 2 wildcards, treat all as backbone
    if len(wildcard_idx) < 2:
        return [0] * num_atoms
    
    # Find shortest path between first two wildcards
    path = bfs_shortest_path(mol, wildcard_idx[0], wildcard_idx[1])
    backbone = set(path)
    
    # Assign roles
    roles = []
    for i in range(num_atoms):
        if i in backbone:
            roles.append(0)  # backbone
        else:
            roles.append(1)  # side-chain
    
    return roles


def batch_identify_backbone_sidechain(psmiles_list: List[str]) -> List[List[int]]:
    """
    Batch process multiple polymer SMILES strings.
    
    Args:
        psmiles_list: List of polymer SMILES strings
        
    Returns:
        List of role lists
    """
    return [identify_backbone_sidechain(smiles) for smiles in psmiles_list]
