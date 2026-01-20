"""CIF -> PyG Graph Dataset.

This dataset converts CIF crystal structures into PyTorch Geometric `Data` objects.

Features:
- Node features: element embedding vectors loaded from a JSON file.
- Edge features: (1) Gaussian distance expansion (50 dims)
                 (2) Normalized direction vector (3 dims), scaled by 1/sqrt(3)
- Edge distances: stored separately as `edge_dist` for compatibility with a learnable
  radius-range gate (e.g., RadiusRangeGate).

Data fields:
    data.x         : [N, node_feat_dim]
    data.edge_index: [2, E]
    data.edge_attr : [E, 53]  (50 Gaussian + 3 direction)
    data.edge_dist : [E, 1]   (raw distance)
    data.y         : [1]      (scalar regression target)

Notes:
- For each atom i, neighbors are collected within `bond_cutoff`, then sorted by
  distance and truncated to `max_num_nbr`.
- Edges are added in both directions (i->j and j->i) with opposite directions.
"""

import os
import json
import numpy as np
import pandas as pd
import torch
from pymatgen.core import Structure
from torch_geometric.data import Dataset, Data


def gaussian_distance_expansion(distances, centers=None, sigma=0.1):
    """Gaussian basis expansion of interatomic distances.

    Args:
        distances: Array-like or Tensor of distances. Expected shape [E] or [E, 1].
        centers:   Optional list/array of Gaussian centers. If None, uses 50 centers
                   linearly spaced in [0, 5].
        sigma:     Width of Gaussian kernels.

    Returns:
        Tensor of shape [E, num_centers], where num_centers=50 by default.
    """
    if centers is None:
        centers = np.linspace(0, 5, 50)

    distances = torch.tensor(distances, dtype=torch.float32).view(-1, 1)
    centers = torch.tensor(centers, dtype=torch.float32).view(1, -1)

    gauss_exp = torch.exp(-((distances - centers) ** 2) / sigma ** 2)
    return gauss_exp


class CIFGraphDataset(Dataset):
    """CIF -> Graph dataset.

    This class:
    - Reads CIF files from `cif_dir`
    - Loads targets from a CSV file (`csv_path`)
    - Loads per-element feature vectors from `element_json`
    - Builds a PyG `Data` object per CIF

    Key hyperparameters:
        bond_cutoff:  Neighbor search radius (Angstrom).
        max_num_nbr:  Maximum neighbors per atom (after sorting by distance).

    Edge construction details (kept exactly as the original logic):
        For each atom i:
            neighbors = get_neighbors(atom_i, bond_cutoff)
            neighbors = sorted(neighbors, by distance)[:max_num_nbr]
            For each neighbor j:
                add edge i->j and j->i
                edge_dist stores the scalar distance
                edge_dir stores a normalized direction vector scaled by 1/sqrt(3)
    """

    def __init__(
        self,
        cif_dir,
        csv_path,
        element_json,
        bond_cutoff=2.0,
        max_num_nbr=12,
        transform=None,
        pre_transform=None,
    ):
        self.element_features = self._load_element_features(element_json)
        self.cif_dir = cif_dir
        self.bond_cutoff = bond_cutoff
        self.max_num_nbr = max_num_nbr

        # Load id -> target mapping from CSV
        csv_data = pd.read_csv(csv_path, header=None)
        csv_data.columns = ["id", "target"]
        self.id_to_target = {str(k): v for k, v in zip(csv_data["id"], csv_data["target"]) }

        # Keep only CIF files that have a corresponding target
        all_cif_files = [f for f in os.listdir(cif_dir) if f.endswith(".cif")]
        self.cif_files = []
        for f in all_cif_files:
            cif_id = f.replace(".cif", "")
            if cif_id in self.id_to_target:
                self.cif_files.append(f)
            else:
                print(f"[WARN] Target not found, skip file: {f}")

        super().__init__(cif_dir, transform, pre_transform)

    def _load_element_features(self, json_path):
        """Load per-element feature vectors from a JSON file."""
        with open(json_path, "r", encoding="utf-8") as f:
            element_dict = json.load(f)
        return {str(k): np.array(v, dtype=float) for k, v in element_dict.items()}

    def __len__(self):
        return len(self.cif_files)

    def __getitem__(self, idx):
        cif_file = self.cif_files[idx]
        cif_path = os.path.join(self.cif_dir, cif_file)

        structure = Structure.from_file(cif_path)

        # Label
        cif_id = cif_file.replace(".cif", "")
        y = torch.tensor([float(self.id_to_target[cif_id])], dtype=torch.float32)

        # Node features
        x = np.array([self.element_features[str(atom.number)] for atom in structure.species])
        x = torch.tensor(x, dtype=torch.float32)

        # Edges
        edge_i, edge_j = [], []
        edge_d = []
        edge_dir = []

        num_atoms = len(structure)
        dir_scale = 1.0 / np.sqrt(3.0)

        for i in range(num_atoms):
            neighbors = structure.get_neighbors(structure[i], self.bond_cutoff)
            neighbors = sorted(neighbors, key=lambda n: n[1])[: self.max_num_nbr]

            ri = structure[i].coords
            for nbr in neighbors:
                j = nbr[2]
                dist = nbr[1]
                rj = nbr[0].coords

                vec = rj - ri
                norm = np.linalg.norm(vec)
                if norm < 1e-8:
                    continue

                dir_ij = (vec / norm) * dir_scale

                # i -> j
                edge_i.append(i)
                edge_j.append(j)
                edge_d.append(dist)
                edge_dir.append(dir_ij)

                # j -> i
                edge_i.append(j)
                edge_j.append(i)
                edge_d.append(dist)
                edge_dir.append(-dir_ij)

        if len(edge_i) == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_d_t = torch.empty((0, 1), dtype=torch.float32)
            edge_attr = torch.empty((0, 53), dtype=torch.float32)
        else:
            edge_index = torch.tensor([edge_i, edge_j], dtype=torch.long)
            edge_d_t = torch.tensor(edge_d, dtype=torch.float32).view(-1, 1)
            edge_attr = torch.cat(
                [
                    gaussian_distance_expansion(edge_d_t),
                    torch.from_numpy(np.asarray(edge_dir)).float(),
                ],
                dim=1,
            )

        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            edge_dist=edge_d_t,
            y=y,
        )

        if self.transform:
            data = self.transform(data)

        return data
