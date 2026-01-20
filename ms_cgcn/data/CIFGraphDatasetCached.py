"""Cached PyG InMemoryDataset for CIF graphs.

This dataset wraps `CIFGraphDataset` and caches processed `Data` objects into a
single `.pt` file under `<root>/processed/`.

Behavior:
- If the cache exists, load it via `torch.load`.
- Otherwise, build from `CIFGraphDataset`, collate, and save.
"""

import os
import torch
from torch_geometric.data import InMemoryDataset

from .data_preproc import CIFGraphDataset


class CIFGraphDatasetCached(InMemoryDataset):
    """Cached InMemoryDataset.

    Args:
        root: Root directory for caching. Cache is stored under `<root>/processed/`.
        cif_dir: Directory containing `.cif` files.
        csv_path: CSV mapping `id -> target`.
        element_json: JSON file for per-element feature vectors.
        bond_cutoff: Neighbor search radius (Angstrom).
        max_num_nbr: Max neighbors per atom.
        transform: Optional PyG transform.
        pre_transform: Optional PyG pre-transform.

    Cached file path:
        <root>/processed/data_cutoff{bond_cutoff}_nbr{max_num_nbr}.pt
    """

    def __init__(
        self,
        root,
        cif_dir,
        csv_path,
        element_json,
        bond_cutoff=2.0,
        max_num_nbr=12,
        transform=None,
        pre_transform=None,
    ):
        self.original_dataset_params = dict(
            cif_dir=cif_dir,
            csv_path=csv_path,
            element_json=element_json,
            bond_cutoff=bond_cutoff,
            max_num_nbr=max_num_nbr,
            transform=transform,
            pre_transform=pre_transform,
        )

        processed_dir = os.path.join(root, "processed")
        os.makedirs(processed_dir, exist_ok=True)

        self.processed_file = os.path.join(
            processed_dir,
            f"data_cutoff{bond_cutoff}_nbr{max_num_nbr}.pt",
        )

        super().__init__(root, transform, pre_transform)

        if os.path.exists(self.processed_file):
            self.data, self.slices = torch.load(self.processed_file)
            print(f"[INFO] Loaded cached dataset: {self.processed_file}")
        else:
            self._generate_cache()

    @property
    def raw_file_names(self):
        files = os.listdir(self.original_dataset_params["cif_dir"])
        files = [f for f in files if f.endswith(".cif")]
        files.append(os.path.basename(self.original_dataset_params["csv_path"]))
        files.append(os.path.basename(self.original_dataset_params["element_json"]))
        return files

    @property
    def processed_file_names(self):
        return [os.path.basename(self.processed_file)]

    def _generate_cache(self):
        print("[INFO] Generating cached dataset...")

        original_dataset = CIFGraphDataset(
            cif_dir=self.original_dataset_params["cif_dir"],
            csv_path=self.original_dataset_params["csv_path"],
            element_json=self.original_dataset_params["element_json"],
            bond_cutoff=self.original_dataset_params["bond_cutoff"],
            max_num_nbr=self.original_dataset_params["max_num_nbr"],
            transform=self.original_dataset_params["transform"],
            pre_transform=self.original_dataset_params["pre_transform"],
        )

        data_list = [original_dataset[i] for i in range(len(original_dataset))]

        self.data, self.slices = self.collate(data_list)
        torch.save((self.data, self.slices), self.processed_file)
        print(f"[INFO] Cache saved: {self.processed_file}")
