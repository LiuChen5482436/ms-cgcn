"""Training script for MultiScaleCGCNN.

This script performs:
- Cached dataset loading
- Random train/val/test split (80/10/10) with a fixed seed
- Training with MSE + overlap regularization
- Validation metrics
- Final test evaluation
- Curve plotting

Paths are provided via CLI arguments to avoid hard-coded absolute paths.
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.loader import DataLoader

from model.MS_CGCN import MultiScaleCGCNN
from data.CIFGraphDatasetCached import CIFGraphDatasetCached


def parse_args():
    parser = argparse.ArgumentParser(description="Train MultiScaleCGCNN")

    # Dataset paths (defaults are under ./dataset_root)
    base = os.path.join(os.getcwd(), "dataset_root")

    parser.add_argument(
        "--root",
        type=str,
        default=base,
        help="Dataset root for caching (processed/ will be created inside). Default: ./dataset_root",
    )
    parser.add_argument(
        "--cif_dir",
        type=str,
        default=os.path.join(base, "cif"),
        help="Directory containing CIF files. Default: ./dataset_root/cif",
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default=os.path.join(base, "id_prop.csv"),
        help="CSV mapping id -> target (2 columns, no header). Default: ./dataset_root/id_prop.csv",
    )
    parser.add_argument(
        "--element_json",
        type=str,
        default=os.path.join(base, "init_weights", "regression", "atom_init.json"),
        help="JSON file for element features. Default: ./dataset_root/init_weights/regression/atom_init.json",
    )

    # Graph construction
    parser.add_argument("--bond_cutoff", type=float, default=10.0)
    parser.add_argument("--max_num_nbr", type=int, default=32)

    # Training
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--lambda_overlap", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)

    # Output
    parser.add_argument("--save_dir", type=str, default="results")

    # Device
    parser.add_argument("--device", type=str, default="auto", help="auto/cpu/cuda")

    return parser.parse_args()


def format_radius_info(model: torch.nn.Module) -> str:
    info = model.get_all_radius_ranges()
    parts = []
    for i, rr in enumerate(info):
        if rr is None:
            parts.append(f"S{i+1}:[None]")
        else:
            r0, r1 = rr
            parts.append(f"S{i+1}:[{r0:.2f},{r1:.2f}]")
    return " | ".join(parts)


def get_ylim(values_list, margin_ratio=0.05, start_from_zero=False):
    vmin = min(min(v) for v in values_list)
    vmax = max(max(v) for v in values_list)
    margin = (vmax - vmin) * margin_ratio

    low = vmin - margin
    high = vmax + margin

    if start_from_zero:
        low = 0.0

    if abs(high - low) < 1e-12:
        high = low + 1e-6

    return low, high


def main():
    args = parse_args()

    # Device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    # Output dir
    os.makedirs(args.save_dir, exist_ok=True)

    # Seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Dataset
    dataset = CIFGraphDatasetCached(
        root=args.root,
        cif_dir=args.cif_dir,
        csv_path=args.csv_path,
        element_json=args.element_json,
        bond_cutoff=args.bond_cutoff,
        max_num_nbr=args.max_num_nbr,
    )

    # Split: Train/Val/Test
    num_total = len(dataset)
    num_train = int(0.80 * num_total)
    num_val = int(0.10 * num_total)
    num_test = num_total - num_train - num_val

    train_set, val_set, test_set = torch.utils.data.random_split(
        dataset,
        [num_train, num_val, num_test],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    print(f"Dataset split: Train={num_train}, Val={num_val}, Test={num_test}")

    # Model
    model = MultiScaleCGCNN(
        dataset=dataset,
        num_scales=3,
        pre_fc_count=2,
        post_fc_count=2,
        radius_init_list=[
            (0.0, 3.5),
            (3.0, 5.5),
            (4.5, 8.0),
        ],
    ).to(device)

    optimizer = Adam(model.parameters(), lr=args.lr)

    # Curves
    train_loss_list = []
    train_mse_list, train_rmse_list, train_mae_list = [], [], []

    val_loss_list = []
    val_mse_list, val_rmse_list, val_mae_list = [], [], []

    # Train / Val loop
    for epoch in range(1, args.epochs + 1):
        # ---- Train ----
        model.train()
        train_mse_sum = 0.0
        train_mae_sum = 0.0
        train_loss_sum = 0.0

        for batch in train_loader:
            batch = batch.to(device)
            y = batch.y.view(-1)

            pred, _ = model(batch)

            mse_mean = F.mse_loss(pred, y, reduction="mean")
            reg = model.overlap_regularization()
            loss = mse_mean + args.lambda_overlap * reg

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            n = batch.num_graphs
            train_mse_sum += F.mse_loss(pred, y, reduction="sum").item()
            train_mae_sum += F.l1_loss(pred, y, reduction="sum").item()
            train_loss_sum += loss.item() * n

        train_mse = train_mse_sum / num_train
        train_rmse = float(np.sqrt(train_mse))
        train_mae = train_mae_sum / num_train
        train_loss = train_loss_sum / num_train

        train_loss_list.append(train_loss)
        train_mse_list.append(train_mse)
        train_rmse_list.append(train_rmse)
        train_mae_list.append(train_mae)

        # ---- Validation ----
        model.eval()
        val_mse_sum = 0.0
        val_mae_sum = 0.0
        val_loss_sum = 0.0

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                y = batch.y.view(-1)

                pred, _ = model(batch)

                mse_sum = F.mse_loss(pred, y, reduction="sum").item()
                mae_sum = F.l1_loss(pred, y, reduction="sum").item()

                reg = model.overlap_regularization()
                mse_mean = mse_sum / batch.num_graphs
                loss_mean = mse_mean + args.lambda_overlap * reg.item()

                n = batch.num_graphs
                val_mse_sum += mse_sum
                val_mae_sum += mae_sum
                val_loss_sum += loss_mean * n

        val_mse = val_mse_sum / num_val
        val_rmse = float(np.sqrt(val_mse))
        val_mae = val_mae_sum / num_val
        val_loss = val_loss_sum / num_val

        val_loss_list.append(val_loss)
        val_mse_list.append(val_mse)
        val_rmse_list.append(val_rmse)
        val_mae_list.append(val_mae)

        # Radius logging
        radius_str = format_radius_info(model)

        print(
            f"Epoch {epoch:04d} | "
            f"Train Loss {train_loss:.4f} | Train MAE {train_mae:.4f} MSE {train_mse:.4f} RMSE {train_rmse:.4f} | "
            f"Val   Loss {val_loss:.4f} | Val   MAE {val_mae:.4f} MSE {val_mse:.4f} RMSE {val_rmse:.4f} | "
            f"{radius_str}"
        )

    # Final test
    model.eval()
    test_mse_sum = 0.0
    test_mae_sum = 0.0

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            y = batch.y.view(-1)

            pred, _ = model(batch)
            test_mse_sum += F.mse_loss(pred, y, reduction="sum").item()
            test_mae_sum += F.l1_loss(pred, y, reduction="sum").item()

    test_mse = test_mse_sum / num_test
    test_rmse = float(np.sqrt(test_mse))
    test_mae = test_mae_sum / num_test

    print("\n===== Final Test Evaluation =====")
    print(f"Test MSE : {test_mse:.6f}")
    print(f"Test RMSE: {test_rmse:.6f}")
    print(f"Test MAE : {test_mae:.6f}")

    # Plot curves
    epochs = np.arange(1, args.epochs + 1)

    ylim_loss = get_ylim([train_loss_list, val_loss_list], margin_ratio=0.05, start_from_zero=False)
    plt.figure()
    plt.plot(epochs, train_loss_list, label="Train Loss")
    plt.plot(epochs, val_loss_list, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.ylim(*ylim_loss)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, "loss_curve.png"), dpi=300)
    plt.close()

    ylim_mse = get_ylim([train_mse_list, val_mse_list], margin_ratio=0.05, start_from_zero=True)
    plt.figure()
    plt.plot(epochs, train_mse_list, label="Train MSE")
    plt.plot(epochs, val_mse_list, label="Val MSE")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.ylim(*ylim_mse)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, "mse_curve.png"), dpi=300)
    plt.close()

    ylim_rmse = get_ylim([train_rmse_list, val_rmse_list], margin_ratio=0.05, start_from_zero=True)
    plt.figure()
    plt.plot(epochs, train_rmse_list, label="Train RMSE")
    plt.plot(epochs, val_rmse_list, label="Val RMSE")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.ylim(*ylim_rmse)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, "rmse_curve.png"), dpi=300)
    plt.close()

    ylim_mae = get_ylim([train_mae_list, val_mae_list], margin_ratio=0.05, start_from_zero=True)
    plt.figure()
    plt.plot(epochs, train_mae_list, label="Train MAE")
    plt.plot(epochs, val_mae_list, label="Val MAE")
    plt.xlabel("Epoch")
    plt.ylabel("MAE")
    plt.ylim(*ylim_mae)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, "mae_curve.png"), dpi=300)
    plt.close()

    print(f"[INFO] Curves saved to {args.save_dir}")


if __name__ == "__main__":
    main()
