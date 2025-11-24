from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import torch
import yaml

from ..data.preprocessing import prepare_dataset
from ..training.data_module import create_dataloaders
from ..training.trainer import Trainer
from ..utils import ensure_dir, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the anomaly detection framework.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--data_root", type=str, default="data/raw", help="Root directory of datasets")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to store results")
    return parser.parse_args()


def save_to_drive_if_mounted(local_path: Path, drive_path: str) -> None:
    """Copy file/folder to Drive if Drive is mounted."""
    drive_mount = Path("/content/drive")
    if drive_mount.exists() and (drive_mount / "MyDrive").exists():
        full_drive_path = drive_mount / "MyDrive" / drive_path
        ensure_dir(full_drive_path.parent if full_drive_path.suffix else full_drive_path)
        if local_path.is_dir():
            if full_drive_path.exists():
                shutil.rmtree(full_drive_path)
            shutil.copytree(local_path, full_drive_path)
        else:
            shutil.copy2(local_path, full_drive_path)
        print(f"✓ Saved to Drive: {drive_path}")


def main() -> None:
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    set_seed(config["seed"])
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")

    dataset_cfg = config["dataset"]
    bundle = prepare_dataset(
        root_dir=args.data_root,
        dataset_name=dataset_cfg["name"],
        machine_id=dataset_cfg["machine_id"],
        sequence_length=dataset_cfg["sequence_length"],
        stride=dataset_cfg["stride"],
        normalize=dataset_cfg["normalize"],
    )

    datamodule = create_dataloaders(
        bundle=bundle,
        batch_size=config["training"]["batch_size"],
        mask_cfg=config["masking"],
        device=device,
    )

    config["model"]["input_dim"] = datamodule.input_dim
    trainer = Trainer(config=config, input_dim=datamodule.input_dim, device=device)

    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)
    
    # Check if Drive is mounted
    drive_backup_dir = f"DMProject_backup/outputs"
    checkpoint_interval = config["training"].get("checkpoint_interval", 5)  # Save every 5 epochs by default

    history = []
    final_eval = None
    for epoch in range(1, config["training"]["epochs"] + 1):
        stats = trainer.train_epoch(datamodule.train_loader)
        eval_metrics = trainer.evaluate(datamodule.test_loader)
        final_eval = eval_metrics
        history.append(
            {
                "epoch": epoch,
                "train": stats.__dict__,
                "eval_mean_score": float(np.mean(eval_metrics["scores"])),
            }
        )
        print(f"Epoch {epoch} | Recon {stats.reconstruction_loss:.4f} | Mask {stats.mask_loss:.4f}")

        # Save history after each epoch
        with open(output_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        save_to_drive_if_mounted(output_dir / "history.json", f"{drive_backup_dir}/history.json")

        # Save checkpoint periodically
        if epoch % checkpoint_interval == 0 or epoch == config["training"]["epochs"]:
            checkpoint_path = output_dir / f"checkpoint_epoch_{epoch}.pt"
            trainer.save(checkpoint_path)
            save_to_drive_if_mounted(checkpoint_path, f"{drive_backup_dir}/checkpoint_epoch_{epoch}.pt")
            print(f"✓ Checkpoint saved at epoch {epoch}")

        # Save final outputs if this is the last epoch
        if epoch == config["training"]["epochs"] and final_eval:
            np.save(output_dir / "scores.npy", final_eval["scores"])
            np.save(output_dir / "labels.npy", final_eval["labels"])
            save_to_drive_if_mounted(output_dir / "scores.npy", f"{drive_backup_dir}/scores.npy")
            save_to_drive_if_mounted(output_dir / "labels.npy", f"{drive_backup_dir}/labels.npy")
            print("✓ Final outputs saved")


if __name__ == "__main__":
    main()

