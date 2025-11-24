import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def plot_losses(history_path: str, output_path: str) -> None:
    with open(history_path, "r", encoding="utf-8") as f:
        history = json.load(f)

    if not history:
        raise ValueError("History file is empty. Run training first.")

    epochs = [entry["epoch"] for entry in history]
    recon = [entry["train"]["reconstruction_loss"] for entry in history]
    mask = [entry["train"]["mask_loss"] for entry in history]
    contrastive = [entry["train"]["contrastive_loss"] for entry in history]
    adversarial = [entry["train"]["adversarial_loss"] for entry in history]

    plt.figure(figsize=(10, 4))
    plt.plot(epochs, recon, label="Reconstruction")
    plt.plot(epochs, mask, label="Mask")
    plt.plot(epochs, contrastive, label="Contrastive")
    plt.plot(epochs, adversarial, label="Adversarial")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Components")
    plt.legend()
    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"Saved loss plot to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot training loss components.")
    parser.add_argument(
        "--history",
        type=str,
        default="outputs/history.json",
        help="Path to training history JSON file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/losses.png",
        help="Path to save the loss plot image.",
    )
    args = parser.parse_args()

    plot_losses(args.history, args.output)


if __name__ == "__main__":
    main()

