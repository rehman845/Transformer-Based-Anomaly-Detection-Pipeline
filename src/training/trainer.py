from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..models.gan import SequenceDiscriminator, SequenceGenerator
from ..models.losses import info_nce_loss
from ..models.transformer import TransformerAutoencoder
from ..utils import ensure_dir, save_checkpoint


@dataclass
class TrainingStats:
    reconstruction_loss: float
    mask_loss: float
    contrastive_loss: float
    adversarial_loss: float


class Trainer:
    def __init__(
        self,
        config: Dict,
        input_dim: int,
        device: torch.device,
    ) -> None:
        self.config = config
        self.device = device

        model_cfg = config["model"]
        self.model = TransformerAutoencoder(
            input_dim=input_dim,
            d_model=model_cfg["d_model"],
            nhead=model_cfg["nhead"],
            num_encoder_layers=model_cfg["num_encoder_layers"],
            num_decoder_layers=model_cfg["num_decoder_layers"],
            dim_feedforward=model_cfg["dim_feedforward"],
            dropout=model_cfg["dropout"],
            latent_dim=model_cfg["latent_dim"],
            projection_dim=config["contrastive"]["projection_dim"],
        ).to(device)

        gan_cfg = config["gan"]
        latent_dim = model_cfg["latent_dim"]
        self.generator = SequenceGenerator(
            latent_dim=latent_dim,
            noise_dim=gan_cfg["noise_dim"],
            hidden_dim=gan_cfg["hidden_dim"],
            output_dim=input_dim,
        ).to(device)
        self.discriminator = SequenceDiscriminator(
            input_dim=input_dim,
            hidden_dim=gan_cfg["hidden_dim"],
        ).to(device)

        training_cfg = config["training"]
        self.optim_model = torch.optim.AdamW(
            list(self.model.parameters()) + list(self.generator.parameters()),
            lr=training_cfg["learning_rate"],
            weight_decay=training_cfg["weight_decay"],
        )
        self.optim_discriminator = torch.optim.AdamW(
            self.discriminator.parameters(),
            lr=training_cfg["learning_rate"],
            weight_decay=training_cfg["weight_decay"],
        )
        self.criterion = nn.MSELoss()

    def _compute_losses(
        self,
        batch: dict,
        other_view: dict,
        enhanced_reconstruction,
        projection,
        disc_fake,
    ) -> TrainingStats:
        cfg = self.config["training"]
        mask = batch["mask"].unsqueeze(-1)
        mask_loss = self.criterion(enhanced_reconstruction * mask, batch["data"] * mask)
        reconstruction_loss = self.criterion(enhanced_reconstruction, batch["data"])

        projections_other = other_view["projection"]
        contrastive_loss = info_nce_loss(
            projection,
            projections_other,
            temperature=self.config["contrastive"]["temperature"],
        )

        adversarial_loss = nn.functional.binary_cross_entropy_with_logits(
            disc_fake,
            torch.ones_like(disc_fake),
        )

        total_loss = (
            cfg["reconstruction_weight"] * reconstruction_loss
            + cfg["mask_weight"] * mask_loss
            + cfg["contrastive_weight"] * contrastive_loss
            + cfg["adversarial_weight"] * adversarial_loss
        )

        total_loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), cfg["grad_clip"])

        return TrainingStats(
            reconstruction_loss=reconstruction_loss.item(),
            mask_loss=mask_loss.item(),
            contrastive_loss=contrastive_loss.item(),
            adversarial_loss=adversarial_loss.item(),
        )

    def train_epoch(self, dataloader: DataLoader) -> TrainingStats:
        self.model.train()
        self.generator.train()
        self.discriminator.train()

        running = TrainingStats(0.0, 0.0, 0.0, 0.0)
        for batch in tqdm(dataloader, desc="Training"):
            self.optim_model.zero_grad()
            self.optim_discriminator.zero_grad()

            reconstruction, latent, projection = self.model(batch["masked"])
            _, _, other_view_projection = self.model(batch["data"])
            noise = torch.randn(batch["data"].size(0), self.config["gan"]["noise_dim"], device=self.device)
            residual = self.generator(latent, noise).unsqueeze(1).repeat(1, batch["data"].size(1), 1)
            enhanced = reconstruction + residual
            disc_fake = self.discriminator(enhanced.mean(dim=1))
            stats = self._compute_losses(
                batch,
                {"projection": other_view_projection.detach()},
                enhanced,
                projection,
                disc_fake,
            )
            self.optim_model.step()

            for _ in range(self.config["training"]["discriminator_steps"]):
                self.optim_discriminator.zero_grad()
                disc_real = self.discriminator(batch["data"].mean(dim=1).detach())
                disc_fake = self.discriminator(enhanced.mean(dim=1).detach())
                loss_real = nn.functional.binary_cross_entropy_with_logits(
                    disc_real,
                    torch.ones_like(disc_real),
                )
                loss_fake = nn.functional.binary_cross_entropy_with_logits(
                    disc_fake,
                    torch.zeros_like(disc_fake),
                )
                disc_loss = 0.5 * (loss_real + loss_fake)
                disc_loss.backward()
                self.optim_discriminator.step()

            running.reconstruction_loss += stats.reconstruction_loss
            running.mask_loss += stats.mask_loss
            running.contrastive_loss += stats.contrastive_loss
            running.adversarial_loss += stats.adversarial_loss

        num_batches = len(dataloader)
        return TrainingStats(
            reconstruction_loss=running.reconstruction_loss / num_batches,
            mask_loss=running.mask_loss / num_batches,
            contrastive_loss=running.contrastive_loss / num_batches,
            adversarial_loss=running.adversarial_loss / num_batches,
        )

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> dict:
        self.model.eval()
        self.generator.eval()
        self.discriminator.eval()

        scores = []
        labels = []
        for batch in tqdm(dataloader, desc="Evaluating"):
            reconstruction, _, _ = self.model(batch["data"])
            residual = (reconstruction - batch["data"]) ** 2
            score = residual.mean(dim=(1, 2))
            scores.append(score.cpu())
            labels.append(batch["label"].cpu())
        scores = torch.cat(scores).numpy()
        labels = torch.cat(labels).numpy()
        return {"scores": scores, "labels": labels}

    def save(self, path: str) -> None:
        ensure_dir(Path(path).parent)
        save_checkpoint(
            {
                "model": self.model.state_dict(),
                "generator": self.generator.state_dict(),
                "discriminator": self.discriminator.state_dict(),
                "config": self.config,
            },
            path,
        )

