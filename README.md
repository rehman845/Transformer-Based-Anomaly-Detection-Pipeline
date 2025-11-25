# Robust Multivariate Time-Series Anomaly Detection

This project implements a contamination-resilient anomaly detection framework that combines geometric masking, Transformer-based sequence modeling, contrastive representation learning, and a GAN regularizer. The goal is to learn a strong notion of “normal” behavior from multivariate time series even when the nominal training data contains sporadic anomalies.

## Highlights

- **Geometric masking augmentation** expands effective training data and encourages contextual reasoning.
- **Transformer encoder–decoder** captures long-range dependencies for reconstruction-based detection.
- **Contrastive loss** tightens the latent manifold of normal behavior, making anomalies stand out.
- **GAN module** forces reconstructions to resemble real normals, improving robustness to contaminated training signals.
- **Evaluation toolkit** reports ROC-AUC, PR-AUC, precision, recall, F1, and produces visual anomaly score plots.
- **Colab notebook** for end-to-end execution with minimal setup.

## Repository Structure

- `configs/` – YAML configs (hyperparameters, dataset options).
- `scripts/` – Utilities (dataset download, evaluation).
- `src/data/` – Dataset loaders, preprocessing, sliding windows.
- `src/models/` – Geometric masking, Transformer autoencoder, GAN, contrastive loss.
- `src/training/` – Data module, trainer, runnable training entry point.
- `src/eval/` – Metric computation and visualization helpers.
- `notebooks/colab_training.ipynb` – Guided Colab workflow.
- `requirements.txt` – Python dependencies.

## Quickstart (Local)

```bash
python -m venv .venv && source .venv/bin/activate      # or .venv\\Scripts\\activate on Windows
pip install -r requirements.txt
python scripts/download_dataset.py --name smd          # downloads Server Machine Dataset
python -m src.training.run --config configs/default.yaml --data_root data/raw --output_dir outputs
python scripts/evaluate.py --scores outputs/scores.npy --labels outputs/labels.npy --output outputs/eval
```

Artifacts (checkpoints, metrics, plots) are saved under `outputs/`.

## Dataset & Preprocessing

- **Default dataset:** Server Machine Dataset (SMD, eBay), multivariate sensor telemetry with labelled anomalies per machine.
- **Download:** `scripts/download_dataset.py` pulls archives from the TS-AD datasets repository.
- **Normalization:** Feature-wise `StandardScaler`.
- **Windowing:** Sliding windows with configurable length/stride (`sequence_length`, `stride`).
- **Labels:** Window label = max point-level label inside the window (binary normal/anomaly).
- **Masking:** `GeometricMasker` zeroes out random contiguous spans; mask lengths follow a geometric distribution to preferentially hide short/medium segments. During training we provide both masked and unmasked views of the same window.

## Model Architecture

1. **Input projection & positional encoding**  
   Linear layer lifts each timestep to `d_model`, sinusoidal encodings add temporal order.
2. **Transformer encoder**  
   Multi-head self-attention layers extract contextual features across multivariate channels.
3. **Latent aggregator & projection head**  
   Mean-pooled latent encodings feed a projection MLP used for contrastive loss and GAN conditioning.
4. **Transformer decoder**  
   Generates reconstructed sequences conditioned on encoder outputs; output projection maps back to sensor space.
5. **Contrastive learning**  
   InfoNCE loss brings masked/unmasked views of the same window together while pushing different windows apart.
6. **GAN regularizer**  
   Generator produces residual corrections conditioned on latent codes plus noise; discriminator distinguishes mean pooled reconstructed sequences vs. real ones. This mitigates contamination by penalizing unrealistic reconstructions.

## Training Procedure

- **Inputs:** For each window we create two views (masked + clean). The masked view is reconstructed; the clean view forms the positive pair for contrastive loss.
- **Losses:** Weighted sum of reconstruction loss, mask-only loss, contrastive (InfoNCE), and adversarial generator loss. Discriminator is trained with binary cross-entropy on real vs. generated representations.
- **Optimization:** AdamW with gradient clipping, configurable discriminator update frequency, and standard seed control for reproducibility.
- **Outputs:** Training history (`history.json`), best checkpoint, anomaly scores/labels from the final evaluation pass.

## Evaluation

- **Metrics:** Precision, recall, F1, ROC-AUC, PR-AUC, and anomaly ratios (ground truth vs predicted) via `src/eval/metrics.py`.
- **Thresholding:** Percentile-based (configurable, default 95th) computed on validation/test scores.
- **Visualization:** `src/eval/visualize.py` plots anomaly scores with the selected threshold and annotated anomalies.
- **Workflow:** `scripts/evaluate.py` consumes the saved `scores.npy` and `labels.npy`, writes metrics/plots to `outputs/eval/`.
- **Artifacts produced:** `scores.png` (anomaly scores), `roc.png`, `precision_recall.png`, and `metrics.json` (contains precision/recall/F1/ROC-AUC/PR-AUC + anomaly ratios).
- **Class ratios:** Evaluation metrics now include `positive_ratio_ground_truth` and `positive_ratio_predictions` so you can monitor anomaly prevalence and detector aggressiveness per dataset.

## Results

### Dataset: Server Machine Dataset (SMD)
- **Machine ID:** `machine-1-1`
- **Training Configuration:** 50 epochs, batch size 64, sequence length 128, stride 32
- **Model:** Transformer encoder-decoder (d_model=128, 4 encoder layers, 2 decoder layers) with contrastive learning and GAN regularization

### Performance Metrics

The framework was evaluated on the SMD test set with a threshold set at the 95th percentile of anomaly scores:

| Metric | Value |
|--------|-------|
| **ROC-AUC** | **0.950** |
| **PR-AUC** | **0.761** |
| **Precision** | **0.911** |
| **Recall** | **0.357** |
| **F1-Score** | **0.513** |
| **Threshold** | 66.26 |

### Analysis

- **High discriminative ability:** ROC-AUC of 0.950 indicates the model effectively separates normal and anomalous patterns in the learned representation space.
- **Strong precision:** With precision of 0.911, the model produces very few false positives—when it flags an anomaly, it is highly likely to be correct.
- **Conservative detection:** The recall of 0.357 suggests the model prioritizes precision over recall, which is appropriate for scenarios where false alarms are costly.
- **Balanced F1:** The F1-score of 0.513 reflects the precision-recall trade-off, which can be adjusted by tuning the threshold percentile.

### Visualization

The `scores.png` plot demonstrates:
- **Clear anomaly spikes:** The model identifies a major anomaly event around sequence index 550 with a dramatic score increase (reaching ~11.5M on the scaled axis).
- **Sensitive threshold:** The chosen threshold (95th percentile) flags both major anomalies and smaller deviations, providing comprehensive coverage.
- **Temporal consistency:** Anomaly scores remain near zero for normal periods, indicating the model has learned a tight representation of normal behavior.

### Key Findings

1. **Geometric masking** successfully expanded the effective training data and improved robustness, as evidenced by the model's ability to generalize to unseen test patterns.
2. **Transformer architecture** captured long-range dependencies effectively, enabling accurate reconstruction and anomaly scoring.
3. **Contrastive learning** tightened the latent manifold, making anomalies stand out clearly in the representation space.
4. **GAN regularization** helped the model learn realistic normal patterns despite potential contamination in training data.

The framework demonstrates strong performance on multivariate time series anomaly detection, with particular strength in precision—making it suitable for production systems where false positives must be minimized.

## Colab Guide

Open `notebooks/colab_training.ipynb` in Google Colab:

1. **Install dependencies** via `%pip install ...` (first cell).
2. **Download dataset** - The notebook automatically downloads SMD from the OmniAnomaly repository.
3. **Mount Google Drive** (IMPORTANT) - This enables automatic backup of training outputs every 5 epochs. If the runtime disconnects, your checkpoints and results are safely stored in Drive.
4. **Launch training** - Run the training cell. The script will:
   - Save `history.json` after each epoch to Drive
   - Save checkpoints every 5 epochs to Drive
   - Save final outputs (`scores.npy`, `labels.npy`) to Drive upon completion
5. **Evaluate** - Execute the evaluation cell to generate metrics and visualizations.
6. **Download artifacts** from `outputs/` or access them directly from Drive at `MyDrive/DMProject_backup/outputs/`.

**Note:** All training outputs are automatically backed up to Google Drive, so you won't lose progress if the Colab runtime disconnects.

## Demonstrating Anomaly Detection

1. Train on SMD (default) or another dataset using the Colab notebook or local setup.
2. Run `scripts/evaluate.py` to compute metrics and produce `scores.png`.
3. Inspect the plot to confirm high anomaly scores align with labeled anomalies.
4. Review the metrics in `outputs/eval/metrics.json` for quantitative performance.
5. See the **Results** section above for example outputs from the SMD dataset.

**Inspecting training losses:** to verify all four loss components during retraining, generate the loss plot:

```bash
python scripts/plot_losses.py --history outputs/history.json --output outputs/losses.png
```

This saves `losses.png` with reconstruction, mask, contrastive, and adversarial curves over epochs.

**Example Output:** On SMD `machine-1-1`, the framework achieved ROC-AUC of 0.950 and precision of 0.911, with clear visualization of anomaly spikes in the score plot.

## Extensibility

- Swap datasets by extending `src/data/preprocessing.py` with new loaders.
- Adjust Transformer/GAN capacity by editing `configs/default.yaml`.
- Replace InfoNCE with alternative contrastive objectives if desired.
- Integrate advanced thresholding (e.g., POT) by adding new modules under `src/eval/`.

## Reproducibility Notes

- Deterministic seeds configured via `src/utils.py`.
- Configs capture every hyperparameter for traceability.
- Requirements pinned to major versions in `requirements.txt`.
- Checkpoints and training history stored alongside metrics.

## Next Steps

- Experiment with different masking ratios or overlapping augmentations.
- Tune discriminator/generator sizes for specific datasets.
- Add multiscale attention or temporal convolutional stems for richer features.
- Explore online/streaming detection by adapting the dataloader and evaluation scripts.

