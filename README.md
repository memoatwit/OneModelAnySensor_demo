# One Model, Any Sensor — Audio Demos & Code

Anonymous supplementary material for NeurIPS 2026 submission.

## Audio Demos

Open `index.html` in a browser (or visit the GitHub Pages URL) to listen to audio demonstrations:

1. **Amplitude Drift** — SI-SDR-trained models produce outputs at incorrect amplitude (up to 8.4× reference) despite high SI-SDR scores. Non-scale-invariant losses (Wang L1, MRSTFT) eliminate this.

2. **Multi-Sensor Fusion** — Five-sensor fusion achieves 5.78% / 7.58% WER (clean/noisy), a 4.5× improvement over single-sensor baselines.

3. **Zero-Shot Sensor Generalization** — Leave-one-out evaluation: the model generalizes to held-out sensor types without retraining.

All samples are from the VibravOx test set (French speech, unseen speakers, 16 kHz).

## Code

- `code/model.py` — Self-contained sensor-agnostic architecture: shared Conv1d frontend, FiLM conditioning, cross-attention, sensor dropout, Mamba backbone, optional parallel STFT branch with gated spectral injection.
- `code/inference.py` — Minimal inference script: load checkpoint, enhance WAV files.

### Quick Start

```bash
pip install torch torchaudio mamba-ssm

python code/inference.py \
    --checkpoint model.pt \
    --input sensor1.wav sensor2.wav sensor3.wav \
    -o enhanced.wav
```

The model accepts any subset of sensors (1 to 5) as separate mono WAV files.

### Architecture Summary

```
Input: (B, S, T) — S sensors, each mono waveform

For each sensor s:
    h_s = GELU(Conv1d(x_s))           # shared frontend
    h_s = γ(e_s) ⊙ h_s + β(e_s)      # FiLM conditioning

Sensor dropout: randomly zero sensors (p=0.5, min_keep=1)
Cross-attention across sensor tokens (optional)
Mean pooling across sensors → (B, T, D)

8× MambaBlock (pre-norm residual)
    Optional: gated STFT spectral injection at layers [1, 3, 5]

LayerNorm → Conv1d(D, 1) + skip connection → (B, 1, T)
```

**Parameters:** 3.9M (base) / 5.1M (with STFT branch)

Full training code and pretrained checkpoints will be released upon acceptance.

## License

Code provided for review purposes only. Not for redistribution.
