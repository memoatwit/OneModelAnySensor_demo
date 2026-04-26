"""
Minimal inference script for the sensor-agnostic speech enhancement model.

Usage:
    python inference.py --checkpoint model.pt --input sensor1.wav sensor2.wav -o enhanced.wav

The model accepts any subset of sensors (1 to 5) as separate mono WAV files
and produces a single enhanced output.
"""

import argparse
import torch
import torchaudio
from model import SensorAgnosticEnhancer


def load_model(checkpoint_path, device="cpu"):
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt.get("model_config", {})
    model = SensorAgnosticEnhancer(
        d_model=cfg.get("d_model", 256),
        n_layers=cfg.get("n_layers", 8),
        num_sensors=cfg.get("num_sensors", 5),
        sensor_cross_attention=cfg.get("sensor_cross_attention", True),
        use_stft_branch=cfg.get("use_stft_branch", False),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model


@torch.no_grad()
def enhance(model, wav_paths, device="cpu", sample_rate=16000):
    sensors = []
    for p in wav_paths:
        wav, sr = torchaudio.load(p)
        if sr != sample_rate:
            wav = torchaudio.functional.resample(wav, sr, sample_rate)
        sensors.append(wav[0])  # mono

    # Stack sensors: (1, S, T)
    min_len = min(s.shape[0] for s in sensors)
    x = torch.stack([s[:min_len] for s in sensors]).unsqueeze(0).to(device)

    enhanced = model(x)  # (1, 1, T)
    return enhanced.squeeze(0).cpu()


def main():
    parser = argparse.ArgumentParser(description="Sensor-agnostic speech enhancement")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--input", nargs="+", required=True, help="Input WAV files (one per sensor)")
    parser.add_argument("-o", "--output", default="enhanced.wav", help="Output WAV path")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    model = load_model(args.checkpoint, args.device)
    enhanced = enhance(model, args.input, args.device)
    torchaudio.save(args.output, enhanced, 16000)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
