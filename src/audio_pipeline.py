"""Whisper speech understanding and audio preprocessing pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .utils import FIGURES_DIR, RESULTS_DIR, word_error_rate


class WhisperAudioPipeline:
    """Speech-to-text pipeline based on OpenAI Whisper."""

    def __init__(self, model_size: str = "base"):
        self.model_size = model_size
        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        try:
            import whisper

            self.model = whisper.load_model(self.model_size)
        except Exception:
            self.model = None

    def load_audio(self, audio_path: str | Path, target_sr: int = 16000) -> tuple[np.ndarray, int]:
        import librosa

        signal, sr = librosa.load(str(audio_path), sr=target_sr, mono=True)
        return signal.astype(np.float32), sr

    def extract_mfcc(
        self,
        audio_path: str | Path,
        n_mfcc: int = 40,
        max_frames: int = 300,
    ) -> np.ndarray:
        import librosa

        signal, sr = self.load_audio(audio_path)
        mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc)
        mfcc = (mfcc - mfcc.mean(axis=1, keepdims=True)) / (mfcc.std(axis=1, keepdims=True) + 1e-8)
        if mfcc.shape[1] < max_frames:
            mfcc = np.pad(mfcc, ((0, 0), (0, max_frames - mfcc.shape[1])), mode="constant")
        else:
            mfcc = mfcc[:, :max_frames]
        return mfcc.astype(np.float32)

    def transcribe(self, audio_path: str | Path) -> dict[str, Any]:
        if self.model is None:
            return {
                "transcript": "",
                "language": "unknown",
                "segments": [],
                "confidence": 0.0,
                "warning": "Whisper model unavailable. Install dependencies or allow model download.",
            }
        result = self.model.transcribe(str(audio_path), fp16=False)
        transcript = result.get("text", "").strip()
        segments = result.get("segments", [])
        avg_logprob = np.mean([seg.get("avg_logprob", -2.0) for seg in segments]) if segments else -2.0
        confidence = float(1 / (1 + np.exp(-avg_logprob)))
        return {
            "transcript": transcript,
            "language": result.get("language", "unknown"),
            "segments": segments,
            "confidence": confidence,
        }

    def evaluate(self, manifest: pd.DataFrame, output_dir: Path = FIGURES_DIR) -> dict[str, float]:
        """Evaluate WER for a manifest containing audio_path and reference_text."""

        output_dir.mkdir(parents=True, exist_ok=True)
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        rows = []
        for _, row in manifest.iterrows():
            pred = self.transcribe(row["audio_path"])
            wer = word_error_rate(row["reference_text"], pred["transcript"])
            rows.append(
                {
                    "audio_path": row["audio_path"],
                    "reference_text": row["reference_text"],
                    "transcript": pred["transcript"],
                    "wer": wer,
                }
            )
        results = pd.DataFrame(rows)
        results.to_csv(RESULTS_DIR / "speech_evaluation.csv", index=False)
        mean_wer = float(results["wer"].mean()) if not results.empty else 0.0

        if not results.empty:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.bar(range(len(results)), results["wer"], color="#9f3f5f")
            ax.axhline(mean_wer, color="#1f2937", linestyle="--", label=f"Mean WER={mean_wer:.2f}")
            ax.set_title("Speech Recognition Word Error Rate")
            ax.set_xlabel("Audio Query")
            ax.set_ylabel("WER")
            ax.legend()
            fig.tight_layout()
            fig.savefig(output_dir / "speech_wer.png", dpi=180)
            plt.close(fig)

        return {"mean_wer": mean_wer, "n_samples": float(len(results))}


def main() -> None:
    print("WhisperAudioPipeline ready. Provide a manifest DataFrame to evaluate audio files.")


if __name__ == "__main__":
    main()
