import random
from typing import Callable, List, Optional, Tuple

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset


class ContrastiveAudioDataset(Dataset):
    """Dataset returning two augmented mel-spectrogram views of the same clip.

    Each view is created by applying random wave and spectrogram augmentations
    such as cropping, time stretching, additive noise, time masking and
    frequency masking.
    """
    def __init__(
        self,
        file_paths: List[str],
        sr: int = 22050,
        duration: float = 5.0,
        n_mels: int = 64,
        augment: Optional[Callable[[np.ndarray, int], np.ndarray]] = None,
    ) -> None:
        """Initialize the dataset.

        Parameters
        ----------
        file_paths : List[str]
            List of paths to audio files.
        sr : int, optional
            Sample rate for loading audio, by default 22050.
        duration : float, optional
            Duration in seconds for each clip, by default 5.0.
        n_mels : int, optional
            Number of mel bins for the spectrogram, by default 64.
        augment : Callable, optional
            Function that applies waveform augmentations. If ``None`` a small
            set of default augmentations is used.
        """
        self.file_paths = file_paths
        self.sr = sr
        self.duration = duration
        self.n_samples = int(sr * duration)
        self.n_mels = n_mels
        self.augment = augment or self._default_augment

    @staticmethod
    def _default_augment(wave: np.ndarray, sr: int) -> np.ndarray:
        """Return a randomly augmented waveform.

        Augmentations include random cropping/padding, time stretching and
        additive Gaussian noise. The clip is always returned with the same
        length as the original input.
        """
        n_samples = len(wave)

        # Random cropping to 4 seconds and pad back to ``n_samples``
        crop_len = int(n_samples * 0.8)
        start = 0 if n_samples == crop_len else random.randint(0, n_samples - crop_len)
        wave = wave[start : start + crop_len]
        if len(wave) < n_samples:
            wave = np.pad(wave, (0, n_samples - len(wave)))

        # Time stretch by +-10%
        rate = random.uniform(0.9, 1.1)
        wave = librosa.effects.time_stretch(wave, rate)
        if len(wave) < n_samples:
            wave = np.pad(wave, (0, n_samples - len(wave)))
        else:
            wave = wave[:n_samples]

        # Additive Gaussian noise
        var = random.uniform(0.01, 0.05)
        wave = wave + np.random.randn(len(wave)) * var

        return wave.astype(np.float32)

    def __len__(self) -> int:
        return len(self.file_paths)

    def _load_audio(self, path: str) -> np.ndarray:
        """Load an audio file and ensure it has the correct length."""
        wave, _ = librosa.load(path, sr=self.sr, duration=self.duration)
        if len(wave) < self.n_samples:
            # Pad if shorter than desired length
            padding = self.n_samples - len(wave)
            wave = np.pad(wave, (0, padding))
        else:
            wave = wave[: self.n_samples]
        return wave

    def _to_log_mel(self, wave: np.ndarray) -> np.ndarray:
        """Convert waveform to log-mel spectrogram."""
        mel = librosa.feature.melspectrogram(
            y=wave, sr=self.sr, n_mels=self.n_mels, center=False
        )
        log_mel = librosa.power_to_db(mel).astype(np.float32)
        return log_mel

    @staticmethod
    def _spec_augment(spec: np.ndarray) -> np.ndarray:
        """Apply time and frequency masking to a spectrogram."""
        time_max = int(spec.shape[1] * 0.3)
        if time_max > 0:
            t = random.randint(0, time_max)
            t0 = random.randint(0, spec.shape[1] - t) if t > 0 else 0
            spec[:, t0 : t0 + t] = spec.min()

        freq_max = int(spec.shape[0] * 0.2)
        if freq_max > 0:
            f = random.randint(0, freq_max)
            f0 = random.randint(0, spec.shape[0] - f) if f > 0 else 0
            spec[f0 : f0 + f, :] = spec.min()

        return spec

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return two augmented spectrograms for the clip at ``index``."""
        path = self.file_paths[index]
        wave = self._load_audio(path)

        # Two separate augmentations
        wave1 = self.augment(wave.copy(), self.sr)
        wave2 = self.augment(wave.copy(), self.sr)

        spec1 = self._to_log_mel(wave1)
        spec2 = self._to_log_mel(wave2)

        spec1 = self._spec_augment(spec1)
        spec2 = self._spec_augment(spec2)

        # [1, n_mels, T]
        spec1 = torch.from_numpy(spec1).unsqueeze(0)
        spec2 = torch.from_numpy(spec2).unsqueeze(0)
        return spec1, spec2


__all__ = ["ContrastiveAudioDataset"]
