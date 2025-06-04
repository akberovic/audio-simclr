# Audio Spectrogram Contrastive Learning

Self-supervised contrastive embeddings of audio via SimCLR on log-mel spectrograms.

## Project Structure

- **data/**: download & preprocess scripts  
- **notebooks/**: EDA & visualization  
- **src/**:
  - `dataset.py` — contrastive-dataset  
  - `model.py`   — SimCLR model  
  - `loss.py`    — InfoNCE loss  
  - `train.py`   — training script  
  - `eval.py`    — downstream evaluation  
- **configs/**: hyperparameter files  
- **blog/**: draft write-ups
- **Dockerfile**: reproducible env

## Dataset

Place the ESC-50 dataset under `data/esc50` so that the directory looks like:

```
data/
  esc50/
    audio/          # WAV files
    meta/esc50.csv  # metadata
```

`src/dataset.py` expects a list of file paths pointing to the WAV files.
