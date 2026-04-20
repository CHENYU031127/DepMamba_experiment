# DepMamba Experiment

This directory is the experiment copy of the original `DepMamba` project. Use this copy for model changes, ablations, and paper-driven experiments while keeping the original project untouched.

## Current scope

- Baseline model: `DepMamba`
- Datasets supported by the current code:
  - `dvlog`
  - `lmvd`
- Planned next step:
  - add lightweight pre-fusion cross-modal alignment losses

## Project layout

- `main.py`: training and evaluation entry point
- `config/config.yaml`: default runtime configuration
- `models/`: model definitions
- `datasets/`: dataset loaders
- `scripts/`: example training commands

## Environment

Create a Python environment first, then install the dependencies from `requirements.txt`.

Example:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For Conda users, creating a dedicated environment is recommended before installing the same package set.

## Data

The current code expects dataset paths outside the repository or in a local, untracked location. Do not commit raw features or checkpoints into Git.

Default config file:

- [config.yaml](./config/config.yaml)

Important path field:

- `data_dir`

## Training

Example scripts:

- `scripts/train_dvlog.sh`
- `scripts/train_lmvd.sh`

Direct example:

```bash
python main.py --dataset dvlog --train True
```

## Recommended Git workflow

1. Develop in this directory.
2. Commit changes locally with clear messages.
3. Push to a private remote repository.
4. On the server, clone once and later update with `git pull`.
5. Keep datasets, checkpoints, and results outside version control.

## Notes before model changes

- Keep the baseline branch recoverable.
- Prefer one focused change per commit.
- Record the config used for every training run.
