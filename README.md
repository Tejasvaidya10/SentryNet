# SentryNet

Fraud detection portfolio project: XGBoost classification on IEEE-CIS
Fraud Detection data, with engineered transaction features, graph-based
shared-attribute features, and a concept-drift monitoring layer.

## Setup

    uv sync

## Tests

    uv run pytest

## Data

Download the IEEE-CIS Fraud Detection dataset from Kaggle and place
`train_transaction.csv` and `train_identity.csv` under `data/` (gitignored).
