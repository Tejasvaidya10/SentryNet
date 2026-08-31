"""Generate the SentryNet narrative notebook skeletons.

Notebooks import from the sentrynet package rather than embedding logic
directly, and guard on the presence of the real Kaggle CSVs (gitignored
under data/) so they execute cleanly end-to-end even before the dataset is
downloaded. Re-run this script to reset notebooks back to their skeleton
state.
"""

from pathlib import Path

import nbformat as nbf

GUARD_CODE = (
    "from sentrynet.config import DATA_DIR\n\n"
    'TRANSACTION_PATH = DATA_DIR / "train_transaction.csv"\n'
    'IDENTITY_PATH = DATA_DIR / "train_identity.csv"\n'
    "DATA_AVAILABLE = TRANSACTION_PATH.exists()\n\n"
    "if not DATA_AVAILABLE:\n"
    '    print(f"Dataset not found at {TRANSACTION_PATH}. Download the IEEE-CIS "\n'
    '          "Fraud Detection dataset from Kaggle and place it under data/ to run this notebook.")'
)


def build_notebook(title: str, sections: list[tuple[str, str]]) -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    cells = [nbf.v4.new_markdown_cell(f"# {title}"), nbf.v4.new_code_cell(GUARD_CODE)]
    for heading, code in sections:
        cells.append(nbf.v4.new_markdown_cell(f"## {heading}"))
        cells.append(nbf.v4.new_code_cell(code))
    nb["cells"] = cells
    return nb


def main() -> None:
    notebooks_dir = Path(__file__).resolve().parents[1] / "notebooks"
    notebooks_dir.mkdir(exist_ok=True)

    eda = build_notebook(
        "SentryNet -- Exploratory Data Analysis",
        [
            (
                "Load data",
                "if DATA_AVAILABLE:\n"
                "    from sentrynet.data.loader import load_transactions\n\n"
                "    df = load_transactions(TRANSACTION_PATH, IDENTITY_PATH)\n"
                "    df.shape",
            ),
            (
                "Class balance and TransactionDT range",
                "if DATA_AVAILABLE:\n"
                '    print(df["isFraud"].value_counts(normalize=True))\n'
                '    print(df["TransactionDT"].min(), df["TransactionDT"].max())',
            ),
        ],
    )

    feature_engineering = build_notebook(
        "SentryNet -- Feature Engineering (including graph features)",
        [
            (
                "Transaction-level features (label-independent, safe to compute before the split)",
                "if DATA_AVAILABLE:\n"
                "    from sentrynet.features.velocity import transaction_velocity\n"
                "    from sentrynet.features.recency import time_since_last\n"
                "    from sentrynet.features.geo import addr_change_flag\n"
                "    from sentrynet.graph.fingerprint import build_card_entity_ids\n\n"
                '    df["card_entity_id"] = build_card_entity_ids(df)\n'
                '    df["velocity_1h"] = transaction_velocity(df, "card_entity_id", "TransactionDT", 3600)\n'
                '    df["time_since_last"] = time_since_last(df, "card_entity_id", "TransactionDT")\n'
                '    df["addr_changed"] = addr_change_flag(df, "card_entity_id", "TransactionDT")\n\n'
                "    # Merchant risk encoding is intentionally NOT computed here: it is fit on\n"
                "    # isFraud, so it must be fit after the temporal split (train only) to avoid\n"
                "    # leakage -- see the Modeling and Evaluation notebook.",
            ),
            (
                "Graph features: device fingerprint + reconstructed card identity",
                "# NOTE: the reconstructed card identity (card_entity_id above) is a\n"
                "# probabilistic heuristic, not a verified cardholder ID -- see\n"
                "# sentrynet.graph.fingerprint.build_card_entity_ids docstring.\n"
                "if DATA_AVAILABLE:\n"
                "    from sentrynet.graph.fingerprint import build_device_fingerprints\n"
                "    from sentrynet.graph.build import build_bipartite_graph\n"
                "    from sentrynet.graph.features import extract_entity_features\n"
                "    from sentrynet.graph.join import transaction_entity_features\n\n"
                '    df["device_fingerprint"] = build_device_fingerprints(df)\n'
                "    g = build_bipartite_graph(\n"
                '        df, transaction_id_col="TransactionID",\n'
                '        entity_cols=("device_fingerprint", "card_entity_id"),\n'
                "    )\n"
                "    entity_features = extract_entity_features(g)\n"
                "    graph_features = transaction_entity_features(\n"
                '        df, entity_cols=("device_fingerprint", "card_entity_id"), entity_features=entity_features\n'
                "    )\n"
                "    df = pd.concat([df, graph_features], axis=1)",
            ),
        ],
    )

    modeling_evaluation = build_notebook(
        "SentryNet -- Modeling and Evaluation",
        [
            (
                "Temporal train/test split, merchant risk encoding (train-only fit), and training",
                "if DATA_AVAILABLE:\n"
                "    from sentrynet.data.split import temporal_split\n"
                "    from sentrynet.features.merchant_risk import MerchantRiskEncoder\n"
                "    from sentrynet.modeling.train import train_model\n\n"
                '    train_df, test_df = temporal_split(df, time_col="TransactionDT")\n\n'
                '    risk_encoder = MerchantRiskEncoder(category_col="ProductCD").fit(train_df)\n'
                "    train_df = train_df.assign(merchant_risk=risk_encoder.transform(train_df))\n"
                "    test_df = test_df.assign(merchant_risk=risk_encoder.transform(test_df))\n\n"
                '    feature_cols = ["TransactionAmt", "dist1", "dist2", "velocity_1h", "time_since_last",\n'
                '                     "addr_changed", "merchant_risk",\n'
                '                     "device_fingerprint_degree", "card_entity_id_degree"]\n'
                '    model = train_model(train_df[feature_cols], train_df["isFraud"])',
            ),
            (
                "Comparison: SMOTE vs. scale_pos_weight (evaluated for completeness, not used as primary)",
                "if DATA_AVAILABLE:\n"
                "    from imblearn.over_sampling import SMOTE\n"
                "    from sentrynet.modeling.evaluate import pr_auc\n"
                "    import xgboost as xgb\n\n"
                "    X_resampled, y_resampled = SMOTE(random_state=42).fit_resample(\n"
                '        train_df[feature_cols], train_df["isFraud"]\n'
                "    )\n"
                '    smote_model = xgb.XGBClassifier(eval_metric="aucpr", random_state=42)\n'
                "    smote_model.fit(X_resampled, y_resampled)\n"
                "    smote_scores = smote_model.predict_proba(test_df[feature_cols])[:, 1]\n"
                '    print("PR-AUC (SMOTE):", pr_auc(test_df["isFraud"], smote_scores))\n'
                '    print("PR-AUC (scale_pos_weight):", pr_auc(test_df["isFraud"], model.predict_proba(test_df[feature_cols])[:, 1]))',
            ),
            (
                "Evaluation: PR-AUC and cost-based threshold",
                "if DATA_AVAILABLE:\n"
                "    from sentrynet.modeling.evaluate import pr_auc, select_threshold_by_cost\n\n"
                "    test_scores = model.predict_proba(test_df[feature_cols])[:, 1]\n"
                '    print("PR-AUC:", pr_auc(test_df["isFraud"], test_scores))\n'
                "    best_threshold, best_cost = select_threshold_by_cost(\n"
                '        test_df["isFraud"], test_scores, cost_fp=5.0, cost_fn=100.0\n'
                "    )\n"
                '    print("Best threshold:", best_threshold, "cost:", best_cost)',
            ),
        ],
    )

    drift_detection = build_notebook(
        "SentryNet -- Concept Drift Detection",
        [
            (
                "Real drift across the temporal split",
                "if DATA_AVAILABLE:\n"
                "    from sentrynet.drift.monitor import population_stability_index, ks_drift_test, is_drifted\n\n"
                '    psi = population_stability_index(train_df["TransactionAmt"], test_df["TransactionAmt"])\n'
                '    print("PSI on TransactionAmt:", psi, "drifted:", is_drifted(psi))',
            ),
            (
                "Synthetic drift injection (controlled demo)",
                "if DATA_AVAILABLE:\n"
                "    from sentrynet.drift.synthetic import perturb_amount\n"
                "    from sentrynet.drift.monitor import population_stability_index\n\n"
                "    perturbed = perturb_amount(test_df, multiplier=3.0)\n"
                '    psi_injected = population_stability_index(train_df["TransactionAmt"], perturbed["TransactionAmt"])\n'
                '    print("PSI after synthetic injection:", psi_injected)',
            ),
        ],
    )

    nbf.write(eda, notebooks_dir / "01_eda.ipynb")
    nbf.write(feature_engineering, notebooks_dir / "02_feature_engineering.ipynb")
    nbf.write(modeling_evaluation, notebooks_dir / "03_modeling_evaluation.ipynb")
    nbf.write(drift_detection, notebooks_dir / "04_drift_detection.ipynb")


if __name__ == "__main__":
    main()
