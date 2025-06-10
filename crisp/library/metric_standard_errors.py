from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import pandas as pd


def bootstrap_accuracy(
    df,
    id_col="participant_id",
    y_true_col="human_code",
    y_pred_col="model_code",
    n_bootstraps=1000,
    random_state=12,
):
    rng = np.random.default_rng(seed=random_state)
    unique_ids = df[id_col].unique()
    n_ids = len(unique_ids)
    scores = []

    for _ in range(n_bootstraps):
        sampled_ids = rng.choice(unique_ids, size=n_ids, replace=True)
        sample_df = df[df[id_col].isin(sampled_ids)]
        scores.append(accuracy_score(sample_df[y_true_col], sample_df[y_pred_col]))

    scores = np.array(scores)
    return (
        scores.mean(),
        scores.std(ddof=1),
        (np.percentile(scores, 2.5), np.percentile(scores, 97.5)),
    )


def bootstrap_precision(
    df,
    id_col="participant_id",
    y_true_col="human_code",
    y_pred_col="model_code",
    n_bootstraps=1000,
    random_state=12,
):
    rng = np.random.default_rng(seed=random_state)
    unique_ids = df[id_col].unique()
    n_ids = len(unique_ids)
    scores = []

    for _ in range(n_bootstraps):
        sampled_ids = rng.choice(unique_ids, size=n_ids, replace=True)
        sample_df = df[df[id_col].isin(sampled_ids)]
        scores.append(precision_score(sample_df[y_true_col], sample_df[y_pred_col]))

    scores = np.array(scores)
    return (
        scores.mean(),
        scores.std(ddof=1),
        (np.percentile(scores, 2.5), np.percentile(scores, 97.5)),
    )


def bootstrap_recall(
    df,
    id_col="participant_id",
    y_true_col="human_code",
    y_pred_col="model_code",
    n_bootstraps=1000,
    random_state=12,
):
    rng = np.random.default_rng(seed=random_state)
    unique_ids = df[id_col].unique()
    n_ids = len(unique_ids)
    scores = []

    for _ in range(n_bootstraps):
        sampled_ids = rng.choice(unique_ids, size=n_ids, replace=True)
        sample_df = df[df[id_col].isin(sampled_ids)]
        scores.append(recall_score(sample_df[y_true_col], sample_df[y_pred_col]))

    scores = np.array(scores)
    return (
        scores.mean(),
        scores.std(ddof=1),
        (np.percentile(scores, 2.5), np.percentile(scores, 97.5)),
    )


def bootstrap_f1(
    df,
    id_col="participant_id",
    y_true_col="human_code",
    y_pred_col="model_code",
    n_bootstraps=1000,
    random_state=12,
):
    rng = np.random.default_rng(seed=random_state)
    unique_ids = df[id_col].unique()
    n_ids = len(unique_ids)
    scores = []

    for _ in range(n_bootstraps):
        sampled_ids = rng.choice(unique_ids, size=n_ids, replace=True)
        sample_df = df[df[id_col].isin(sampled_ids)]
        scores.append(f1_score(sample_df[y_true_col], sample_df[y_pred_col]))

    scores = np.array(scores)
    return (
        scores.mean(),
        scores.std(ddof=1),
        (np.percentile(scores, 2.5), np.percentile(scores, 97.5)),
    )


def bootstrap_f1_micro(
    df,
    id_col="participant_id",
    y_true_col="human_code",
    y_pred_col="model_code",
    n_bootstraps=1000,
    random_state=12,
):
    rng = np.random.default_rng(seed=random_state)
    unique_ids = df[id_col].unique()
    n_ids = len(unique_ids)
    scores = []

    for _ in range(n_bootstraps):
        sampled_ids = rng.choice(unique_ids, size=n_ids, replace=True)
        sample_df = df[df[id_col].isin(sampled_ids)]
        scores.append(
            f1_score(sample_df[y_true_col], sample_df[y_pred_col], average="micro")
        )

    scores = np.array(scores)
    return (
        scores.mean(),
        scores.std(ddof=1),
        (np.percentile(scores, 2.5), np.percentile(scores, 97.5)),
    )


def bootstrap_f1_macro(
    df,
    id_col="participant_id",
    y_true_col="human_code",
    y_pred_col="model_code",
    n_bootstraps=1000,
    random_state=12,
):
    rng = np.random.default_rng(seed=random_state)
    unique_ids = df[id_col].unique()
    n_ids = len(unique_ids)
    scores = []

    for _ in range(n_bootstraps):
        sampled_ids = rng.choice(unique_ids, size=n_ids, replace=True)
        sample_df = df[df[id_col].isin(sampled_ids)]
        scores.append(
            f1_score(sample_df[y_true_col], sample_df[y_pred_col], average="macro")
        )

    scores = np.array(scores)
    return (
        scores.mean(),
        scores.std(ddof=1),
        (np.percentile(scores, 2.5), np.percentile(scores, 97.5)),
    )
