"""Leakage-free model training utilities for TurPilot Review Intelligence.

The supervised model may only see the customer review text and the Google star
rating. Rule-based aspect signals are intentionally excluded because they are
derived from the same logic used to create labels and would leak target
information into the feature matrix.
"""

from __future__ import annotations

import json
import re
import string
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import KFold, train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


BASE_DIR = Path(__file__).resolve().parent
DATASET_FILENAMES = ["dataset.csv", "updated_dataset.csv"]
TARGET_COLUMNS = ["Ulasim", "Rehber", "Organizasyon", "Otel", "Yemek"]
FEATURE_COLUMNS = ["CleanYorum", "Yildiz"]

LABEL_NAMES = {
    0: "Bahsedilmemiş",
    1: "Övgü",
    2: "Şikayet",
}

TURKISH_STOPWORDS = {
    "acaba", "ama", "aslında", "az", "bazı", "belki", "biri", "birkaç",
    "birşey", "biz", "bu", "çok", "çünkü", "da", "daha", "de", "defa",
    "diye", "eğer", "en", "gibi", "hem", "hep", "hepsi", "her", "hiç",
    "için", "ile", "ise", "kez", "ki", "kim", "mı", "mu", "mü", "nasıl",
    "ne", "neden", "nerde", "nerede", "nereye", "niçin", "niye", "o",
    "sanki", "şey", "siz", "şu", "tüm", "ve", "veya", "ya", "yani",
    "bir", "olarak", "olan", "oldu", "olduk", "olur", "oluyor", "ben",
    "bana", "beni", "bizim", "sizin", "onlar", "onların", "kadar", "sonra",
    "önce", "var", "yok", "şöyle", "böyle", "ancak", "fakat", "lakin",
}

CONTEXT_WORDS_TO_KEEP = {"çok", "hiç", "değil", "değildi", "kötü", "iyi"}


def clean_text(text: object) -> str:
    """Clean Turkish review text before TF-IDF vectorization."""
    text = "" if pd.isna(text) else str(text)
    text = text.lower()
    punctuation_pattern = f"[{re.escape(string.punctuation)}“”‘’…]"
    text = re.sub(punctuation_pattern, " ", text)
    text = re.sub(r"\d+", " ", text)

    tokens = [
        token
        for token in text.split()
        if (token not in TURKISH_STOPWORDS or token in CONTEXT_WORDS_TO_KEEP)
        and len(token) > 1
    ]
    return " ".join(tokens)


def resolve_dataset_path(base_dir: Path = BASE_DIR) -> Path:
    """Return the first available dataset path expected by the application."""
    data_path = next(
        (base_dir / filename for filename in DATASET_FILENAMES if (base_dir / filename).exists()),
        None,
    )
    if data_path is None:
        expected_files = ", ".join(DATASET_FILENAMES)
        raise FileNotFoundError(f"Dataset bulunamadı. Beklenen dosyalar: {expected_files}")
    return data_path


def prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Validate schema and create leakage-free feature columns."""
    required_columns = ["Yorum", "Yildiz", *TARGET_COLUMNS]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Eksik kolonlar: {', '.join(missing_columns)}")

    prepared_df = df.dropna(subset=["Yorum", "Yildiz", *TARGET_COLUMNS]).copy()
    prepared_df["Yildiz"] = pd.to_numeric(prepared_df["Yildiz"], errors="coerce").fillna(3).astype(float)
    prepared_df["Yildiz"] = prepared_df["Yildiz"].clip(lower=1, upper=5)

    for column in TARGET_COLUMNS:
        prepared_df[column] = pd.to_numeric(prepared_df[column], errors="coerce").fillna(0).astype(int)
        prepared_df[column] = prepared_df[column].clip(lower=0, upper=2)

    prepared_df["CleanYorum"] = prepared_df["Yorum"].apply(clean_text)
    prepared_df = prepared_df[prepared_df["CleanYorum"].str.len() > 0].copy()
    return prepared_df.reset_index(drop=True)


def load_dataset(base_dir: Path = BASE_DIR) -> pd.DataFrame:
    """Load and prepare the labelled tourism review dataset."""
    data_path = resolve_dataset_path(base_dir)
    raw_df = pd.read_csv(data_path, encoding="utf-8-sig")
    return prepare_dataset(raw_df)


def build_model() -> Pipeline:
    """Build the leakage-free text + star-rating multi-output classifier."""
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "word_tfidf",
                TfidfVectorizer(
                    max_features=20000,
                    ngram_range=(1, 3),
                    min_df=1,
                    sublinear_tf=True,
                ),
                "CleanYorum",
            ),
            (
                "char_tfidf",
                TfidfVectorizer(
                    analyzer="char_wb",
                    max_features=12000,
                    ngram_range=(3, 5),
                    min_df=1,
                    sublinear_tf=True,
                ),
                "CleanYorum",
            ),
            ("star_rating", StandardScaler(), ["Yildiz"]),
        ],
        remainder="drop",
    )

    return Pipeline(
        steps=[
            ("features", preprocessor),
            (
                "classifier",
                MultiOutputClassifier(
                    LinearSVC(
                        class_weight="balanced",
                        C=1.0,
                        max_iter=10000,
                        random_state=42,
                    )
                ),
            ),
        ]
    )


def score_predictions(y_true: pd.DataFrame, y_pred) -> dict:
    """Calculate strict and per-aspect metrics for multi-output predictions."""
    reports = {
        column: classification_report(
            y_true[column],
            y_pred[:, index],
            labels=[0, 1, 2],
            target_names=[LABEL_NAMES[0], LABEL_NAMES[1], LABEL_NAMES[2]],
            output_dict=True,
            zero_division=0,
        )
        for index, column in enumerate(TARGET_COLUMNS)
    }
    per_category_accuracy = {
        column: accuracy_score(y_true[column], y_pred[:, index])
        for index, column in enumerate(TARGET_COLUMNS)
    }

    return {
        "exact_match_accuracy": float((y_true.to_numpy() == y_pred).all(axis=1).mean()),
        "hamming_loss": float((y_true.to_numpy() != y_pred).mean()),
        "mean_category_accuracy": float(sum(per_category_accuracy.values()) / len(per_category_accuracy)),
        "mean_macro_f1": float(
            sum(report["macro avg"]["f1-score"] for report in reports.values()) / len(reports)
        ),
        "mean_weighted_f1": float(
            sum(report["weighted avg"]["f1-score"] for report in reports.values()) / len(reports)
        ),
        "per_category_accuracy": per_category_accuracy,
        "reports": reports,
    }


def train_model(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Train a final model and report a reproducible hold-out evaluation."""
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMNS]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )

    model = build_model()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = score_predictions(y_test, y_pred)
    metrics.update(
        {
            "train_size": len(X_train),
            "test_size": len(X_test),
            "feature_columns": FEATURE_COLUMNS,
            "evaluation": "holdout_80_20",
        }
    )
    return model, metrics


def cross_validate_model(df: pd.DataFrame, n_splits: int = 5, random_state: int = 42) -> dict:
    """Run K-Fold cross-validation and aggregate fold-level metrics."""
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMNS]
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    fold_metrics = []
    for fold_index, (train_index, test_index) in enumerate(splitter.split(X), start=1):
        model = build_model()
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = score_predictions(y_test, y_pred)
        metrics["fold"] = fold_index
        metrics["train_size"] = len(X_train)
        metrics["test_size"] = len(X_test)
        fold_metrics.append(metrics)

    summary_keys = [
        "exact_match_accuracy",
        "hamming_loss",
        "mean_category_accuracy",
        "mean_macro_f1",
        "mean_weighted_f1",
    ]
    summary = {}
    for key in summary_keys:
        values = [metrics[key] for metrics in fold_metrics]
        summary[f"{key}_mean"] = float(pd.Series(values).mean())
        summary[f"{key}_std"] = float(pd.Series(values).std(ddof=0))

    per_category_accuracy_mean = {}
    for category in TARGET_COLUMNS:
        values = [metrics["per_category_accuracy"][category] for metrics in fold_metrics]
        per_category_accuracy_mean[category] = float(pd.Series(values).mean())

    summary.update(
        {
            "evaluation": f"{n_splits}_fold_cross_validation",
            "n_splits": n_splits,
            "rows_used": len(df),
            "feature_columns": FEATURE_COLUMNS,
            "per_category_accuracy_mean": per_category_accuracy_mean,
            "folds": fold_metrics,
        }
    )
    return summary


def save_metrics(metrics: dict, output_path: Path) -> None:
    """Persist metrics as JSON for documentation and reproducibility."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
