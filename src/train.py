"""Train a RandomForestClassifier on a diverse synthetic classification
dataset and track it with MLflow.

The dataset is deliberately harder than a toy dataset like Iris/Wine: it has
multiple classes, imbalanced class sizes, label noise, and a mix of
informative / redundant / pure-noise features. This means model quality
actually depends on the hyperparameters you choose, so the metrics logged to
MLflow move meaningfully between runs instead of clustering near 100%.

Usage:
    python src/train.py
    python src/train.py --n-estimators 300 --max-depth 8
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

EXPERIMENT_NAME = "diverse-random-forest"
ARTIFACT_DIR = "artifacts"

# Dataset "shape" — tweak these to make the problem easier/harder.
N_SAMPLES = 3000
N_FEATURES = 20
N_INFORMATIVE = 6
N_REDUNDANT = 4
N_CLASSES = 4
N_CLUSTERS_PER_CLASS = 2
CLASS_WEIGHTS = [0.40, 0.30, 0.20, 0.10]  # imbalanced on purpose
FLIP_Y = 0.05  # fraction of labels randomly flipped -> label noise
CLASS_SEP = 1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--min-samples-split", type=int, default=2)
    parser.add_argument("--min-samples-leaf", type=int, default=1)
    parser.add_argument("--max-features", default="sqrt")
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--run-name", default=None)
    return parser.parse_args()


def load_data(test_size: float, random_state: int):
    """Generate a diverse, imbalanced, noisy multi-class dataset.

    Columns are named (and left unshuffled) so they line up with how
    `make_classification` builds the feature matrix: informative features
    first, then redundant (linear combinations of informative ones), then
    pure noise. This makes the feature-importance plot directly show the
    model correctly ranking informative > redundant > noise features.
    """
    n_noise = N_FEATURES - N_INFORMATIVE - N_REDUNDANT
    X, y = make_classification(
        n_samples=N_SAMPLES,
        n_features=N_FEATURES,
        n_informative=N_INFORMATIVE,
        n_redundant=N_REDUNDANT,
        n_repeated=0,
        n_classes=N_CLASSES,
        n_clusters_per_class=N_CLUSTERS_PER_CLASS,
        weights=CLASS_WEIGHTS,
        flip_y=FLIP_Y,
        class_sep=CLASS_SEP,
        shuffle=False,  # keep informative/redundant/noise blocks in order
        random_state=random_state,
    )

    feature_names = (
        [f"informative_{i}" for i in range(N_INFORMATIVE)]
        + [f"redundant_{i}" for i in range(N_REDUNDANT)]
        + [f"noise_{i}" for i in range(n_noise)]
    )
    X = pd.DataFrame(X, columns=feature_names)
    y = pd.Series(y, name="target")
    class_names = [f"class_{i}" for i in range(N_CLASSES)]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test, class_names


def plot_confusion_matrix(model, X_test, y_test, class_names, out_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_estimator(
        model, X_test, y_test, display_labels=class_names, cmap="Blues", ax=ax
    )
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_feature_importance(model, feature_names, out_path):
    importances = pd.Series(model.feature_importances_, index=feature_names)
    importances = importances.sort_values(ascending=True)

    # Color bars by feature kind so informative vs. redundant vs. noise
    # is easy to see at a glance.
    def kind(name):
        if name.startswith("informative"):
            return "informative"
        if name.startswith("redundant"):
            return "redundant"
        return "noise"

    palette = {"informative": "#2b6cb0", "redundant": "#63b3ed", "noise": "#cbd5e0"}
    colors = [palette[kind(name)] for name in importances.index]

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.barh(importances.index, importances.values, color=colors)
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    ax.set_title("Random Forest Feature Importance\n(blue = informative, light blue = redundant, gray = noise)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main():
    args = parse_args()
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    X_train, X_test, y_train, y_test, class_names = load_data(
        args.test_size, args.random_state
    )

    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=args.run_name):
        params = {
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "min_samples_split": args.min_samples_split,
            "min_samples_leaf": args.min_samples_leaf,
            "max_features": args.max_features,
            "random_state": args.random_state,
        }
        mlflow.log_params(params)
        mlflow.log_param("test_size", args.test_size)
        mlflow.log_params(
            {
                "dataset_n_samples": N_SAMPLES,
                "dataset_n_classes": N_CLASSES,
                "dataset_flip_y": FLIP_Y,
                "dataset_class_weights": CLASS_WEIGHTS,
            }
        )

        model = RandomForestClassifier(oob_score=True, **params)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision_macro": precision_score(y_test, y_pred, average="macro"),
            "recall_macro": recall_score(y_test, y_pred, average="macro"),
            "f1_macro": f1_score(y_test, y_pred, average="macro"),
            "f1_weighted": f1_score(y_test, y_pred, average="weighted"),
            "oob_score": model.oob_score_,
        }
        mlflow.log_metrics(metrics)

        report = classification_report(y_test, y_pred, target_names=class_names)
        report_path = f"{ARTIFACT_DIR}/classification_report.txt"
        with open(report_path, "w") as f:
            f.write(report)
        mlflow.log_artifact(report_path)

        cm_path = f"{ARTIFACT_DIR}/confusion_matrix.png"
        plot_confusion_matrix(model, X_test, y_test, class_names, cm_path)
        mlflow.log_artifact(cm_path)

        fi_path = f"{ARTIFACT_DIR}/feature_importance.png"
        plot_feature_importance(model, X_train.columns, fi_path)
        mlflow.log_artifact(fi_path)

        mlflow.sklearn.log_model(model, artifact_path="model", input_example=X_train.head(3))

        print("Params:", params)
        print("Metrics:", metrics)
        print("\n" + report)


if __name__ == "__main__":
    main()
