"""Run a small hyperparameter sweep for the RandomForest classifier on the
diverse synthetic dataset, logging every combination as its own MLflow run
so results can be compared in the MLflow UI.

Because the dataset (see train.py: imbalanced classes, label noise,
informative/redundant/noise features) is genuinely harder than a toy
dataset, these runs produce real spread in accuracy/f1 instead of every
combination scoring ~1.0 — which is what makes comparing runs in MLflow
worthwhile.

Usage:
    python src/sweep.py
"""
import itertools

import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

from train import EXPERIMENT_NAME, load_data

PARAM_GRID = {
    "n_estimators": [10, 50, 200],
    "max_depth": [3, 8, None],
    "max_features": ["sqrt", 1.0],
}


def main():
    X_train, X_test, y_train, y_test, class_names = load_data(
        test_size=0.3, random_state=42
    )

    mlflow.set_experiment(EXPERIMENT_NAME)

    keys = list(PARAM_GRID.keys())
    combos = list(itertools.product(*PARAM_GRID.values()))
    print(f"Running sweep over {len(combos)} hyperparameter combinations...")

    best_f1 = -1.0
    best_params = None

    for combo in combos:
        params = dict(zip(keys, combo))
        run_name = "_".join(f"{k}={v}" for k, v in params.items())

        with mlflow.start_run(run_name=run_name):
            mlflow.log_params(params)

            model = RandomForestClassifier(random_state=42, **params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="macro")
            mlflow.log_metrics({"accuracy": accuracy, "f1_macro": f1})

            if f1 > best_f1:
                best_f1 = f1
                best_params = params

            print(f"{run_name} -> accuracy={accuracy:.3f} f1_macro={f1:.3f}")

    print(f"\nBest params: {best_params} (f1_macro={best_f1:.3f})")


if __name__ == "__main__":
    main()
