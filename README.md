# Random Forest Classification with MLflow

A hands-on example for learning the **Random Forest** algorithm, using
[scikit-learn](https://scikit-learn.org/) for modeling and
[MLflow](https://mlflow.org/) to track parameters, metrics, and model
artifacts.

## 1. What is a Random Forest?

A Random Forest is an **ensemble of decision trees**. A single decision tree
tends to overfit the data it was trained on (low bias, high variance).
Random Forest fixes this with two sources of randomness, then averages
across many trees to get a model that generalizes much better:

1. **Bagging (Bootstrap Aggregating)** — each tree is trained on a random
   sample of the training rows, drawn *with replacement*. Each tree sees a
   slightly different dataset.
2. **Random feature subsets** — at every split, the tree only considers a
   random subset of the features (controlled by `max_features`), instead of
   all of them. This decorrelates the trees so they don't all make the same
   mistakes.

For **classification**, the forest's prediction is the majority vote across
all trees. Because rows left out of a tree's bootstrap sample ("out-of-bag",
~37% of the data) were never seen by that tree, scikit-learn can use them as
a free internal validation set — this is the `oob_score`.

Key hyperparameters you'll see in this example:

| Parameter | What it controls |
|---|---|
| `n_estimators` | Number of trees in the forest. More trees = more stable predictions, at the cost of compute. |
| `max_depth` | Maximum depth of each tree. Shallower trees underfit; deeper trees can overfit. |
| `min_samples_split` / `min_samples_leaf` | Minimum samples required to split a node / to be a leaf. Higher values = simpler, more regularized trees. |
| `max_features` | How many features are considered at each split (`sqrt`, `log2`, or an int/float). |

A big practical bonus: after training, `model.feature_importances_` tells
you which input features drove the predictions the most — useful both for
debugging and for explaining the model to others.

## 2. The example dataset

This example intentionally does **not** use a toy dataset like Iris or
Wine — those are so easy that Random Forest hits ~98-100% accuracy
regardless of hyperparameters, which makes MLflow's run-comparison view
boring (every run looks the same). Instead, `src/train.py` generates a
synthetic dataset with `sklearn.datasets.make_classification` that is
diverse and genuinely hard, so **the hyperparameters you choose actually
change the metrics**:

- **3,000 samples, 4 classes, imbalanced** (`weights=[0.4, 0.3, 0.2, 0.1]`) —
  mimics real-world class imbalance, where accuracy alone is misleading and
  per-class precision/recall (visible in the confusion matrix and
  classification report) matter.
- **20 features, only 10 carry signal**: 6 *informative* features, 4
  *redundant* features (linear combinations of the informative ones), and
  10 pure-noise features. The feature importance plot color-codes these
  three groups, so you can visually confirm Random Forest correctly learns
  to ignore the noise.
- **5% label noise** (`flip_y=0.05`) — some training labels are randomly
  wrong, like real annotated data often is.

With this dataset, sweeping over hyperparameters produces real spread —
e.g. `accuracy` ranges roughly 0.62–0.82 and `f1_macro` roughly 0.50–0.79
across the grid in `sweep.py`, instead of every run scoring ~1.0.

## 3. Project layout

```
classification-algo/
├── requirements.txt
├── src/
│   ├── train.py   # trains one Random Forest run and logs it to MLflow
│   └── sweep.py   # runs a small hyperparameter grid, one MLflow run per combo
└── README.md
```

## 4. Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 5. Train a single model (with MLflow tracking)

```bash
python src/train.py
```

This will:
1. Generate the synthetic dataset and split it 70/30 into train/test
   (stratified, so class imbalance is preserved in both splits).
2. Train a `RandomForestClassifier` with the given hyperparameters.
3. Log to MLflow, under the `diverse-random-forest` experiment:
   - **Params**: `n_estimators`, `max_depth`, `min_samples_split`,
     `min_samples_leaf`, `max_features`, `random_state`, `test_size`, plus
     the dataset generation settings (sample count, class count, noise
     level, class weights) so every run records exactly what data it saw.
   - **Metrics**: `accuracy`, `precision_macro`, `recall_macro`,
     `f1_macro`, `f1_weighted`, `oob_score`
   - **Artifacts**: a text classification report, a confusion matrix plot,
     a feature importance plot (color-coded by informative/redundant/noise),
     and the trained model itself (loadable via `mlflow.sklearn.load_model`)

Try different hyperparameters from the CLI, e.g.:

```bash
python src/train.py --n-estimators 10 --max-depth 3 --run-name weak-forest
python src/train.py --n-estimators 200 --max-depth 20 --run-name strong-forest
```

Compare the two runs' `accuracy`/`f1_macro` in the MLflow UI to see the
effect directly.

## 6. Compare many runs: a hyperparameter sweep

```bash
python src/sweep.py
```

This trains 18 Random Forests (varying `n_estimators`, `max_depth`,
`max_features`) on the same diverse dataset and logs each as its own MLflow
run in the same experiment. This is where MLflow really pays off: instead
of scrolling through printed logs, you can open the UI and sort/filter/plot
all runs side by side — and because the dataset is hard, the runs actually
disagree with each other.

## 7. View results in the MLflow UI

```bash
mlflow ui
```

Then open http://127.0.0.1:5000 in your browser. From there you can:
- See the `diverse-random-forest` experiment with every run you've logged.
- Sort runs by `f1_macro` or `accuracy` to find the best hyperparameters.
- Click into a run to see its params, metrics, confusion matrix, feature
  importance plot, and download the logged model.
- Use the "Compare" view to plot metrics against a hyperparameter (e.g.
  `f1_macro` vs `max_depth`) across all sweep runs — you'll see a clear
  upward trend as `max_depth` increases from 3 to `None`.

By default MLflow stores run metadata in a local `mlflow.db` SQLite file and
artifacts under `mlruns/`, both created automatically the first time you run
`train.py` or `sweep.py` — no external server needed.

## 8. Saving artifacts to S3

By default, artifacts (plots, the classification report, the model) are
written to the local `mlruns/` folder. To store them in an S3 bucket
instead, set these environment variables before running `train.py` or
`sweep.py`:

```bash
export MLFLOW_S3_BUCKET=your-bucket-name     # required to enable S3
export MLFLOW_S3_PREFIX=random-forest-mlflow # optional, defaults to this

# Standard AWS credentials — any of the usual boto3 methods work:
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_DEFAULT_REGION=us-east-1
# (or export AWS_PROFILE=your-profile to use ~/.aws/credentials instead)
```

Then run as usual:

```bash
python src/train.py
```

New runs will be stored under `s3://your-bucket-name/random-forest-mlflow/...`.

**Notes:**
- MLflow only lets you set an experiment's artifact location at creation
  time. If you already ran `train.py` locally (creating the
  `diverse-random-forest` experiment with a local artifact location),
  setting `MLFLOW_S3_BUCKET` afterwards won't retroactively move it — delete
  the experiment in the MLflow UI (or use a different `EXPERIMENT_NAME`)
  before switching to S3, so it gets created fresh with the S3 location.
- You need an S3 bucket that already exists (this project doesn't create
  one for you) and credentials with at least `s3:PutObject`, `s3:GetObject`,
  `s3:ListBucket`, and `s3:DeleteObject` on that bucket:
  ```bash
  aws s3 mb s3://your-bucket-name
  ```
- `mlflow ui` also needs the same AWS credentials available in its
  environment, since it fetches artifacts (plots, models) directly from S3
  when you view a run.
- For an S3-compatible store (e.g. MinIO) instead of AWS, also set
  `MLFLOW_S3_ENDPOINT_URL=http://your-host:9000`.

## 9. Loading a saved model later

Every run's model is logged as an MLflow model artifact, so you can reload
it (e.g. in a notebook or another script) without re-training:

```python
import mlflow.sklearn

model = mlflow.sklearn.load_model("runs:/<RUN_ID>/model")
predictions = model.predict(X_new)
```

You can find `<RUN_ID>` in the MLflow UI or via `mlflow.search_runs(...)`.
