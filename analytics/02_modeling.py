"""Q2 Part B: Titanic modeling, evaluation, tuning, regression, and reporting."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree


MODULE_DIR = Path(__file__).resolve().parent
CHART_DIR = MODULE_DIR / "charts"
CLEANED_CSV_PATH = MODULE_DIR / "titanic_cleaned.csv"
EDA_REPORT_PATH = MODULE_DIR / "eda_report.md"
MODELING_REPORT_PATH = MODULE_DIR / "modeling_report.md"
README_PATH = MODULE_DIR / "README.md"
BEST_PIPELINE_PATH = MODULE_DIR / "best_classifier_pipeline.joblib"

RANDOM_STATE = 42
NUMERIC_FEATURES = ["pclass", "age", "sibsp", "parch"]
CATEGORICAL_FEATURES = ["sex", "embarked"]
CLASSIFICATION_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def save_plot(path: Path) -> None:
    """Save the current figure and close it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def build_preprocessor() -> ColumnTransformer:
    """Create train-only preprocessing for numeric and categorical features."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def build_classifier_pipeline(model: Any) -> Pipeline:
    """Wrap preprocessing and classifier in one leak-safe pipeline."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", model),
        ]
    )


def evaluate_classifier(name: str, pipeline: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> dict[str, Any]:
    """Calculate the full required classification metric suite."""
    predictions = pipeline.predict(x_test)
    probabilities = pipeline.predict_proba(x_test)[:, 1]
    matrix = confusion_matrix(y_test, predictions)

    return {
        "model": name,
        "confusion_matrix": matrix.tolist(),
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "auc": roc_auc_score(y_test, probabilities),
        "probabilities": probabilities,
        "predictions": predictions,
    }


def plot_confusion_matrices(metrics: list[dict[str, Any]], y_test: pd.Series) -> None:
    """Save one confusion-matrix image for each classifier."""
    for metric in metrics:
        matrix = np.array(metric["confusion_matrix"])
        display = ConfusionMatrixDisplay(matrix, display_labels=["Not survived", "Survived"])
        display.plot(cmap="Blues", values_format="d")
        plt.title(f"Confusion Matrix: {metric['model']}")
        save_plot(CHART_DIR / f"confusion_matrix_{metric['model'].lower().replace(' ', '_')}.png")


def plot_roc_curves(metrics: list[dict[str, Any]], y_test: pd.Series) -> None:
    """Save a combined ROC curve for the three classifiers."""
    plt.figure(figsize=(8, 6))
    for metric in metrics:
        false_positive_rate, true_positive_rate, _ = roc_curve(y_test, metric["probabilities"])
        plt.plot(
            false_positive_rate,
            true_positive_rate,
            label=f"{metric['model']} (AUC={metric['auc']:.3f})",
        )
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend()
    save_plot(CHART_DIR / "roc_curves.png")


def plot_decision_tree_model(pipeline: Pipeline) -> None:
    """Render the trained decision tree with transformed feature names."""
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]
    feature_names = preprocessor.get_feature_names_out()

    plt.figure(figsize=(22, 10))
    plot_tree(
        classifier,
        feature_names=feature_names,
        class_names=["Not survived", "Survived"],
        filled=True,
        rounded=True,
        max_depth=4,
        fontsize=8,
    )
    plt.title("Decision Tree Classifier")
    save_plot(CHART_DIR / "decision_tree.png")


def run_imbalance_comparison(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> pd.DataFrame:
    """Compare baseline, class weighting, and SMOTE on the training fold only."""
    variants = {
        "baseline_logistic_regression": build_classifier_pipeline(
            LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
        ),
        "class_weight_balanced": build_classifier_pipeline(
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)
        ),
        "smote_training_only": ImbPipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("smote", SMOTE(random_state=RANDOM_STATE)),
                ("classifier", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
            ]
        ),
    }

    rows = []
    for name, pipeline in variants.items():
        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_test)
        rows.append(
            {
                "variant": name,
                "precision": precision_score(y_test, predictions, zero_division=0),
                "recall": recall_score(y_test, predictions, zero_division=0),
                "f1": f1_score(y_test, predictions, zero_division=0),
            }
        )

    return pd.DataFrame(rows).round(4)


def run_grid_search(x_train: pd.DataFrame, y_train: pd.Series) -> tuple[GridSearchCV, dict[str, Any], float]:
    """Tune Random Forest and report the OOB score from an oob_score=True estimator."""
    pipeline = build_classifier_pipeline(
        RandomForestClassifier(
            random_state=RANDOM_STATE,
            oob_score=True,
            bootstrap=True,
        )
    )
    parameter_grid = {
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [3, 5, None],
        "classifier__max_features": ["sqrt", "log2"],
    }
    grid_search = GridSearchCV(
        pipeline,
        parameter_grid,
        scoring="f1",
        cv=3,
        n_jobs=-1,
    )
    grid_search.fit(x_train, y_train)
    best_rf = grid_search.best_estimator_.named_steps["classifier"]
    return grid_search, grid_search.best_params_, float(best_rf.oob_score_)


def run_regression_side_task(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Predict fare with multivariate linear regression and save a residual plot."""
    regression_features = ["survived", "pclass", "age", "sibsp", "parch", "sex", "embarked"]
    regression_target = "fare"
    regression_df = df[regression_features + [regression_target]].dropna().copy()

    x = regression_df[regression_features]
    y = regression_df[regression_target]
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    numeric_features = ["survived", "pclass", "age", "sibsp", "parch"]
    categorical_features = ["sex", "embarked"]
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical_features,
            ),
        ]
    )
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", LinearRegression()),
        ]
    )
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)
    residuals = y_test - predictions

    n_rows = len(y_test)
    n_features = pipeline.named_steps["preprocessor"].transform(x_test).shape[1]
    r2 = r2_score(y_test, predictions)
    adjusted_r2 = 1 - (1 - r2) * (n_rows - 1) / (n_rows - n_features - 1)
    rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))

    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=predictions, y=residuals)
    plt.axhline(0, color="red", linestyle="--")
    plt.xlabel("Predicted Fare")
    plt.ylabel("Residual")
    plt.title("Regression Residual Plot")
    save_plot(CHART_DIR / "regression_residuals.png")

    spread_low = float(np.abs(residuals[predictions <= np.median(predictions)]).mean())
    spread_high = float(np.abs(residuals[predictions > np.median(predictions)]).mean())
    heteroscedasticity = (
        "The residual plot suggests heteroscedasticity because residual spread changes between lower and higher predicted fares."
        if spread_high > spread_low * 1.25 or spread_low > spread_high * 1.25
        else "The residual plot does not show strong heteroscedasticity because the residual spread is fairly similar across predicted fare ranges."
    )

    metrics = pd.DataFrame(
        [
            {
                "model": "Linear Regression",
                "MAE": mean_absolute_error(y_test, predictions),
                "RMSE": rmse,
                "R2": r2,
                "Adjusted_R2": adjusted_r2,
            }
        ]
    ).round(4)

    return metrics, heteroscedasticity


def choose_best_classifier(metrics_df: pd.DataFrame, trained_pipelines: dict[str, Pipeline]) -> tuple[str, Pipeline]:
    """Select the classifier with the highest F1 score for deployment."""
    best_model_name = str(metrics_df.sort_values("f1", ascending=False).iloc[0]["model"])
    return best_model_name, trained_pipelines[best_model_name]


def write_modeling_report(
    class_balance: pd.DataFrame,
    classification_metrics: pd.DataFrame,
    imbalance_results: pd.DataFrame,
    imbalance_conclusion: str,
    best_params: dict[str, Any],
    oob_score: float,
    regression_metrics: pd.DataFrame,
    heteroscedasticity: str,
    best_model_name: str,
    reload_predictions: list[int],
) -> None:
    """Write modeling outputs and recommendations to markdown."""
    best_row = classification_metrics.loc[classification_metrics["model"] == best_model_name].iloc[0]
    recommendation = (
        f"I would deploy {best_model_name} because it produced the strongest F1 score "
        f"({best_row['f1']:.3f}) among the three baseline classifiers while also achieving "
        f"accuracy {best_row['accuracy']:.3f}, precision {best_row['precision']:.3f}, "
        f"recall {best_row['recall']:.3f}, and AUC {best_row['auc']:.3f}. "
        "F1 is a good primary metric here because the survived and not-survived classes are not perfectly balanced. "
        "The saved joblib artifact contains preprocessing plus the classifier, so it can predict from raw feature rows."
    )

    final_table = classification_metrics[
        ["model", "accuracy", "precision", "recall", "f1", "auc"]
    ].copy()
    final_table["metric_group"] = "classification"
    regression_table = regression_metrics.copy()
    regression_table["metric_group"] = "regression"

    lines = [
        "# Q2 Part B: Modeling Report",
        "",
        "## Class Balance and Split",
        "",
        class_balance.to_markdown(index=False),
        "",
        "A stratified train/test split is used before preprocessing so the survived/not-survived class ratio stays similar in both folds.",
        "",
        "## Preprocessing",
        "",
        "Numeric features are median-imputed and scaled with StandardScaler. Categorical features are most-frequent-imputed and one-hot encoded. These steps are inside scikit-learn Pipeline/ColumnTransformer objects, so they are fit only on training data and only transformed on test data.",
        "",
        "## Classifier Comparison",
        "",
        classification_metrics[["model", "confusion_matrix", "accuracy", "precision", "recall", "f1", "auc"]].to_markdown(index=False),
        "",
        "## Imbalance Handling Comparison",
        "",
        imbalance_results.to_markdown(index=False),
        "",
        imbalance_conclusion,
        "",
        "## Random Forest GridSearchCV",
        "",
        f"- Best parameters: `{best_params}`",
        f"- OOB score from best RandomForestClassifier(oob_score=True): {oob_score:.4f}",
        "",
        "## Regression Side Task",
        "",
        regression_metrics.to_markdown(index=False),
        "",
        heteroscedasticity,
        "",
        "## Final Model Comparison Table",
        "",
        "Classification metrics and regression metrics are shown as separate metric groups because they are on different scales.",
        "",
        "### Classification Metrics",
        "",
        final_table.to_markdown(index=False),
        "",
        "### Regression Metrics",
        "",
        regression_table.to_markdown(index=False),
        "",
        "## Final Recommendation",
        "",
        recommendation,
        "",
        "## Saved Pipeline Reload Check",
        "",
        f"- Saved artifact: `{BEST_PIPELINE_PATH.name}`",
        f"- Reloaded pipeline predictions on five raw rows: {reload_predictions}",
        "",
    ]

    MODELING_REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def write_final_readme() -> None:
    """Combine instructions, EDA report, and modeling report into the module README."""
    eda_text = EDA_REPORT_PATH.read_text(encoding="utf-8")
    modeling_text = MODELING_REPORT_PATH.read_text(encoding="utf-8")
    lines = [
        "# Q2 Analytics Pipeline",
        "",
        "This module completes the Titanic analytics and modeling task for the Zepto capstone project.",
        "",
        "## Install",
        "",
        "```bash",
        "python -m pip install -r requirements.txt",
        "```",
        "",
        "## Run",
        "",
        "```bash",
        "python run_all.py",
        "```",
        "",
        "The raw Titanic dataset is loaded exactly once in `01_eda.py` using `sns.load_dataset('titanic')`, then immediately saved as `titanic.csv`. The modeling script reads `titanic_cleaned.csv` and does not load the raw dataset again.",
        "",
        "## Generated Files",
        "",
        "- `titanic.csv`: raw offline fallback.",
        "- `titanic_cleaned.csv`: cleaned dataset used by modeling.",
        "- `charts/`: EDA and modeling chart images.",
        "- `best_classifier_pipeline.joblib`: saved preprocessing plus classifier pipeline.",
        "- `eda_report.md` and `modeling_report.md`: detailed generated reports.",
        "",
        eda_text,
        "",
        modeling_text,
        "",
    ]
    README_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Run Part B modeling from cleaned data through final README generation."""
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    if not CLEANED_CSV_PATH.exists():
        raise FileNotFoundError("Run 01_eda.py first so titanic_cleaned.csv exists.")

    df = pd.read_csv(CLEANED_CSV_PATH)
    x = df[CLASSIFICATION_FEATURES]
    y = df["survived"]

    # Required Step 7: stratified split before any preprocessing.
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    class_balance = (
        y.value_counts()
        .rename_axis("survived")
        .reset_index(name="count")
        .assign(percent=lambda table: (table["count"] / len(y) * 100).round(2))
        .sort_values("survived")
    )

    # Required Step 8-10: train three classifiers on the same split and evaluate all metrics.
    model_specs = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
    }
    trained_pipelines: dict[str, Pipeline] = {}
    metric_rows: list[dict[str, Any]] = []
    for model_name, model in model_specs.items():
        pipeline = build_classifier_pipeline(model)
        pipeline.fit(x_train, y_train)
        trained_pipelines[model_name] = pipeline
        metric_rows.append(evaluate_classifier(model_name, pipeline, x_test, y_test))

    plot_confusion_matrices(metric_rows, y_test)
    plot_roc_curves(metric_rows, y_test)
    plot_decision_tree_model(trained_pipelines["Decision Tree"])

    classification_metrics = pd.DataFrame(metric_rows)
    report_metrics = classification_metrics.drop(columns=["probabilities", "predictions"]).copy()
    for column in ["accuracy", "precision", "recall", "f1", "auc"]:
        report_metrics[column] = report_metrics[column].round(4)

    # Required Step 11: compare imbalance strategies; SMOTE is inside an imblearn pipeline after preprocessing.
    imbalance_results = run_imbalance_comparison(x_train, x_test, y_train, y_test)
    best_imbalance = imbalance_results.sort_values("f1", ascending=False).iloc[0]
    imbalance_conclusion = (
        f"The best imbalance strategy by F1 was {best_imbalance['variant']} "
        f"with F1 {best_imbalance['f1']:.4f}. This comparison keeps SMOTE inside the training pipeline only, "
        "so synthetic examples are never created from the test fold."
    )

    # Required Step 12: tune Random Forest and report OOB score.
    grid_search, best_params, oob_score = run_grid_search(x_train, y_train)

    # Required Step 13: regression side task.
    regression_metrics, heteroscedasticity = run_regression_side_task(df)

    # Required Step 15: save and reload the best complete classifier pipeline.
    best_model_name, best_pipeline = choose_best_classifier(report_metrics, trained_pipelines)
    joblib.dump(best_pipeline, BEST_PIPELINE_PATH)
    reloaded_pipeline = joblib.load(BEST_PIPELINE_PATH)
    reload_predictions = reloaded_pipeline.predict(x_test.head(5)).astype(int).tolist()

    write_modeling_report(
        class_balance,
        report_metrics,
        imbalance_results,
        imbalance_conclusion,
        best_params,
        oob_score,
        regression_metrics,
        heteroscedasticity,
        best_model_name,
        reload_predictions,
    )
    write_final_readme()

    print(f"Saved modeling report: {MODELING_REPORT_PATH}")
    print(f"Saved final README: {README_PATH}")
    print(f"Saved complete best pipeline: {BEST_PIPELINE_PATH}")


if __name__ == "__main__":
    main()
