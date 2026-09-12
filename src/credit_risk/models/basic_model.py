"""Basic model implementation for Credit Risk."""

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from delta.tables import DeltaTable
from loguru import logger
from mlflow import MlflowClient
from pyspark.sql import SparkSession
from scipy import stats as stat
from scipy.stats import ks_2samp
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.validation import check_is_fitted
from mlflow.exceptions import MlflowException

from credit_risk.config import ProjectConfig, Tags


class BasicModel:
    """A basic model class for Probability of Default prediction using Logistic Regression with Platt probability calibration.

    This class handles data loading, feature preparation, model training, and MLflow logging.
    """

    def __init__(self, config: ProjectConfig, tags: Tags, spark: SparkSession) -> None:
        """Initialize the model with project configuration.

        :param config: Project configuration object
        :param tags: Tags object
        :param spark: SparkSession object
        """
        self.config = config
        self.spark = spark

        # Extract settings from the config
        self.final_features = self.config.final_features
        self.target = self.config.target
        self.reference_categories = self.config.reference_categories
        self.catalog_name = self.config.catalog_name
        self.schema_name = self.config.schema_name
        self.experiment_name = self.config.experiment_name_basic
        self.model_name = f"{self.catalog_name}.{self.schema_name}.credit_risk_model_basic"
        self.tags = tags.to_dict()

    def load_data(self) -> None:
        """Load training and testing data from Delta tables.

        Splits data into features (X_train, X_test) and target (y_train, y_test).
        """
        logger.info("🔄 Loading data from Databricks tables...")
        self.train_set_spark = self.spark.table(f"{self.catalog_name}.{self.schema_name}.train_set")
        self.train_set = self.train_set_spark.toPandas()

        self.validation_val_set_spark = self.spark.table(f"{self.catalog_name}.{self.schema_name}.validation_val_set")
        self.validation_val_set = self.validation_val_set_spark.toPandas()

        self.validation_test_set_spark = self.spark.table(f"{self.catalog_name}.{self.schema_name}.validation_test_set")
        self.validation_test_set = self.validation_test_set_spark.toPandas()

        self.test_set_spark = self.spark.table(f"{self.catalog_name}.{self.schema_name}.test_set")
        self.test_set = self.test_set_spark.toPandas()

        self.X_train = self.train_set[self.final_features].astype(str)
        self.y_train = self.train_set[self.target]

        self.X_validation_val = self.validation_val_set[self.final_features].astype(str)
        self.y_validation_val = self.validation_val_set[self.target]

        self.X_validation_test = self.validation_test_set[self.final_features].astype(str)
        self.y_validation_test = self.validation_test_set[self.target]

        self.X_test = self.test_set[self.final_features].astype(str)
        self.y_test = self.test_set[self.target]

        self.eval_data = self.test_set[self.final_features + [self.target]]

        train_delta_table = DeltaTable.forName(self.spark, f"{self.catalog_name}.{self.schema_name}.train_set")

        self.train_data_version = str(train_delta_table.history().select("version").first()[0])

        test_delta_table = DeltaTable.forName(self.spark, f"{self.catalog_name}.{self.schema_name}.validation_test_set")
        self.test_data_version = str(test_delta_table.history().select("version").first()[0])

        logger.info("✅ Data successfully loaded.")

    def prepare_model_features(self) -> None:
        """Encode categorical features and define a preprocessing pipeline.

        Constructs a pipeline combining preprocessing and Logistic Regression classification model.
        """
        logger.info("🔄 Defining preprocessing pipeline...")

        class LogisticRegressionWithPValues(BaseEstimator, ClassifierMixin):
            def __init__(self, max_iter=1000):
                self.max_iter = max_iter

            def fit(self, X, y):
                self.model_ = LogisticRegression(C=1e9, max_iter=self.max_iter, solver="lbfgs")
                self.model_.fit(X, y)
                denom = 2.0 * (1.0 + np.cosh(self.model_.decision_function(X)))
                denom = np.tile(denom, (X.shape[1], 1)).T
                F_ij = np.dot((X / denom).T, X)
                Cramer_Rao = np.linalg.inv(F_ij)
                sigma_estimates = np.sqrt(np.diagonal(Cramer_Rao))
                z_scores = self.model_.coef_[0] / sigma_estimates
                self.p_values_ = [stat.norm.sf(abs(x)) * 2 for x in z_scores]
                self.coef_ = self.model_.coef_
                self.intercept_ = self.model_.intercept_
                self.classes_ = self.model_.classes_
                return self

            def predict(self, X):
                check_is_fitted(self)
                return self.model_.predict(X)

            def predict_proba(self, X):
                check_is_fitted(self)
                return self.model_.predict_proba(X)

        enc_temp = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        enc_temp.fit(self.X_train)

        drop_cats = []
        for feat, cats in zip(self.final_features, enc_temp.categories_, strict=False):
            ref = self.reference_categories[feat]

            if ref not in cats:
                raise ValueError(f"\nReference mismatch for '{feat}'.\n  Expected : '{ref}'\n  Available: {list(cats)}")
            drop_cats.append(ref)

        # ── Build and fit sklearn pipeline ─────────────────────────
        self.pipeline = Pipeline(
            steps=[
                ("encoder", OneHotEncoder(drop=drop_cats, sparse_output=False, handle_unknown="ignore")),
                ("model", LogisticRegressionWithPValues(max_iter=1000)),
            ]
        )

        logger.info("✅ Preprocessing pipeline defined.")

    def train(self) -> None:
        """Train the model."""
        logger.info("🚀 Starting training...")

        self.pipeline.fit(self.X_train, self.y_train)

        self.y_validation_val_proba_raw = self.pipeline.predict_proba(self.X_validation_val)[:, 1]
        self.y_validation_test_proba_raw = self.pipeline.predict_proba(self.X_validation_test)[:, 1]
        self.y_train_proba_raw = self.pipeline.predict_proba(self.X_train)[:, 1]

        # Fit Platt scaler on validation scores — never on training,
        # as the base model already saw that data

        from sklearn.linear_model import LogisticRegression as _LR

        self.platt_scaler = _LR(C=1e9, solver="lbfgs")
        self.platt_scaler.fit(self.y_validation_val_proba_raw.reshape(-1, 1), self.y_validation_val)

        # Calibrated probabilities
        self.y_validation_test_proba_cal = self.platt_scaler.predict_proba(
            self.y_validation_test_proba_raw.reshape(-1, 1)
        )[:, 1]

    def log_model(self) -> None:
        """Log the model using MLflow."""

        class CalibratedPDModel(mlflow.pyfunc.PythonModel):
            """Wraps the base pipeline + Platt scaler into a single MLflow model.
            Input  : DataFrame with raw (string-typed) binned feature columns
            Output : DataFrame with a single column 'pd_calibrated'
            """

            def __init__(self, pipeline, platt_scaler, features):
                self.pipeline = pipeline
                self.platt_scaler = platt_scaler
                self.features = features

            def predict(self, context, model_input):
                X_raw = model_input[self.features].astype(str)
                raw_proba = self.pipeline.predict_proba(X_raw)[:, 1]
                cal_proba = self.platt_scaler.predict_proba(raw_proba.reshape(-1, 1))[:, 1]
                return cal_proba

        def compute_ks(y_true, y_proba):
            bads = y_proba[y_true == 1]
            goods = y_proba[y_true == 0]
            ks_stat, _ = ks_2samp(bads, goods)
            return round(ks_stat, 4)

        def compute_roc_auc(y_true, y_proba):
            return round(roc_auc_score(y_true, y_proba), 4)

        # ─────────────────────────────────────────────
        # PLOTS
        # ─────────────────────────────────────────────

        def plot_roc_curve(y_true, y_proba, title="ROC Curve — Validation"):
            fpr, tpr, _ = roc_curve(y_true, y_proba)
            auc = roc_auc_score(y_true, y_proba)
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.plot(fpr, tpr, color="steelblue", lw=2, label=f"AUC = {auc:.4f}")
            ax.plot([0, 1], [0, 1], "k--", lw=1)
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title(title)
            ax.legend(loc="lower right")
            plt.tight_layout()
            return fig

        def plot_ks_curve(y_true, y_proba, title="KS Curve — Validation"):
            df_plot = pd.DataFrame({"score": y_proba, "target": y_true})
            df_plot = df_plot.sort_values("score", ascending=False).reset_index(drop=True)
            n = len(df_plot)
            df_plot["cum_bad"] = (df_plot["target"] == 1).cumsum() / (y_true == 1).sum()
            df_plot["cum_good"] = (df_plot["target"] == 0).cumsum() / (y_true == 0).sum()
            df_plot["ks"] = df_plot["cum_bad"] - df_plot["cum_good"]
            ks_idx = df_plot["ks"].abs().idxmax()
            ks_stat = df_plot["ks"].abs().max()
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.plot(np.arange(n) / n, df_plot["cum_bad"], color="firebrick", lw=2, label="Cumulative Bad")
            ax.plot(np.arange(n) / n, df_plot["cum_good"], color="steelblue", lw=2, label="Cumulative Good")
            ax.axvline(x=ks_idx / n, color="gray", linestyle="--", lw=1)
            ax.annotate(
                f"KS = {ks_stat:.4f}",
                xy=(ks_idx / n, (df_plot["cum_bad"][ks_idx] + df_plot["cum_good"][ks_idx]) / 2),
                fontsize=10,
                color="gray",
            )
            ax.set_xlabel("Population (sorted by score descending)")
            ax.set_ylabel("Cumulative Rate")
            ax.set_title(title)
            ax.legend()
            plt.tight_layout()
            return fig

        def plot_calibration_curve(y_true, y_proba_raw, y_proba_cal, n_bins=10, title="Calibration Curve — Validation"):
            """Plots raw and Platt-calibrated calibration curves on the same graph.

            Parameters
            ----------
            y_true      : array-like, true binary labels
            y_proba_raw : array-like, uncalibrated predicted probabilities
            y_proba_cal : array-like, Platt-calibrated predicted probabilities
            n_bins      : int, number of bins for calibration curve
            title       : str, plot title

            Returns
            -------
            fig         : matplotlib Figure
            ece_raw     : float, ECE before calibration
            ece_cal     : float, ECE after calibration

            """

            def _compute_curve_and_ece(y_true, y_proba):
                prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=n_bins, strategy="uniform")
                bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
                bin_sizes_all = np.histogram(y_proba, bins=bin_edges)[0]
                non_empty = bin_sizes_all > 0
                bin_sizes = bin_sizes_all[non_empty]
                ece = np.sum(np.abs(prob_true - prob_pred) * bin_sizes) / len(y_true)
                return prob_true, prob_pred, ece

            prob_true_raw, prob_pred_raw, ece_raw = _compute_curve_and_ece(y_true, y_proba_raw)
            prob_true_cal, prob_pred_cal, ece_cal = _compute_curve_and_ece(y_true, y_proba_cal)

            fig, ax = plt.subplots(figsize=(8, 6))

            # Perfect calibration reference
            ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")

            # Raw curve
            ax.plot(prob_pred_raw, prob_true_raw, "s-", color="tomato", lw=2, label=f"Raw (ECE={ece_raw:.4f})")

            # Calibrated curve
            ax.plot(
                prob_pred_cal,
                prob_true_cal,
                "s-",
                color="steelblue",
                lw=2,
                label=f"Platt calibrated (ECE={ece_cal:.4f})",
            )

            # ECE improvement annotation
            delta = ece_raw - ece_cal
            ax.annotate(
                f"ECE improvement: {delta:+.4f}",
                xy=(0.98, 0.04),
                xycoords="axes fraction",
                ha="right",
                fontsize=9,
                color="dimgray",
                bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="gray", alpha=0.8),
            )

            ax.set_xlabel("Mean Predicted Probability")
            ax.set_ylabel("Fraction of Positives")
            ax.set_title(title)
            ax.legend()
            plt.tight_layout()

            return fig, ece_raw, ece_cal

        # ─────────────────────────────────────────────
        # COEFFICIENT TABLE
        # ─────────────────────────────────────────────

        def build_coefficient_table(pipeline, final_features):
            encoder = pipeline.named_steps["encoder"]
            model = pipeline.named_steps["model"]
            feature_names = encoder.get_feature_names_out(final_features)
            coef_df = pd.DataFrame(
                {
                    "Feature_Category": feature_names,
                    "Coefficient": model.coef_[0],
                    "P_Value": model.p_values_,
                }
            )
            coef_df["Feature"] = coef_df["Feature_Category"].apply(lambda x: "_".join(x.split("_")[:-1]))
            coef_df["Category"] = coef_df["Feature_Category"].apply(lambda x: x.split("_")[-1])
            coef_df["Significant"] = coef_df["P_Value"] < 0.05

            return (
                coef_df[["Feature", "Category", "Coefficient", "P_Value", "Significant"]]
                .sort_values(["Feature", "Coefficient"], ascending=[True, False])
                .reset_index(drop=True)
            )

        mlflow.set_experiment(self.experiment_name)
        with mlflow.start_run(tags=self.tags) as run:
            self.run_id = run.info.run_id

            train_auc = compute_roc_auc(self.y_train, self.y_train_proba_raw)
            val_auc = compute_roc_auc(self.y_validation_test, self.y_validation_test_proba_cal)
            train_ks = compute_ks(self.y_train, self.y_train_proba_raw)
            val_ks = compute_ks(self.y_validation_test, self.y_validation_test_proba_cal)

            mlflow.log_metrics(
                {"train_roc_auc": train_auc, "val_roc_auc": val_auc, "train_ks": train_ks, "val_ks": val_ks}
            )

            mlflow.log_params(
                {
                    "features": str(self.final_features),
                    "n_features": len(self.final_features),
                    "target": self.target,
                    "regularization": "none (C=1e9)",
                }
            )

            coef_df = build_coefficient_table(self.pipeline, self.final_features)
            mlflow.log_text(coef_df.to_csv(index=False), "model_diagnostics/coefficients.csv")

            # ── ROC plot ───────────────────────────────────────────
            fig_roc = plot_roc_curve(self.y_validation_test, self.y_validation_test_proba_cal)
            mlflow.log_figure(fig_roc, "model_diagnostics/roc_curve_validation.png")
            plt.close(fig_roc)

            # ── KS plot ────────────────────────────────────────────
            fig_ks = plot_ks_curve(self.y_validation_test, self.y_validation_test_proba_cal)
            mlflow.log_figure(fig_ks, "model_diagnostics/ks_curve_validation.png")
            plt.close(fig_ks)

            # ── Calibration plot ──────────────────────────────────
            fig_cal, ece_raw, ece_cal = plot_calibration_curve(
                self.y_validation_test, self.y_validation_test_proba_raw, self.y_validation_test_proba_cal
            )
            mlflow.log_figure(fig_cal, "model_diagnostics/calibration_curve_validation.png")
            plt.close(fig_cal)
            mlflow.log_metric("val_ece_raw", round(ece_raw, 6))
            mlflow.log_metric("val_ece_cal", round(ece_cal, 6))

            # ── Log sklearn pipeline ──────────────────────────────

            self.calibrated_model = CalibratedPDModel(
                pipeline=self.pipeline,
                platt_scaler=self.platt_scaler,
                features=self.final_features,
            )

            train_dataset = mlflow.data.from_spark(
                self.train_set_spark,
                table_name=f"{self.catalog_name}.{self.schema_name}.train_set",
                version=self.train_data_version,
            )
            mlflow.log_input(train_dataset, context="training")

            test_dataset = mlflow.data.from_spark(
                self.validation_test_set_spark,
                table_name=f"{self.catalog_name}.{self.schema_name}.validation_test_set",
                version=self.test_data_version,
            )
            mlflow.log_input(test_dataset, context="testing")

            self.model_info = mlflow.pyfunc.log_model(
                python_model=self.calibrated_model, artifact_path="pd-model", input_example=self.X_validation_test[0:1]
            )

            self.ks_metric = val_ks

    def model_improved(self) -> bool:
        """Evaluate the model performance on the test set.

        Compares the current model with the latest registered model using KS.
        :return: True if the current model performs better, False otherwise.
        """

        def compute_ks(y_true, y_proba):
            bads = y_proba[y_true == 1]
            goods = y_proba[y_true == 0]
            ks_stat, _ = ks_2samp(bads, goods)
            return round(ks_stat, 4)

        client = MlflowClient()

        try:
            old_model_version = client.get_model_version_by_alias(
                name=self.model_name,
                alias="latest-model"
            )

        except MlflowException:
            logger.info(
                "No previous registered model found. "
                "Assuming current model is the first model."
            )
            return True

        model_uri = f"models:/{old_model_version.model_id}"

        old_model = mlflow.pyfunc.load_model(model_uri)
        y_pred_proba_old = old_model.predict(self.X_validation_test)

        ks_metric_old = compute_ks(self.y_validation_test, y_pred_proba_old)

        if self.ks_metric >= ks_metric_old:
            logger.info("Current model performs better. Returning True.")
            return True
        else:
            logger.info("Current model does not improve over latest. Returning False.")
            return False

    def register_model(self) -> None:
        """Register model in Unity Catalog."""
        logger.info("🔄 Registering the model in UC...")

        registered_model = mlflow.register_model(
            model_uri=f"runs:/{self.run_id}/pd-model",
            name=self.model_name,
            tags=self.tags,
        )
        logger.info(f"✅ Model registered as version {registered_model.version}.")

        latest_version = registered_model.version

        client = MlflowClient()
        client.set_registered_model_alias(
            name=self.model_name,
            alias="latest-model",
            version=latest_version,
        )
        return latest_version
