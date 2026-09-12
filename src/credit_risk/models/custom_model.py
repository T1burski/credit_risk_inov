from datetime import datetime

import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from mlflow.models import infer_signature
from mlflow.pyfunc import PythonModelContext
from mlflow.utils.environment import _mlflow_conda_env

from credit_risk.config import Tags


class CreditModelWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context: PythonModelContext) -> None:
        self.model = mlflow.pyfunc.load_model(context.artifacts["pd-pipeline"])

    def predict(self, context: PythonModelContext, model_input: pd.DataFrame | np.ndarray) -> dict:
        predictions = self.model.predict(model_input)
        return {"Probability of Default": [round(pred, 3) for pred in predictions]}

    def log_register_model(
        self,
        wrapped_model_uri: str,
        pyfunc_model_name: str,
        experiment_name: str,
        tags: Tags,
        code_paths: list[str],
        input_example: pd.DataFrame,
    ) -> None:
        mlflow.set_experiment(experiment_name=experiment_name)
        with mlflow.start_run(run_name=f"wrapper-pd-{datetime.now().strftime('%Y-%m-%d')}", tags=tags.to_dict()):
            additional_pip_deps = []
            for package in code_paths:
                whl_name = package.split("/")[-1]
                additional_pip_deps.append(f"code/{whl_name}")

            conda_env = _mlflow_conda_env(additional_pip_deps=additional_pip_deps)

            signature = infer_signature(model_input=input_example, model_output={"Probability of Default": [0.112]})
            model_info = mlflow.pyfunc.log_model(
                python_model=self,
                name="pyfunc-wrapper",
                artifacts={"pd-pipeline": wrapped_model_uri},
                code_paths=code_paths,
                signature=signature,
                conda_env=conda_env,
            )

        client = MlflowClient()
        registered_model = mlflow.register_model(
            model_uri=model_info.model_uri,
            name=pyfunc_model_name,
            tags=tags.to_dict(),
        )
        latest_version = registered_model.version
        client.set_registered_model_alias(
            name=pyfunc_model_name,
            alias="latest-model",
            version=latest_version,
        )
        return latest_version
