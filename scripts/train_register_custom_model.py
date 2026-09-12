import argparse

import mlflow
from loguru import logger
from pyspark.dbutils import DBUtils
from pyspark.sql import SparkSession
from importlib.metadata import version

from credit_risk.config import ProjectConfig, Tags
from credit_risk.models.basic_model import BasicModel
from credit_risk.models.custom_model import CreditModelWrapper

parser = argparse.ArgumentParser()
parser.add_argument(
    "--root_path",
    action="store",
    default=None,
    type=str,
    required=True,
)

parser.add_argument(
    "--env",
    action="store",
    default=None,
    type=str,
    required=True,
)
parser.add_argument("--git_sha", type=str, required=True, help="git sha of the commit")
parser.add_argument("--job_run_id", type=str, required=True, help="run id of the run of the databricks job")
parser.add_argument("--branch", type=str, required=True, help="branch of the project")

args = parser.parse_args()
root_path = args.root_path
config_path = f"{root_path}/files/project_config_credit.yml"

config = ProjectConfig.from_yaml(config_path=config_path, env=args.env)
spark = SparkSession.builder.getOrCreate()
dbutils = DBUtils(spark)
tags_dict = {"git_sha": args.git_sha, "branch": args.branch, "job_run_id": args.job_run_id}
tags = Tags(**tags_dict)

basic_model = BasicModel(config=config, tags=tags, spark=spark)
logger.info("Credit BasicModel initialized.")

basic_model.load_data()
logger.info("Credit data loaded.")

# Prepare features
basic_model.prepare_model_features()

basic_model.train()
logger.info("Credit model training completed.")

basic_model.log_model()

model_improved = basic_model.model_improved()
logger.info("Credit risk model evaluation completed. Model improved: %s", model_improved)

if model_improved:
    # Register the model
    basic_model.register_model()
    credit_risk_v = version("credit_risk")

    pyfunc_model_name = f"{config.catalog_name}.{config.schema_name}.credit_risk_model_custom"
    code_paths=[f"{root_path}/artifacts/.internal/credit_risk-{credit_risk_v}-py3-none-any.whl"]

    wrapper = CreditModelWrapper()
    latest_version = wrapper.log_register_model(wrapped_model_uri=f"{basic_model.model_info.model_uri}",
                            pyfunc_model_name=pyfunc_model_name,
                            experiment_name=config.experiment_name_custom,
                            input_example=basic_model.X_test[0:1],
                            tags=tags,
                            code_paths=code_paths)

    logger.info("New model registered with version:", latest_version)
    dbutils.jobs.taskValues.set(key="model_version", value=latest_version)
    dbutils.jobs.taskValues.set(key="model_updated", value=1)

else:
    dbutils.jobs.taskValues.set(key="model_updated", value=0)