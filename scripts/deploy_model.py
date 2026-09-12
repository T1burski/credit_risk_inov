import argparse

from loguru import logger
from pyspark.dbutils import DBUtils
from pyspark.sql import SparkSession

from credit_risk.config import ProjectConfig
from credit_risk.serving.model_serving import ModelServing

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

args = parser.parse_args()

config_path = f"{args.root_path}/files/project_config_credit.yml"

spark = SparkSession.builder.getOrCreate()
dbutils = DBUtils(spark)
model_version = dbutils.jobs.taskValues.get(taskKey="train_model", key="model_version")

# Load project config
config = ProjectConfig.from_yaml(config_path=config_path, env=args.env)
logger.info("Loaded config file.")

catalog_name = config.catalog_name
schema_name = config.schema_name
endpoint_name = "credit-risk-model-serving"

model_serving = ModelServing(
    model_name=f"{catalog_name}.{schema_name}.credit_risk_model_custom",
    endpoint_name=endpoint_name
)

model_serving.deploy_or_update_serving_endpoint(version=model_version)
logger.info("Started deployment/update of the Credit risk model serving endpoint.")