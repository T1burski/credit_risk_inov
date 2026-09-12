import argparse
import yaml
from loguru import logger
from pyspark.sql import SparkSession
import pandas as pd

from credit_risk.config import ProjectConfig
from credit_risk.data_processor import DataProcessor

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

config = ProjectConfig.from_yaml(config_path=config_path, env=args.env)

logger.info("Configuration loaded:")
logger.info(yaml.dump(config, default_flow_style=False))

# Load the Marvel characters dataset
spark = SparkSession.builder.getOrCreate()

df = spark.table("bigquery_credit_analytics_catalog.credit_analytics.loan_data")

logger.info("Credit data loaded for processing.")

# Initialize DataProcessor
data_processor = DataProcessor(df, config, spark)

# Preprocess the data
data_processor.preprocess()

# Split the data
df_train, df_validation_val, df_validation_test, df_test = data_processor.split_data()
logger.info(f"Training set shape: {str(df_train.shape)}")
logger.info(f"Validation val set shape: {str(df_validation_val.shape)}")
logger.info(f"Validation test set shape: {str(df_validation_test.shape)}")
logger.info(f"Test set shape: {df_test.shape}")

# Save to catalog
logger.info("Saving data to catalog")
data_processor.save_to_catalog(df_train, df_validation_val, df_validation_test, df_test)