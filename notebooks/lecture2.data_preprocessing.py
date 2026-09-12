# Databricks notebook source

# MAGIC %pip install -e ..
# MAGIC %restart_python

# COMMAND ----------
# from pathlib import Path
# import sys
# sys.path.append(str(Path.cwd().parent / 'src'))

# COMMAND ----------
import pandas as pd
import yaml
from loguru import logger
from pyspark.sql import SparkSession

from credit_risk.config import ProjectConfig
from credit_risk.data_processor import DataProcessor

config = ProjectConfig.from_yaml(config_path="../project_config_credit.yml", env="dev")

logger.info("Configuration loaded:")
logger.info(yaml.dump(config, default_flow_style=False))

# COMMAND ----------

spark = SparkSession.builder.getOrCreate()

# Load the data
df = spark.table("bigquery_credit_analytics_catalog.credit_analytics.loan_data")

# Display basic info about the dataset
logger.info(f"Dataset size: {df.count()}")

# COMMAND ----------

data_processor = DataProcessor(df, config, spark)

# Preprocess the data
data_processor.preprocess()

logger.info(f"Data preprocessing completed.")

# COMMAND ----------

# Split the data
df_train, df_validation_val, df_validation_test, df_test = data_processor.split_data()
logger.info(f"Training set shape: {str(df_train.shape)}")
logger.info(f"Validation val set shape: {str(df_validation_val.shape)}")
logger.info(f"Validation test set shape: {str(df_validation_test.shape)}")
logger.info(f"Test set shape: {df_test.shape}")

# COMMAND ----------
# Save to catalog
logger.info("Saving data to catalog")
data_processor.save_to_catalog(df_train, df_validation_val, df_validation_test, df_test)

# Enable change data feed (only once!)
logger.info("Enable change data feed")
data_processor.enable_change_data_feed()
# COMMAND ---------- 