# Databricks notebook source
# MAGIC %pip install credit_risk-1.0.1-py3-none-any.whl

# COMMAND ----------
# MAGIC %restart_python

# COMMAND ----------
import time
import os
import requests
from pyspark.dbutils import DBUtils
from pyspark.sql import SparkSession
from mlflow import mlflow

from credit_risk.config import ProjectConfig
from credit_risk.serving.model_serving import ModelServing
from credit_risk.utils import is_databricks

# COMMAND ----------
# spark session
spark = SparkSession.builder.getOrCreate()

if is_databricks():
    from pyspark.dbutils import DBUtils
    dbutils = DBUtils(spark)
    os.environ["DBR_TOKEN"] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
    os.environ["DBR_HOST"] = spark.conf.get("spark.databricks.workspaceUrl")
else:
    from dotenv import load_dotenv
    load_dotenv()
    # DBR_TOKEN and DBR_HOST should be set in your .env file
    assert os.environ.get("DBR_TOKEN"), "DBR_TOKEN must be set in your environment or .env file."
    assert os.environ.get("DBR_HOST"), "DBR_HOST must be set in your environment or .env file."
    profile = os.environ["PROFILE"]
    mlflow.set_tracking_uri(f"databricks://{profile}")
    mlflow.set_registry_uri(f"databricks-uc://{profile}")

# Load project config
config = ProjectConfig.from_yaml(config_path="../project_config_credit.yml", env="dev")
catalog_name = config.catalog_name
schema_name = config.schema_name

# COMMAND ----------
# Initialize model serving
model_serving = ModelServing(
    model_name=f"{catalog_name}.{schema_name}.credit_risk_model_custom", endpoint_name="credit-risk-model-serving"
)

# COMMAND ----------
# Deploy the model serving endpoint
model_serving.deploy_or_update_serving_endpoint()


# COMMAND ----------
# Create a sample request body
required_columns = config.final_features

test_set = spark.table(f"{config.catalog_name}.{config.schema_name}.test_set").toPandas()

sampled_records = test_set[required_columns].sample(n=2000, replace=True).to_dict(orient="records")
dataframe_records = [[record] for record in sampled_records]

# COMMAND ----------
# Call the endpoint with one sample record

def call_endpoint(record):
    """
    Calls the model serving endpoint with a given input record.
    """
    # Ensure the host URL is complete with domain suffix (.com, etc.)
    host = os.environ['DBR_HOST']
    # If the host doesn't contain a dot, it's likely missing the domain suffix
    if '.' not in host:
        print(f"Warning: DBR_HOST '{host}' may be incomplete. Adding '.com' domain suffix.")
        host = f"{host}.com"
        
    serving_endpoint = f"https://{host}/serving-endpoints/credit-risk-model-serving/invocations"
    
    print(f"Calling endpoint: {serving_endpoint}")
    
    response = requests.post(
        serving_endpoint,
        headers={"Authorization": f"Bearer {os.environ['DBR_TOKEN']}"},
        json={"dataframe_records": record},
    )
    return response.status_code, response.text


status_code, response_text = call_endpoint(dataframe_records[0])
print(f"Response Status: {status_code}")
print(f"Response Text: {response_text}")

# COMMAND ----------
# Load test
for i in range(len(dataframe_records)):
    status_code, response_text = call_endpoint(dataframe_records[i])
    print(f"Response Status: {status_code}")
    print(f"Response Text: {response_text}")
    time.sleep(0.2) 
# COMMAND ----------
