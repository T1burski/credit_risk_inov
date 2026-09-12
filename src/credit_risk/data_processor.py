"""Data preprocessing module for Credit risk."""

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp, to_utc_timestamp
from sklearn.model_selection import train_test_split

from credit_risk.config import ProjectConfig
from credit_risk.feature_engineering import feature_engineering


class DataProcessor:
    """A class for preprocessing and managing Credit risk data operations.

    This class handles data preprocessing, splitting, and saving to Databricks tables.
    """

    def __init__(self, spark_df: pd.DataFrame, config: ProjectConfig, spark: SparkSession) -> None:
        self.df = spark_df  # Store the DataFrame as self.df
        self.config = config  # Store the configuration
        self.spark = spark

    def preprocess(self) -> None:
        """Preprocess the Credit risk data stored in self.df.

        This method handles missing values, converts data types, and performs feature engineering.
        """
        original_features = self.config.original_features

        self.df = self.df.select(*original_features)

        self.df = feature_engineering(self.df)

    def split_data(self, test_size: float = 0.4, random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split the DataFrame (self.df) into training and test sets.

        :param test_size: The proportion of the dataset to include in the test split.
        :param random_state: Controls the shuffling applied to the data before applying the split.
        :return: A tuple containing the training and test DataFrames.
        """
        reference_categories = self.config.reference_categories
        final_features = self.config.final_features
        target = self.config.target

        # Make sure the column is datetime
        self.df["issue_d_date"] = pd.to_datetime(self.df["issue_d_date"])

        # Test set (year == 2015)
        df_test = self.df[self.df["issue_d_date"].dt.year == 2015]

        # Train + validation (year < 2015)
        df_train_validation = self.df[self.df["issue_d_date"].dt.year < 2015]

        # Train set (before 2014-09-01)
        df_train = df_train_validation[df_train_validation["issue_d_date"] < "2014-09-01"]

        # Validation set (on/after 2014-09-01)
        df_validation = df_train_validation[df_train_validation["issue_d_date"] >= "2014-09-01"]

        X_val_val, X_val_test, y_val_val, y_val_test = train_test_split(
            df_validation[final_features].astype(str),
            df_validation[target],
            test_size=test_size,
            random_state=random_state,
            stratify=df_validation[target],
        )

        df_validation_val = pd.concat([X_val_val, y_val_val], axis=1)
        df_validation_test = pd.concat([X_val_test, y_val_test], axis=1)

        df_test = df_test[final_features + [target]]
        df_train = df_train[final_features + [target]]

        return df_train, df_validation_val, df_validation_test, df_test

    def save_to_catalog(
        self,
        df_train: pd.DataFrame,
        df_validation_val: pd.DataFrame,
        df_validation_test: pd.DataFrame,
        df_test: pd.DataFrame,
    ) -> None:
        """Save the train and test sets into Databricks tables.

        :param train_set: The training DataFrame to be saved.
        :param test_set: The test DataFrame to be saved.
        """
        train_set_with_timestamp = self.spark.createDataFrame(df_train).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
        )

        val_val_set_with_timestamp = self.spark.createDataFrame(df_validation_val).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
        )

        val_test_set_with_timestamp = self.spark.createDataFrame(df_validation_test).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
        )

        test_set_with_timestamp = self.spark.createDataFrame(df_test).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
        )

        train_set_with_timestamp.write.mode("overwrite").saveAsTable(
            f"{self.config.catalog_name}.{self.config.schema_name}.train_set"
        )

        val_val_set_with_timestamp.write.mode("overwrite").saveAsTable(
            f"{self.config.catalog_name}.{self.config.schema_name}.validation_val_set"
        )

        val_test_set_with_timestamp.write.mode("overwrite").saveAsTable(
            f"{self.config.catalog_name}.{self.config.schema_name}.validation_test_set"
        )

        test_set_with_timestamp.write.mode("overwrite").saveAsTable(
            f"{self.config.catalog_name}.{self.config.schema_name}.test_set"
        )

    def enable_change_data_feed(self) -> None:
        """Enable Change Data Feed for train and test set tables.

        This method alters the tables to enable Change Data Feed functionality.
        """
        self.spark.sql(
            f"ALTER TABLE {self.config.catalog_name}.{self.config.schema_name}.train_set "
            "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
        )

        self.spark.sql(
            f"ALTER TABLE {self.config.catalog_name}.{self.config.schema_name}.validation_val_set "
            "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
        )

        self.spark.sql(
            f"ALTER TABLE {self.config.catalog_name}.{self.config.schema_name}.validation_test_set "
            "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
        )

        self.spark.sql(
            f"ALTER TABLE {self.config.catalog_name}.{self.config.schema_name}.test_set "
            "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
        )
