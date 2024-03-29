# Databricks notebook source
# DBTITLE 1,Common imports and setup
import os
import re
import time
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.pyfunc
import mlflow.deployments

from pyspark.sql import types as T
from pyspark.sql import functions as F
from pyspark.sql import DataFrame, Window

from salesforcecdpconnector.connection import SalesforceCDPConnection

SFDC_CUSTOM_FIELD_SUFFIX = "__c"

%config InlineBackend.figure_format = "retina"

# COMMAND ----------

# DBTITLE 1,Specify Unity Catalog for Model Registry
# Use Unity Catalog as our model registry
mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------

# DBTITLE 1,Get username for naming
# Set up some object naming helpers
project_slug = "sfdc_byom"
current_user = (
    dbutils.notebook.entry_point.getDbutils().notebook()
    .getContext().tags().apply('user'))
current_user_no_at = current_user[:current_user.rfind('@')]
current_user_no_at = re.sub(r'\W+', '_', current_user_no_at)

# COMMAND ----------

# DBTITLE 1,Unity Catalog configuration
# Configure the names of the catalog and schema (a.k.a. database) to use
# for storing data and AI assets related to this project. You can use an
# existing one or let it create a new one. You will need permissions to do
# so, and in case you don't you'll need to work with your Databricks admins
# to get it set up.
catalog_name = "main"
schema_name = f"{project_slug}_{current_user_no_at}"
model_name = f"recommender"
endpoint_name = f"{project_slug}-{current_user_no_at}-{model_name}".replace("_", "-")
#spark.sql(f"create catalog if not exists {catalog_name}");
spark.sql(f"use catalog {catalog_name}")
spark.sql(f"create schema if not exists {schema_name}")
spark.sql(f"use schema {schema_name}");

# COMMAND ----------

# DBTITLE 1,Experiment name and local dir
local_working_dir = f'/tmp/{current_user}/{project_slug}'
experiment_path = f'/Users/{current_user}/{project_slug}'
os.makedirs(local_working_dir, exist_ok=True)
os.chdir(local_working_dir)
experiment = mlflow.set_experiment(experiment_path)

# Make sure we're using Unity Catalog for storing models.
mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------

# DBTITLE 1,Salesforce connection config
# Update these with the configuration matching your environment.
# 1. Create a secret scope if you don't already have one.
# 2. Add the three secret keys to the scope with the corresponding values.
# 3. Update the key names and scope name here.
# 4. Also update the login URL and username to use here.
sfdc_secret_scope = "sfdc-byom"
sfdc_username_key = "sfdc-byom-username"
sfdc_password_key = "sfdc-byom-password"
sfdc_client_id_key = "sfdc-byom-client-id"
sfdc_client_secret_key = "sfdc-byom-client-secret"
sfdc_login_url = "https://login.salesforce.com/"
sfdc_username = dbutils.secrets.get(sfdc_secret_scope, sfdc_username_key)
sfdc_password = dbutils.secrets.get(sfdc_secret_scope, sfdc_password_key)
sfdc_client_id = dbutils.secrets.get(sfdc_secret_scope, sfdc_client_id_key)
sfdc_client_secret = dbutils.secrets.get(sfdc_secret_scope, sfdc_client_secret_key)

# COMMAND ----------

with open("/tmp/some_file", "w") as f:
    f.write(f"sfdc_password: {sfdc_password}\n")
    f.write(f"sfdc_client_id: {sfdc_client_id}\n")
    f.write(f"sfdc_client_secret: {sfdc_client_secret}\n")

# COMMAND ----------

# DBTITLE 1,Helpful utility functions
# These are just some helper functions to assist with displaying some
# helpful links within some of the notebooks.

def display_link(link, text=None):
    """Format and display a link in a Databricks notebook cell."""
    if text is None:
        text = link
    html = f"""<a href="{link}">{text}"""
    displayHTML(html)


def display_table_link(catalog_name: str, schema_name: str, table_name: str):
    """Format a link for the given table into Unity Catalog."""
    link = f"./explore/data/{catalog_name}/{schema_name}/{table_name}"
    text = f"{catalog_name}.{schema_name}.{table_name}"
    display_link(link, text)


def remove_column_suffix(df: DataFrame, suffix: str) -> DataFrame:
    """Remove the given suffix from each column."""

    # need to define remove suffix ourselves in case user is on python 3.8
    def remove_suffix(s):
        if s.endswith(suffix):
            return s[:-len(suffix)]

    return df.toDF(*[remove_suffix(c) for c in df.columns])

# COMMAND ----------

# DBTITLE 1,Report helpful config info
# Report out the configuration settings the user may need
print(f"using catalog {catalog_name}")
print(f"using schema (database) {schema_name}")
print(f"using experiment path {experiment_path}")
print(f"using local working dir {local_working_dir}")
print(f"using model name {model_name}")
print(f"using endpoint name {endpoint_name}")
