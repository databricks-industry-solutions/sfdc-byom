# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC
# MAGIC # Inference Table Monitoring
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/llm-eval-online-0.png?raw=true" style="float: right" width="900px">
# MAGIC
# MAGIC #### About this notebook
# MAGIC This starter notebook is intended to be used with **Databricks Model Serving** endpoints which have the *Inference Table* feature enabled. To set up a generation endpoint, refer to the guide on model serving endpoints ([AWS](https://docs.databricks.com/en/machine-learning/model-serving/score-model-serving-endpoints.html)|[Azure](https://learn.microsoft.com/en-us/azure/databricks/machine-learning/model-serving/score-model-serving-endpoints)).</br>
# MAGIC This notebook has three high-level purposes:
# MAGIC
# MAGIC 1. Unpack the logged requests and responses by converting your model raw JSON payloads as string.
# MAGIC 2. Compute text evaluation metrics over the extracted input/output.
# MAGIC 3. Setup Databricks Lakehouse Monitoring on the resulting table to produce data and model quality/drift metrics.
# MAGIC
# MAGIC #### How to run the notebook
# MAGIC The notebook is set up to be run step-by-step. Here are the main configuration to set:
# MAGIC * Define your model serving endpoint name (mandatory)
# MAGIC * Ensure the unpacking function works with your model input/output schema
# MAGIC * Define the checkpoint location (prefer using a Volume within your schema)
# MAGIC For best results, run this notebook on any cluster running **Machine Learning Runtime 12.2LTS or higher**.
# MAGIC
# MAGIC #### Scheduling
# MAGIC Feel free to run this notebook manually to test out the parameters; when you're ready to run it in production, you can schedule it as a recurring job.</br>
# MAGIC Note that in order to keep this notebook running smoothly and efficiently, we recommend running it at least **once a week** to keep output tables fresh and up to date.

# COMMAND ----------

# MAGIC %pip install "https://ml-team-public-read.s3.amazonaws.com/wheels/data-monitoring/a4050ef7-b183-47a1-a145-e614628e3146/databricks_lakehouse_monitoring-0.4.6-py3-none-any.whl"
# MAGIC
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./common

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## Exploring the Model Serving Inference table content
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/rag-inference-table.png?raw=true" style="float: right" width="600px">
# MAGIC
# MAGIC Let's start by analyzing what's inside our inference table.
# MAGIC
# MAGIC The inference table name can be fetched from the model serving endpoint configuration. 
# MAGIC
# MAGIC We'll first get the table name and simply run a query to view its content.

# COMMAND ----------

import requests
from typing import Dict


def get_endpoint_status(endpoint_name: str) -> Dict:
    """Fetch the PAT token to send in the API request."""
    workspace_url = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
    token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)

    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(f"{workspace_url}/api/2.0/serving-endpoints/{endpoint_name}", json={"name": endpoint_name}, headers=headers).json()

    # Verify that Inference Tables is enabled.
    if "auto_capture_config" not in response.get("config", {}) or not response["config"]["auto_capture_config"]["enabled"]:
        raise Exception(f"Inference Tables is not enabled for endpoint {endpoint_name}. \n"
                        f"Received response: {response} from endpoint.\n"
                        "Please create an endpoint with Inference Tables enabled before running this notebook.")

    return response

response = get_endpoint_status(endpoint_name=endpoint_name)

auto_capture_config = response["config"]["auto_capture_config"]
catalog = auto_capture_config["catalog_name"]
schema = auto_capture_config["schema_name"]
# These values should not be changed - if they are, the monitor will not be accessible from the endpoint page.
payload_table_name = auto_capture_config["state"]["payload_table"]["name"]
payload_table_name = f"`{catalog}`.`{schema}`.`{payload_table_name}`"
print(f"Endpoint {endpoint_name} configured to log payload in table {payload_table_name}")

processed_table_name = f"{auto_capture_config['table_name_prefix']}_processed"
processed_table_name = f"`{catalog}`.`{schema}`.`{processed_table_name}`"
print(f"Processed requests with text evaluation metrics will be saved to: {processed_table_name}")

payloads = spark.table(payload_table_name).limit(10)
display(payloads)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## Unpacking the inference table requests and responses
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/llm-eval-online-1.png?raw=true" style="float: right" width="900px">
# MAGIC
# MAGIC ### Unpacking the table
# MAGIC
# MAGIC The request and response columns contains your model input and output as a `string`.
# MAGIC
# MAGIC Note that the format depends of your model definition and can be custom. Inputs are usually represented as JSON with TF format, and the output depends of your model definition.
# MAGIC
# MAGIC Because our model is designed to potentially batch multiple entries, we need to unpack the value from the request and response.
# MAGIC
# MAGIC We will use Spark JSON Path annotation to directly access the query and response as string, concatenate the input/output together with an `array_zip` and ultimately `explode` the content to have 1 input/output per line (unpacking the batches)
# MAGIC
# MAGIC **Make sure you change the following selectors based on your model definition**
# MAGIC
# MAGIC *Note: This will be made easier within the product directly, we provide this notebook to simplify this task for now.*

# COMMAND ----------

# The format of the input payloads, following the TF "inputs" serving format with a "query" field.
# Single query input format: {"inputs": [{"query": "User question?"}]}
INPUT_REQUEST_JSON_PATH = "inputs[*].query"
# Matches the schema returned by the JSON selector (inputs[*].query is an array of string)
INPUT_JSON_PATH_TYPE = "array<string>"

# Answer format: {"predictions": ["answer"]}
OUTPUT_REQUEST_JSON_PATH = "predictions"
# Matches the schema returned by the JSON selector (predictions is an array of string)
OUPUT_JSON_PATH_TYPE = "array<string>"

# COMMAND ----------

# MAGIC %md
# MAGIC Since we want to measure the model performance over time, we also need an ingestion pipeline to ingest the actual purchases once they're known in the system, wherever those are available. For demo purposes, we'll suppose the data for purchases is again coming from Salesforce and simply ingest it inline within this notebook, but for a production setting you'd want to set this up alongside the rest of your data pipeline (e.g., via a DLT pipeline).

# COMMAND ----------

packed_requests = spark.table(f"{catalog_name}.{schema_name}.recommender_payload")
display(packed_requests)

# COMMAND ----------

input_request_json_path = ""

def unpack_requests(packed_requests: DataFrame) -> DataFrame:

    input_record_type = (
        spark.table("product_interest_silver")
        .drop("product_purchased").schema)

    request_schema = T.StructType([
        T.StructField("dataframe_records", T.ArrayType(input_record_type))])
    
    response_schema = T.StructType([
        T.StructField("predictions", T.ArrayType(T.StringType()))])

    df = (
        packed_requests
        .filter(F.col("status_code") == "200")
        .withColumn("request_unpacked", F.from_json("request", request_schema))
        .withColumn("response_unpacked", F.from_json("response", response_schema))
        .withColumn("request_response_unpacked", F.arrays_zip(
            "request_unpacked.dataframe_records", 
            "response_unpacked.predictions"))
        .withColumn("exploded", F.explode(F.expr("request_response_unpacked")))
        .withColumn("model_id", F.concat(
            F.col("request_metadata.model_name").alias("model_name"),
            F.lit("/"),
            F.col("request_metadata.model_version").alias("model_version")))
        .select(
            "databricks_request_id",
            (F.col("timestamp_ms") / 1000).cast("timestamp").alias("timestamp"),
            "exploded.dataframe_records.*",
            F.col("exploded.predictions").alias("prediction"),
            "model_id"))

    return df

unpacked_requests = unpack_requests(packed_requests)

# COMMAND ----------

# DBTITLE 1,Ingest the ground truth labels
def get_sfdc_actual_purchases():
    """Retrieve our actual purchases table from SFDC."""

    conn = SalesforceCDPConnection(
            sfdc_login_url, 
            sfdc_username, 
            sfdc_password,  
            sfdc_client_id,
            sfdc_client_secret)

    query = """
    SELECT
      id__c,
      product_purchased__c
    FROM
      sfdc_byom_demo_validate__dll
    """

    df_pandas = conn.get_pandas_dataframe(query)
    df = spark.createDataFrame(df_pandas)
    return remove_column_suffix(df, SFDC_CUSTOM_FIELD_SUFFIX)


# For demo purposes, we'll go directly to a silver layer table.
# In a real setting you'll want to stick with ingesting to bronze
# and then letting silver be a cleansed and processed layer.

# Create the table by loading from SFDC in case it doesn't exist.
if not spark.catalog.tableExists("actual_purchases_silver"):
    get_sfdc_actual_purchases().write.saveAsTable("actual_purchases_silver")

# Load the actual purchases table from Delta Lake
df_actual_purchases = spark.table("actual_purchases_silver")
display(df_actual_purchases)

# COMMAND ----------

product_purchases_join = (
    unpacked_requests
    .join(df_actual_purchases, "id", "left"))

display(product_purchases_join)

# COMMAND ----------

import delta
from delta import DeltaTable

processed_table_name = "recommender_payload_with_actuals"

if spark.catalog.tableExists(processed_table_name):
    payload_actuals_table = DeltaTable.forName(spark, processed_table_name)
    payload_actuals_merge = (
        payload_actuals_table.alias("target")
        .merge(product_purchases_join.alias("source"), "source.id = target.id")
        .whenNotMatchedInsertAll()
        .whenMatchedUpdate(set={"target.product_purchased": "source.product_purchased"}))
    payload_actuals_merge.execute()
else:
    product_purchases_join.write.saveAsTable(processed_table_name)
    spark.sql(f"""
        ALTER TABLE {processed_table_name} 
        SET TBLPROPERTIES (delta.enableChangeDataFeed = true)""")

display(spark.table(processed_table_name))

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC ### Monitor the inference table
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/llm-eval-online-2.png?raw=true" style="float: right" width="900px">
# MAGIC
# MAGIC In this step, we create a monitor on our inference table by using the `create_monitor` API. If the monitor already exists, we pass the same parameters to `update_monitor`. In steady state, this should result in no change to the monitor.
# MAGIC
# MAGIC Afterwards, we queue a metric refresh so that the monitor analyzes the latest processed requests.
# MAGIC
# MAGIC See the Lakehouse Monitoring documentation ([AWS](https://docs.databricks.com/lakehouse-monitoring/index.html) | [Azure](https://learn.microsoft.com/azure/databricks/lakehouse-monitoring/index)) for more details on the parameters and the expected usage.

# COMMAND ----------

import databricks.lakehouse_monitoring as lm

# Optional parameters to control monitoring analysis.
# For help, use the command help(lm.create_monitor).
GRANULARITIES = ["1 day"]       # Window sizes to analyze data over
SLICING_EXPRS = None            # Expressions to slice data with
CUSTOM_METRICS = None           # A list of custom metrics to compute
BASELINE_TABLE = None           # Baseline table name, if any, for computing baseline drift

monitor_params = {
    "profile_type": lm.InferenceLog(
        timestamp_col="timestamp",
        granularities=GRANULARITIES,
        problem_type="classification",
        prediction_col="prediction",
        label_col="product_purchased",
        model_id_col="model_id"
    ),
    "output_schema_name": f"{catalog_name}.{schema_name}",
    "schedule": None,  # We will refresh the metrics on-demand in this notebook
    "baseline_table_name": BASELINE_TABLE,
    "slicing_exprs": SLICING_EXPRS,
    "custom_metrics": CUSTOM_METRICS
}

try:
    info = lm.create_monitor(table_name=processed_table_name, **monitor_params)
    print(info)
except Exception as e:
    # Ensure the exception was expected
    assert "RESOURCE_ALREADY_EXISTS" in str(e), f"Unexpected error: {e}"

    # Update the monitor if any parameters of this notebook have changed.
    lm.update_monitor(table_name=processed_table_name, updated_params=monitor_params)
    # Refresh metrics calculated on the requests table.
    refresh_info = lm.run_refresh(table_name=processed_table_name)
    print(refresh_info)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Our table is now monitored
# MAGIC
# MAGIC Databricks Lakehouse Monitoring automatically builds dashboard to track your metrics and their evolution over time.
# MAGIC
# MAGIC You can leverage your metric table to track your model behavior over time, and setup alerts to detect potential changes in accuracy or drift, and even trigger retrainings.
