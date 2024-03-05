# Databricks notebook source
# MAGIC %md
# MAGIC # Test Model Inference
# MAGIC
# MAGIC This supplementary notebook provides a simple way to test the model serving endpoint we created in the prior notebooks. It assumes you completed all the previous notebooks and have the silver table configured, the model registered, and the model serving endpoint deployed, so please make sure you've completed all that before trying this one!

# COMMAND ----------

# DBTITLE 1,Run common setup
# MAGIC %run ./common

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define URI and token
# MAGIC
# MAGIC As before, to use the REST API we need a URL and token, so let's get those from our notebook context for ease of use.
# MAGIC
# MAGIC Note: for production deployments, please use service principals with the secrets API for this.

# COMMAND ----------

# DBTITLE 1,Configure token
# Get a token from the notebook context for testing purposes. For production you'll want
# to access a service principal token you create and store in the dbutils secrets API.
notebook_context = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
databricks_url = notebook_context.apiUrl().getOrElse(None)
databricks_token = notebook_context.apiToken().getOrElse(None)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Grab some sample data
# MAGIC
# MAGIC We need some data to send to the endpoint.
# MAGIC
# MAGIC Fortunately, since we already created a silver table from the Salesforce Data earlier, we can just pull a few records from that.

# COMMAND ----------

# DBTITLE 1,Grab some sample rows from the silver table
df = spark.table("product_interest_silver").limit(10).toPandas()

# Separate features and target variable
X = df.drop("product_purchased", axis=1)
y = df["product_purchased"]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prepare the payload
# MAGIC
# MAGIC To feed the data to the REST API, we need to convert it to JSON. The easiest way to do that is via the Pandas `to_json` method using one of the `orient` options. I tend to use the `records` format as that is particularly readable and easy to inspect manually, but others can work as well. The `records` format happens to also line up with what's expected in the [monitoring notebook]($./08_monitoring) and what we've mentioned in the instructions for setting up the integration in the instructions for Salesforce Data Cloud.

# COMMAND ----------

# DBTITLE 1,Prepare the payload
import json
payload = {"dataframe_records": json.loads(X[:3].to_json(orient="records"))}
print(json.dumps(payload, indent=4))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prepare the headers and URI
# MAGIC
# MAGIC To make the call, you need to format the URI with the name of the endpoint to call, and also put the token in an authorization header. Conveniently, this is similar to the information you'll put into Salesforce when you configure the endpoint in Salesforce (the main difference being you'll want to use a different token than the one we got from the Notebook context for that one).

# COMMAND ----------

# DBTITLE 1,Prepare the headers and URI
# The model serving API is hosted at an endpoint in your Databricks workspace.
invocations_uri = f"{databricks_url}/serving-endpoints/{endpoint_name}/invocations"
headers = {"Authorization": f"Bearer {databricks_token}"}

# COMMAND ----------

# MAGIC %md
# MAGIC ## Post to the endpoint
# MAGIC
# MAGIC With that information defined, you now just make the `POST` call to that endpoint, passing along the authorization header and the payload.

# COMMAND ----------

# DBTITLE 1,Invoke the model serving endpoint
# Call the endpoint and print the response!
response = requests.post(invocations_uri, headers=headers, json=payload)
print(response.json())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Conclusion
# MAGIC
# MAGIC That's it. You now know how to successfully hit the model serving endpoint via the REST API.
# MAGIC
# MAGIC You are definitely ready by this point to go set up the integration in Salesforce Data Cloud. After you've done that and have some data ready to review from the inference tables, don't forgot to come back and have a look at the [monitoring notebook]($./08_monitoring) as well!
