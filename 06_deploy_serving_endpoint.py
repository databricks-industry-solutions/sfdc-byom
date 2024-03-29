# Databricks notebook source
# MAGIC %md 
# MAGIC You may find this series of notebooks at https://github.com/databricks-industry-solutions/sfdc-byom. For more information about this solution accelerator, visit https://www.databricks.com/solutions/accelerators/sfdc-byom.

# COMMAND ----------

# MAGIC %md
# MAGIC # Deploy Serving Endpoint
# MAGIC
# MAGIC You've come a long way. You got data from Salesforce, prepared feature tables from it, and even trained and registered a model from it. Now it's time to make that model available for use in Salesforce. In this notebook, we're going to deploy our model serving endpoint.
# MAGIC
# MAGIC As with most things in Databricks, you can do this either via the UI or via the API. Here, we're going to use the API. Fortunately for us, Databricks makes this a piece of cake in both cases. Once you have a model registered in Unity Catalog, its basically as easy as pointing to that model, telling it how much concurrency you need for your users, and then clicking go.
# MAGIC
# MAGIC Databricks takes care of the rest!

# COMMAND ----------

# DBTITLE 1,Run common setup
# MAGIC %run ./common

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configure MLflow client and token
# MAGIC
# MAGIC First, we need a couple of pieces of information. To be able to refer to the model we created previously, we'll look up the version of the model by its alias, which we named as `champion`. 
# MAGIC
# MAGIC We also need a token and URL to access the REST API. Normally you'd use a service principle for these and look them up from the Databricks secrets utility, but as with many of our demo notebooks, we'll grab them from our notebook context to keep the immediate focus on the deployment process and simplify things just a little more.

# COMMAND ----------

# DBTITLE 1,Configure client and token
# Get the SDK client handles.
mlflow_client = mlflow.MlflowClient()

# Pull our latest champion version from the registry
model_version = mlflow_client.get_model_version_by_alias(model_name, "champion")
print(model_version)

# Get a token from the notebook context for testing purposes. For production you'll want
# to access a service principal token you create and store in the dbutils secrets API.
notebook_context = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
databricks_url = notebook_context.apiUrl().getOrElse(None)
databricks_token = notebook_context.apiToken().getOrElse(None)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configure serving endpoint
# MAGIC
# MAGIC The API call to create the serving endpoint needs to know the model to be served, and what we want to call this endpoint. In this case, we already looked up the model to be served, and we can derive the name of the endpoint from that information. We also need to tell it the workload size, and since its just a demo we'll turn on _scale to zero_ so we don't incur any costs when its not in use (Note: you'd typically want to turn this off for production deployments).
# MAGIC
# MAGIC We're also going to turn on inference tables to facilitate monitoring of our models request response pairs, which you can see in the `auto_capture_config` section below.

# COMMAND ----------

# DBTITLE 1,Define endpoint configuration
# Define the endpoint configuration.
served_entity_name = f"{model_name}-{model_version.version}"
config = {
    "served_entities": [
        {
            "name": served_entity_name,
            "entity_name": model_version.name,
            "entity_version": model_version.version,
            "workload_size": "Small",
            "workload_type": "CPU",
            "scale_to_zero_enabled": True
        }
    ],
    "traffic_config": {
        "routes": [
            {
                "served_model_name": served_entity_name,
                "traffic_percentage": 100
            }
        ]
    },
    "auto_capture_config": {
        "catalog_name": catalog_name,
        "schema_name": schema_name,
        "table_name_prefix": model_name,
        "enabled": True
    }
}

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create or update the serving endpoint
# MAGIC
# MAGIC The code below looks a little fancy, but that's basically just because its handling the case where the endpoint already exists in case you are running this for the second time. At the end of the day, once you have the configuration specified, creating the endpoint itself just boils down to the one-liner to `POST` to the serving-endpoints API endpoint to initiate the deployment of the configured endpoint (or `PUT` in case you're updating it). It works basically the same through the UI.
# MAGIC
# MAGIC Once you execute this line, it'll probably take 5 to 10 minutes to actually bring up your model serving endpoint, so run the next cell and then go grab a fresh cup of coffee ☕️, and hopefully by the time your back the endpoint will be ready to go!

# COMMAND ----------

# DBTITLE 1,Create or update the model serving endpoint
# The model serving API is hosted at an endpoint in your Databricks workspace.
serving_api_endpoint = f"{databricks_url}/api/2.0/serving-endpoints"
headers = {"Authorization": f"Bearer {databricks_token}"}

# Determine if we need to create a new endpoint or update an existing one.
list_endpoint_response = requests.get(serving_api_endpoint, headers=headers)
all_endpoints = list_endpoint_response.json()["endpoints"]
endpoint_names = [endpoint["name"] for endpoint in all_endpoints]
endpoint_already_exists = endpoint_name in endpoint_names

# Create or update the endpoint based ont he config.
if not endpoint_already_exists:
    print("creating new endpoint")
    create_json = { 
        "name": endpoint_name,
        "config": config
    }
    endpoint_response = requests.post(serving_api_endpoint, headers=headers, json=create_json)
    endpoint = endpoint_response.json()
else:
    print("updating existing endpoint")
    update_endpoint_uri = f"{serving_api_endpoint}/{endpoint_name}/config"
    update_json = config
    endpoint_response = requests.put(update_endpoint_uri, headers=headers, json=update_json)
    endpoint = endpoint_response.json()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Wait for the endpoint to be ready
# MAGIC
# MAGIC That's it really.
# MAGIC
# MAGIC The rest of the code here basically just polls the API to let us know when its ready (or, when necessary, to get some information to help with troubleshooting). You can either watch it here, or over in the endpoints UI for the endpoint we just created.

# COMMAND ----------

# DBTITLE 1,Poll periodically to check if the endpoint is ready
if "error_code" in endpoint:
    print(endpoint)
else:
    print("waiting for endpoint to be ready...")

    # Wait for the endpoint to be ready.
    endpoint_ready = False
    endpoint_status_check_interval = 10

    while True:
        endpoint_response = requests.get(f"{serving_api_endpoint}/{endpoint_name}", headers=headers)
        if "error_code" in endpoint_response:
            print(endpoint_response)
            break
        state = endpoint["state"]
        if state["ready"] == "READY" and state["config_update"] == "NOT_UPDATING":
            print("endpoint ready")
            break
        # TODO: check for better ways to identify failed state or other conditions to watch out for
        elif "FAILED" in state["ready"] or "FAILED" in state["config_update"]:
            print("deployment failed - please check the logs")
            break
        else:
            endpoint = endpoint_response.json()
            time.sleep(endpoint_status_check_interval)

print(endpoint)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Conclusion
# MAGIC
# MAGIC Congratulations! 🎉 
# MAGIC
# MAGIC Now your endpoint is ready to go! You could stop here if you want and continue on with the instructions we mentioned in the README for this repo to configure the endpoint integration on the Salesforce Data Cloud side.
# MAGIC
# MAGIC However, we've provided two additional notebooks to check out as well if your interested.
# MAGIC
# MAGIC - **[Test Model Inference]($./07_test_model_inference):** This notebook basically just lets you hit the model serving endpoint you just created via the REST API directly so you can test it out and experiment with it. This is particularly useful if for some reason you run into trouble when you try to set things up on the Salesforce side. So if you want to test it out before hand, you can go to that one next to try things out.
# MAGIC
# MAGIC - **[Monitoring]($./08_monitoring):** This notebook sets up a Lakehouse Monitoring pipeline based on the inference table we creater earlier. It also simulates getting labels down stream and shows you how to join those in, and then creates an inference table monitor based on that. It's a little more meaningful after you have some inferences to look at in the table, so it might be worthwhile to either test things out using the above notebook first and setting things up in Salesforce before you come back around to this one.
# MAGIC
# MAGIC Whichever way you go, thanks for making it this far! Great job! 🥳
