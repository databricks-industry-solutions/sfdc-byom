# Databricks notebook source
# MAGIC %md 
# MAGIC You may find this series of notebooks at https://github.com/databricks-industry-solutions/sfdc-byom. For more information about this solution accelerator, visit https://www.databricks.com/solutions/accelerators/sfdc-byom.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Feature Engineering
# MAGIC
# MAGIC Welcome to the Feature Engineering Notebook, a crucial part of the SalesForce Data Cloud to Databricks integration! This notebook plays a pivotal role in deriving meaningful features from raw data ingested from SalesForce, ensuring that our machine learning models are well-equipped to make accurate predictions and generate valuable insights. This step is also how we publish our features in a way that our colleagues can find and reuse them as well, using Unity Catalog as our Feature Store, in addition to providing useful capabilities for automated feature lookup and online feature serving.
# MAGIC
# MAGIC Let's get started!

# COMMAND ----------

# DBTITLE 1,Run common setup
# MAGIC %run ./common

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load the Feature Engineering in Unity Catalog API
# MAGIC The API for [Databricks Feature Engineering in Unity Catalog](https://docs.databricks.com/en/machine-learning/feature-store/uc/feature-tables-uc.html) is provided by the package `databricks.feature_engineering`. To use it, simply import the required classes from the package and then instantiate the `FeatureEngineeringClient`.

# COMMAND ----------

# DBTITLE 1,Import the feature engineering library
from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup

fe = FeatureEngineeringClient()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load the product interest silver table
# MAGIC
# MAGIC Feature tables are a great place to store all sorts of features about a particular entity. For this particular use case, the tables we retrieved from Salesforce are already more or less feature ready, with each row representing a user and each column representing a feature of the user. However, for most of your use cases this won't be so simple. You'll likely be pulling together features from silver and gold tables across your Lakehouse, both sourced from Salesforce Data Cloud as well as from other systems within and outside your organization.
# MAGIC
# MAGIC The good news is, you don't need to learn any new technologies to pull all this data together in Databricks. At the end of the day, as long as you can retrieve and process those features using a Spark DataFrame, you can create and maintain a feature table in Unity Catalog for those features. The key requirement is that the resulting table has a primary key. To represent this process for our example here, we're simply going to load up our product interest silver table and use that to create our first feature table.
# MAGIC
# MAGIC You can use this as a baseline to know how to create and extend your own feature tables specific to your use case.

# COMMAND ----------

# DBTITLE 1,Load and view silver product interest table
df = (
    spark.table("product_interest_silver")
    .drop("product_purchased"))

display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create feature table
# MAGIC
# MAGIC The main API for creating a feature table is `create_table`.
# MAGIC
# MAGIC Let's take a look at the API help for this function.

# COMMAND ----------

help(fe.create_table)

# COMMAND ----------

# MAGIC %md
# MAGIC As you can see, you can either call this function with a schema or a dataframe.
# MAGIC
# MAGIC If you call it with a dataframe, it will take the schema of the provided dataframe and then write the table with all the rows of that dataframe as the initial content of the table.
# MAGIC
# MAGIC Alternatively, you can just provide a schema, and then write it later. Here, we'll demonstrate this method, taking the schema from the product interest silver table we loaded earlier. Note that this latter method is idempotent: if you call it again, it will just provide a helpful warning that the table already exists, but won't otherwise fail. You can check for the table existence explitly as well.

# COMMAND ----------

# DBTITLE 1,Create the empty feature table based on the schema
base_table_name = "product_interest_features"
full_table_name = f"{catalog_name}.{schema_name}.{base_table_name}"

fe.create_table(
    name=full_table_name,
    primary_keys=["id"],
    schema=df.schema)

display_table_link(catalog_name, schema_name, base_table_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write the table content
# MAGIC
# MAGIC Now that we have the feature table created, we can write our feature data into the table. By default, this will use a merge statement, inserting new rows and updating existing ones based on the primary key.

# COMMAND ----------

# DBTITLE 1,Merge our feature records into the table
fe.write_table(name=full_table_name, df=df)

# COMMAND ----------

# MAGIC %md
# MAGIC Now that we've written the table, let's demonstrate reading it back.

# COMMAND ----------

# DBTITLE 1,Read back the records and display them
display(fe.read_table(name=full_table_name))
display_table_link(catalog_name, schema_name, base_table_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Conclusion
# MAGIC
# MAGIC And that's it for our Feature Engineering Notebook! You have successfully transformed raw SalesForce data into a comprehensive set of enriched features, paving the way for the development of powerful machine learning models. Use this simple example as a baseline for combining sources from across your enterprise to feed your SalesForce Data Cloud use cases with powerful machine learning models. Driving your machine learning models with well defined feature tables helps you achieve the following:
# MAGIC
# MAGIC - **Aids in reuse across models and teams:** By using Unity Catalog and the Feature Engineering library, you enable your feature tables to be defined once and discovered for reuse in additional use cases so that teams don't have to reinvent the wheel every time they need the same feature. This also helps maintain a consistent and correct definition of the feature logic.
# MAGIC - **Avoid feature / serving skew:** Since the same feature logic and tables can be used for both training the model and serving it, whether its served via batch, streaming, or as in the case with SalesForce a real-time model serving endpoint, you can rest assured that you won't have to reimplement the logic again and potentially introduce costly errors. Define the features correctly, once, and then use those same features for training, evaluation and inference pipelines.
# MAGIC
# MAGIC Now that we have the features ready to go, let's continue on to the next notebook: [Build and Train Model]($./05_build_and_train_model).
