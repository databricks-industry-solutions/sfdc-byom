# Databricks notebook source
# MAGIC %md 
# MAGIC You may find this series of notebooks at https://github.com/databricks-industry-solutions/sfdc-byom. For more information about this solution accelerator, visit https://www.databricks.com/solutions/accelerators/sfdc-byom.

# COMMAND ----------

# MAGIC %md
# MAGIC # Data Ingestion
# MAGIC
# MAGIC In this notebook, we'll demonstrate how to load data from Salesforce Data Cloud into Databricks. The primary objective is to give you the tools necessary to construct a data pipeline that pulls in the required data from Salesforce so you can combine it with the rest of the data in your Databricks Lakehouse to produce effective machine learning models. The method outlined here focuses on the Salesforce CDP Connection Python Library.
# MAGIC
# MAGIC ## What You Will Achieve
# MAGIC
# MAGIC By following this notebook, you will learn how to:
# MAGIC
# MAGIC - **Connect and Extract Data**: Establish a connection to Salesforce Data Cloud, enabling you to extract product interest data.
# MAGIC - **Transform Data**: Employ transformation techniques to transition the data from its raw form in the bronze layer to a refined, cleansed state in the silver layer.
# MAGIC - **Load Data into Databricks**: Load your transformed data into Databricks, making it available for analysis and discovery.
# MAGIC
# MAGIC ## Why This Matters
# MAGIC
# MAGIC In today's data-driven world, the ability to efficiently process and analyze data is paramount. This notebook helps you:
# MAGIC
# MAGIC - **Enhance Data Quality**: Through the transformation process, you will improve the quality of your data, making it more reliable for decision-making.
# MAGIC - **Accelerate Time-to-Insight**: By streamlining the data ingestion process, you reduce the time from data collection to actionable insights, enabling faster decision-making.
# MAGIC - **Simplify Data Management**: The use of the Salesforce CDP Connection Python Library simplifies the complexity of data management, making it accessible to users with varying levels of technical expertise.
# MAGIC
# MAGIC ## Separation of Concerns
# MAGIC
# MAGIC Having the data ingestion as a separate notebook from the rest of the model training process provides a couple of key advantages over just loading it directly in your model training notebook:
# MAGIC
# MAGIC - **Speed up Model Experimentation**: If you reload the dataframe every time you start the model training notebook, during experimentation, this can slow things down considerably. Preloading the table as a Delta table in Databricks where it is optimized for both BI and AI workloads can speed up your experimentation greatly.
# MAGIC - **Scheduled Independently**: You may want to have new data or a fresh snapshot on a different schedule than other parts of the workload. Having it as a separate notebook, and thus configurable as a separate task in Databricks Workflows, provides flexibility in this scheduling.
# MAGIC - **Team Collaboration**: It may be a different engineer or SME who is responsible for loading the data from Salesforce, optimizing the data loading process and determining the right tables to query and the right joins to make. Separating concerns in this way makes it easier for the right people to focus on the right parts of the development process.
# MAGIC

# COMMAND ----------

# DBTITLE 1,Common setup
# MAGIC %run ./common

# COMMAND ----------

# MAGIC %md
# MAGIC ## Set up Salesforce CDP Connection
# MAGIC
# MAGIC The first step towards data ingestion involves establishing a connection to the Salesforce Customer Data Platform (CDP). This connection is the bridge that allows us to access the product interest data stored within Salesforce Data Cloud. To achieve this, we leverage the `SalesforceCDPConnection` class, provided by the [Salesforce CDP Connection Python Library](https://github.com/forcedotcom/salesforce-cdp-connector). Below, we detail the process of initializing this connection, ensuring a secure and efficient link to your Salesforce data.
# MAGIC
# MAGIC In this code snippet, we instantiate the `SalesforceCDPConnection` object with five parameters:
# MAGIC
# MAGIC - `sfdc_login_url`: The URL used for logging into Salesforce. This is your gateway to authenticate against the Salesforce platform, ensuring secure access.
# MAGIC - `sfdc_username`: Your Salesforce username. This credential identifies you to the Salesforce services and ensures that your connection is personalized and secure.
# MAGIC - `sfdc_password`: The password associated with your Salesforce account. Combined with your username, it authenticates your access to Salesforce's data.
# MAGIC - `sfdc_client_id`: The client ID provided when you register your application with Salesforce. It's part of the OAuth credentials needed to authorize your application to access Salesforce data on your behalf.
# MAGIC - `sfdc_client_secret`: The client secret is another component of your OAuth credentials, working alongside the client ID to provide a secure authentication mechanism.
# MAGIC
# MAGIC These variables are already initialized in the `common` notebook, where they are configured there using [Databricks secrets management](https://docs.databricks.com/en/security/secrets/index.html).

# COMMAND ----------

# DBTITLE 1,Connect to Salesforce data cloud
conn = SalesforceCDPConnection(
        sfdc_login_url, 
        sfdc_username, 
        sfdc_password,  
        sfdc_client_id,
        sfdc_client_secret)

print(conn)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Retrieving Data with a Salesforce Query
# MAGIC
# MAGIC Following the successful establishment of our connection to Salesforce CDP, we proceed to extract product interest data using a specific SQL query. The query is structured to pull a targeted set of fields from the `sfdc_byom_demo_train__dll`, focusing on crucial metrics such as product purchases, customer engagement scores, and interaction metrics, limited to the first 10,000 records for manageability and performance optimization.
# MAGIC
# MAGIC By executing this command, we fetch the data directly into a pandas DataFrame, leveraging the `get_pandas_dataframe` method of our Salesforce connection object.

# COMMAND ----------

# DBTITLE 1,Query product interest data
query = """
SELECT
  id__c,
  product_purchased__c,
  club_member__c,
  campaign__c,
  state__c,
  month__c,
  case_count__c,
  case_type_return__c,
  case_type_shipment_damaged__c,
  pages_visited__c,
  engagement_score__c,
  tenure__c,
  clicks__c
FROM
  sfdc_byom_demo_train__dll
LIMIT
  10000
"""

df_pandas = conn.get_pandas_dataframe(query)
display(df_pandas)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Transforming Data for Analysis
# MAGIC
# MAGIC Once the product interest data is retrieved into a pandas DataFrame, the next step is to convert this DataFrame into a Spark DataFrame and refine the column names for ease of analysis. This conversion leverages the Apache Spark framework within Databricks, allowing for scalable data processing.
# MAGIC
# MAGIC This code snippet performs two key actions:
# MAGIC 1. **Conversion to Spark DataFrame**: The `spark.createDataFrame(df_pandas)` command transforms the pandas DataFrame into a Spark DataFrame, enabling the utilization of Spark's distributed data processing capabilities.
# MAGIC 2. **Column Name Refinement**: The subsequent line iterates over the column names, removing the `__c` suffix that Salesforce appends to custom fields. This simplification of column names facilitates easier access and manipulation of the data in downstream processes.
# MAGIC
# MAGIC The final `display(df_spark)` command visually presents the transformed Spark DataFrame, allowing for a quick verification of the transformations applied.

# COMMAND ----------

# DBTITLE 1,Prepare Spark dataframe
# Convert to spark dataframe
df_spark = spark.createDataFrame(df_pandas)

# Remove the __c suffix from the column names
df_spark = remove_column_suffix(df_spark, SFDC_CUSTOM_FIELD_SUFFIX)

# Inspect the results
display(df_spark)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Storing Data in the Bronze Table
# MAGIC
# MAGIC After transforming the product interest data into a Spark DataFrame with cleaned column names, the next step involves persisting this data into a storage layer for further processing. This is achieved by writing the data to a bronze Delta table, which serves as the raw data layer in our lakehouse architecture.
# MAGIC
# MAGIC In this code block, we define the name of the bronze table as `product_interest_bronze`. Using the Spark DataFrame's `.write` method, we specify the write mode as `"overwrite"` to ensure that any existing data in the table with the same name is replaced. This approach helps in maintaining the most current dataset for analysis. The `.saveAsTable(bronze_table_name)` command then persists the DataFrame as a table in the data lake, using the specified table name. This approach is more for simplicities sake, as alternatives such as using a [merge statement](https://docs.databricks.com/en/delta/merge.html) or employing [Delta Live Tables](https://www.databricks.com/product/delta-live-tables) may be better suited depending on your specific use case.
# MAGIC
# MAGIC This process of saving the transformed data into a bronze table is a critical step in building a scalable and reliable data pipeline. It ensures that the raw data is stored in a structured format, ready for subsequent cleansing, enrichment, and analysis in the silver layer. This structured approach to data storage and management, known as the [medallion or multi-hop architecture](https://www.databricks.com/glossary/medallion-architecture), facilitates efficient data processing workflows and supports advanced analytics and machine learning projects.

# COMMAND ----------

# DBTITLE 1,Write bronze table
bronze_table_name = "product_interest_bronze"

(df_spark.write
    .option("mergeSchema", "true")
    .mode("overwrite")
    .saveAsTable(bronze_table_name))

# COMMAND ----------

# MAGIC %md
# MAGIC Let's take a look at the table produced. We've also provided a link so you can easily jump to the table in [Unity Catalog](https://www.databricks.com/product/unity-catalog).

# COMMAND ----------

# DBTITLE 1,Visualize bronze table
display(spark.table(bronze_table_name))
display_table_link(catalog_name, schema_name, bronze_table_name)

# COMMAND ----------

# MAGIC %md
# MAGIC The next code cell focuses on creating and cleansing the data for the `product_interest_silver` table, which is aimed at refining the dataset stored in the Spark DataFrame `df_spark`:
# MAGIC
# MAGIC 1. **Basic Cleansing**: The operation `.na.drop()` is applied to the DataFrame, which removes any rows containing null or missing values. This step is crucial for ensuring the quality and reliability of the data by eliminating incomplete records that could potentially skew analysis results.
# MAGIC
# MAGIC 2. **Displaying the Cleansed Data**: After the cleansing process, the `display(product_interest_silver)` function is used to visually present the cleansed dataset. This allows for immediate verification of the data cleaning step, ensuring that the dataset now contains only complete and valid entries, ready for more sophisticated analysis or processing.
# MAGIC
# MAGIC Your data cleansing steps are likely to be much more involved, and will be highly dependent on your use case. By loading the data from Salesforce in a raw fashion into the bronze layer, as you iterate on these cleansing steps you don't need to continually pull data back across connection to Salesforce.

# COMMAND ----------

# DBTITLE 1,Cleanse and process incoming data
# Create product interest silver

# basic cleansing
product_interest_silver = (
    df_spark
    .na.drop())

display(product_interest_silver)

# COMMAND ----------

# MAGIC %md
# MAGIC The next cell is responsible for persisting the cleansed and processed data into the silver table, which is the next step in our medallion architecture.
# MAGIC
# MAGIC - **Specify Silver Table Name**: The variable `silver_table_name` is assigned the value `"product_interest_silver"`, defining the name of the table where the cleansed data will be stored.
# MAGIC
# MAGIC - **Data Persistence Operation**: The `product_interest_silver` Spark DataFrame, which holds the cleansed data, is written to the Silver table using the `.write` method. The `.mode("overwrite")` option specifies that if the table already exists, its current contents should be replaced with the new dataset. Finally, `.saveAsTable(silver_table_name)` persists the DataFrame as a table in the data lake under the defined name.
# MAGIC
# MAGIC This process ensures that the silver table is updated with the latest version of the cleansed data, ready for advanced analytics, reporting, or further processing. The use of the "overwrite" mode ensures that the data remains current, reflecting the latest available information.

# COMMAND ----------

# DBTITLE 1,Write silver table
silver_table_name = "product_interest_silver"

(product_interest_silver.write
    .option("mergeSchema", "true")
    .mode("overwrite")
    .saveAsTable(silver_table_name))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Further Processing: Gold Layer
# MAGIC
# MAGIC While not applicable in this particular example, many ML projects need feature engineering that involves aggregating and combining data across many different sources, producing a gold layer in the medallion architecture. For instance, if you had brought in transactional level data from Salesforce or other systems into your lakehouse and wanted to aggregate counts of data or other statistics at a user level, you would perform those aggregates on the silver layer tables to produce a gold table.
# MAGIC
# MAGIC The table we're extracting here is already having features from elsewhere in Salesforce that will lend themselves well to our downstream modeling tasks, but this is definitely something to keep in mind as you tackle new use cases.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Conclusion
# MAGIC
# MAGIC Congratulations on getting the data ingested from Salesforce Data Cloud! This is often one of the most challenging steps in the process for teams that are perhaps used to Databricks but new to Salesforce Data Cloud. Through this notebook, you have successfully navigated the process of connecting to Salesforce CDP, extracting product interest data, and performed essential transformations to prepare the data for advanced analysis. By persisting the data first in the bronze layer and then refining it for the silver layer, you've laid a solid foundation for insightful analytics and data-driven decision-making.
# MAGIC
# MAGIC ### Key Takeaways
# MAGIC
# MAGIC - **Streamlined Data Ingestion**: You've seen firsthand how to efficiently extract data from Salesforce CDP using the Salesforce CDP Connection Python Library, simplifying the process of data retrieval.
# MAGIC - **Data Transformation and Cleansing**: The transformation from the bronze to the silver layer (and in many cases a gold layer), including basic cleansing and column name refinement, ensures that the data is not only more accessible but also of higher quality.
# MAGIC - **Scalable Data Storage**: By leveraging Databricks and [Delta Lake](https://docs.databricks.com/en/delta/index.html), you have stored your data in a structured format that supports scalable analysis and processing within a data lake architecture.
# MAGIC
# MAGIC ### Next Steps
# MAGIC
# MAGIC Now that you have some cleansed tables, let's explore the data from a data science perspective and determine which features we want to include in our model. Also, while we're building this set of notebooks in a linear fashion to facilitate learning, please note that in practice this is often a highly iterative process. You'll likely uncover things during data exploration that drive changes to your ingestion process.
# MAGIC
# MAGIC From here, please continue to the [Exploratory Data Analysis notebook]($./03_exploratory_data_analysis).
