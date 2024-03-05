# Databricks notebook source
# MAGIC %md
# MAGIC # Distributed bulk load example (optional)
# MAGIC
# MAGIC Some customers may have a large amount of data to be loaded from Salesforce CDP into Databricks, in which case the straightforward implementation may unfortunately timeout, or even if it succeeds may be excessively slow. There are a couple of approaches we can suggest in this situation. In this notebook, we look at one of the approaches: distributed ingest using primary key sampling. We assume here that we have a string id column as is typically found in Salesforce Data Cloud data model objects.
# MAGIC
# MAGIC Note that this approach also works with the dataset loaded in this demo, since we store the primary key as a text column of the form `IdXXXXXX`.
# MAGIC
# MAGIC The high level approach is as follows:
# MAGIC
# MAGIC 1. Do an initial top level aggregate query for a few key basic statistics of the table to be loaded, including the row count and the minimum and maximum id.
# MAGIC 2. Collect a small but useful sample of the keys and then assign tiles based on the number of shards desired. The aggregation happens on the Salesforce side for the tiling, and since its over only a sample of the entire dataset should still run relatively fast.
# MAGIC 3. Use the resulting keys as a guide for distributing the queries over the Databricks cluster. We'll use mapInPandas to execute a Python function on each core, each with its own connection, to collect the shard it is assigned. All of these will be collected in parallel, and since the primary keys are used directly, it should be an indexed query that should be performant.
# MAGIC
# MAGIC Let's get started!

# COMMAND ----------

# DBTITLE 1,Setup and common imports
# MAGIC %run ./common

# COMMAND ----------

# DBTITLE 1,Set the table name and it's ID column
table_name = "sfdc_byom_demo_train__dll"
id_col = "id__c"

# COMMAND ----------

# DBTITLE 1,Helper functions
def get_salesforce_cdp_connection():
    """Connect to Salesforce Data Cloud."""
    return SalesforceCDPConnection(
        sfdc_login_url, 
        sfdc_username, 
        sfdc_password,  
        sfdc_client_id,
        sfdc_client_secret)


def get_id_stats(conn, table_name, id_col):
    """Collect a few basic stats about the table and its ID column."""
    query = f"""
        SELECT
            count(*) AS count,
            min({id_col}) AS min_id,
            max({id_col}) AS max_id,
            max(length({id_col})) AS max_length
        FROM
            {table_name}"""
    df_pandas = conn.get_pandas_dataframe(query)
    return df_pandas.iloc[0].to_dict()


def get_shard_ranges(conn, table_name, id_col, n_shards, id_stats):
    """Get the shard ranges to use for collecting the dataset."""
    # We could potentially use the size of the table to determine 
    # the proportion to use here, since we collected it in id_stats.
    proportion = 1.0

    # Sample the id column at some proportion, and then within the 
    # resulting sample assign which shards each should fall in, and
    # finally aggregate the shards to find the start_id for each.
    query = f"""
        SELECT
            shard,
            MIN({id_col}) AS start_id
        FROM (
            SELECT 
                {id_col}, 
                NTILE({n_shards}) OVER (ORDER BY {id_col}) AS shard
            FROM (
                SELECT
                    {id_col}
                FROM
                    {table_name}
                TABLESAMPLE BERNOULLI({proportion})))
        GROUP BY shard
    """

    # Now the we have the sample, the first start should be close to 
    # the beginning, and the last should be close to the end. To guarentee
    # we don't miss any from the edges, we'll set the beginning of the first
    # shard to the empty string, which will sort lexicographically before 
    # anything else, and a string that is lexicographically higher than
    # any other string in our dataset. Each task will collect id >= start_id 
    # and id < end_id, which guarentees we get all the records, and statistically
    # should produce shards that are close to the same size.
    shards = conn.get_pandas_dataframe(query)
    shards.set_index("shard", inplace=True, drop=False)
    shards.sort_index(inplace=True)

    # Make sure the start_id of the first shard enables the >= check for the 
    # entire first shard. We could do this either by using an empty string, or
    # just by using the true min_id. Since we already have the min_id we can
    # use that.
    shards.loc[1, "start_id"] = id_stats["min_id"]
    shards.loc[1:(n_shards - 1), "end_id"] = shards.loc[2:, "start_id"].to_numpy()

    # Make sure the end_id of the last shard is higher than the max id we can get.
    # since we're dealing with strings, if we take the current max_id and just append
    # any extra character to it, the resulting string will meet that criteria
    # --- 
    # Note: We can't just use "max_id" here because the upper bound check for a shard
    #       must be < end_id. It has to be < end_id because we're only sampling the key
    #       space and don't have a mechanism to partition correctly in the other shards
    #       otherwise.
    extra_char = "_"
    greater_than_max_id = id_stats["max_id"] + extra_char
    assert id_stats["max_id"] < greater_than_max_id
    shards.loc[n_shards, "end_id"] = greater_than_max_id

    return shards
    

from contextlib import contextmanager

@contextmanager
def spark_conf_set(key, value):
    """Temporarily set a spark config setting within a code block."""
    prior_value = spark.conf.get(key)
    try:
        yield spark.conf.set(key, value)
    finally:
        spark.conf.set(key, prior_value)

# COMMAND ----------

# DBTITLE 1,Establish connection to Salesforce Data Cloud
conn = get_salesforce_cdp_connection()

# COMMAND ----------

# DBTITLE 1,Define the query and template
import jinja2

user_query_string = f"""
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
  {table_name}
"""

query_template_string = f"""
SELECT
  *
FROM (
  {user_query_string}
)
WHERE {id_col} >= '{{{{ start_id }}}}' AND {id_col} < '{{{{ end_id }}}}'
"""


# COMMAND ----------

# DBTITLE 1,Collect a small sample for the schema
from pyspark.sql import types as T

sample_query = f"""{user_query_string} LIMIT 10"""
sample_result_pandas = conn.get_pandas_dataframe(sample_query)
sample_result = spark.createDataFrame(sample_result_pandas)
result_schema = sample_result.schema
result_schema.add(T.StructField('shard', T.LongType(), True));

# COMMAND ----------

# DBTITLE 1,Define ingestion function
def ingest_records(dfs):
    """Ingest a batch of records.
    
    In general we'll get only a single dataframe, but if we end up with 
    more than one it's no problem. Each dataframe would consist of one or
    a few rows specifying the shard assigned along with its start and end
    id. We append the shard for this example just so we can assess the
    resulting distribution, but it would be fine to remove that later on
    if its not needed.
    """
    # Each worker core will need its own connection.
    conn = get_salesforce_cdp_connection()

    # Along with its own jinja environment.
    environment = jinja2.Environment()

    # Set up the query from the template string we closed over from earlier.
    query_template = environment.from_string(query_template_string)

    for df in dfs:
        for i, (shard, start_id, end_id) in df.iterrows():
            
            # Query for this particular shard using the query template
            query = query_template.render(start_id=start_id, end_id=end_id)
            df = conn.get_pandas_dataframe(query)

            # Append the shard so we can analyze it later and return the result
            df['shard'] = shard
            yield df

# COMMAND ----------

# DBTITLE 1,Define the shards to collect
num_shards = 32
id_stats = get_id_stats(conn, table_name, id_col)
shard_ranges = get_shard_ranges(conn, table_name, id_col, num_shards, id_stats)
df_shards = spark.createDataFrame(shard_ranges)
display(df_shards)

# COMMAND ----------

# DBTITLE 1,Define the dataset in terms of the shards to collect
ingested_data = (
    df_shards
    .repartition(num_shards)
    .mapInPandas(ingest_records, result_schema))

# COMMAND ----------

# DBTITLE 1,Inspect the results
display(ingested_data)

# COMMAND ----------

# DBTITLE 1,Inspect the resulting shard sizes
# Let's see how many records we ended up with in each shard.
shard_counts = (
    ingested_data
    .groupBy("shard")
    .count()
    .orderBy("shard"))

# Display the shard counts (you may need to re-add the bar chart visualization)
with spark_conf_set("spark.sql.adaptive.enabled", "false"):
    display(shard_counts)

# COMMAND ----------

# DBTITLE 1,Make sure we got all the records
# As a sanity check, let's just make sure we got all the records we're expecting.
record_count = ingested_data.count()
if id_stats["count"] == record_count:
    print(f"We got exactly {record_count} records as expected.")
else:
    print(f"Oops, we only got {record_count} records, but expected {id_stats['count']}!")

# COMMAND ----------

# DBTITLE 1,Benchmark with a noop write
# Note: We're turning off AQE here because of how the DataFrame is
#       for planning the shards collection. In general, AQE is great
#       and you should leave it on whenever possible. However, when
#       you have a DataFrame where you have a small number of rows
#       but each row actually drives a lot of compute, AQE will still
#       in many cases try to coalesce partitions which we don't want.
#       Here, we really do want the partitions to stay the same even
#       through from AQE's perspective that may seem suboptimal. We
#       use a context manager to make sure its only in effect temporarily
#       as we execute this query.

with spark_conf_set("spark.sql.adaptive.enabled", "false"):
    ingested_data.write.format("noop").mode("overwrite").save()
