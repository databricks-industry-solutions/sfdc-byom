# Databricks notebook source
# MAGIC %md This notebook sets up the companion cluster(s) to run the solution accelerator. It also creates the Workflow to illustrate the order of execution. Happy exploring! 
# MAGIC 🎉
# MAGIC
# MAGIC **Steps**
# MAGIC 1. Simply attach this notebook to a cluster and hit Run-All for this notebook. A multi-step job and the clusters used in the job will be created for you and hyperlinks are printed on the last block of the notebook. 
# MAGIC
# MAGIC 2. Run the accelerator notebooks: Feel free to explore the multi-step job page and **run the Workflow**, or **run the notebooks interactively** with the cluster to see how this solution accelerator executes. 
# MAGIC
# MAGIC     2a. **Run the Workflow**: Navigate to the Workflow link and hit the `Run Now` 💥. 
# MAGIC   
# MAGIC     2b. **Run the notebooks interactively**: Attach the notebook with the cluster(s) created and execute as described in the `job_json['tasks']` below.
# MAGIC
# MAGIC **Prerequisites** 
# MAGIC 1. You need to have cluster creation permissions in this workspace.
# MAGIC
# MAGIC 2. In case the environment has cluster-policies that interfere with automated deployment, you may need to manually create the cluster in accordance with the workspace cluster policy. The `job_json` definition below still provides valuable information about the configuration these series of notebooks should run with. 
# MAGIC
# MAGIC **Notes**
# MAGIC 1. The pipelines, workflows and clusters created in this script are not user-specific. Keep in mind that rerunning this script again after modification resets them for other users too.
# MAGIC
# MAGIC 2. If the job execution fails, please confirm that you have set up other environment dependencies as specified in the accelerator notebooks. Accelerators may require the user to set up additional cloud infra or secrets to manage credentials. 
# MAGIC
# MAGIC 3. The job doesn't deploy the model serving endpoint to prevent situations in which additional compute charges are incurred unexpectedly. However, it is perfectly fine and expected that you may want to do this in your workflows. Feel free to add this task and the downstream model testing and monitoring notebooks back into the workflow as needed.

# COMMAND ----------

# DBTITLE 0,Install util packages
# MAGIC %pip install git+https://github.com/databricks-academy/dbacademy@v1.0.13 git+https://github.com/databricks-industry-solutions/notebook-solution-companion@safe-print-html --quiet --disable-pip-version-check
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from solacc.companion import NotebookSolutionCompanion

# COMMAND ----------

libraries = [
    {
        "pypi": {
            "package": "salesforce-cdp-connector==1.0.13"
        }
    }
]

job_json = {
        "timeout_seconds": 28800,
        "max_concurrent_runs": 1,
        "tags": {
            "usage": "solacc_testing",
            "group": "SOLACC"
        },
        "tasks": [
            {
                "job_cluster_key": "sfdc_byom_cluster",
                "notebook_task": {
                    "notebook_path": f"01_introduction"
                },
                "libraries": libraries,
                "task_key": "sfdc_byom_01"
            },
            {
                "job_cluster_key": "sfdc_byom_cluster",
                "notebook_task": {
                    "notebook_path": f"02_ingest_data"
                },
                "task_key": "sfdc_byom_02",
                "libraries": libraries,
                "depends_on": [
                    {
                        "task_key": "sfdc_byom_01"
                    }
                ]
            },
            {
                "job_cluster_key": "sfdc_byom_cluster",
                "notebook_task": {
                    "notebook_path": f"03_exploratory_data_analysis"
                },
                "task_key": "sfdc_byom_03",
                "libraries": libraries,
                "depends_on": [
                    {
                        "task_key": "sfdc_byom_02"
                    }
                ]
            },
            {
                "job_cluster_key": "sfdc_byom_cluster",
                "notebook_task": {
                    "notebook_path": f"04_feature_engineering"
                },
                "task_key": "sfdc_byom_04",
                "libraries": libraries,
                "depends_on": [
                    {
                        "task_key": "sfdc_byom_03"
                    }
                ]
            },
            {
                "job_cluster_key": "sfdc_byom_cluster",
                "notebook_task": {
                    "notebook_path": f"05_build_and_train_model"
                },
                "task_key": "sfdc_byom_05",
                "libraries": libraries,
                "depends_on": [
                    {
                        "task_key": "sfdc_byom_04"
                    }
                ]
            }
        ],
        "job_clusters": [
            {
                "job_cluster_key": "sfdc_byom_cluster",
                "new_cluster": {
                    "spark_version": "14.3.x-cpu-ml-scala2.12",
                "spark_conf": {
                    "spark.databricks.delta.formatCheck.enabled": "false"
                    },
                    "num_workers": 2,
                    "node_type_id": {"AWS": "i3.xlarge", "MSA": "Standard_DS3_v2", "GCP": "n1-highmem-4"},
                    "custom_tags": {
                        "usage": "solacc_testing"
                    },
                }
            }
        ]
    }

# COMMAND ----------

spark.sql(f"CREATE DATABASE IF NOT EXISTS databricks_solacc LOCATION '/databricks_solacc/'")
spark.sql(f"CREATE TABLE IF NOT EXISTS databricks_solacc.dbsql (path STRING, id STRING, solacc STRING)")
dbsql_config_table = "databricks_solacc.dbsql"

# COMMAND ----------

dbutils.widgets.dropdown("run_job", "False", ["True", "False"])
run_job = dbutils.widgets.get("run_job") == "True"
nsc = NotebookSolutionCompanion()
nsc.deploy_compute(job_json, run_job=run_job)
#_ = nsc.deploy_dbsql("./dashboards/IoT Streaming SA Anomaly Detection.dbdash", dbsql_config_table, spark)
