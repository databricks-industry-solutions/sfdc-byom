# Databricks notebook source
# MAGIC %md 
# MAGIC You may find this series of notebooks at https://github.com/databricks-industry-solutions/sfdc-byom. For more information about this solution accelerator, visit https://www.databricks.com/solutions/accelerators/sfdc-byom.

# COMMAND ----------

# MAGIC %md
# MAGIC # Set Up, Build, Train, and Deploy model in Databricks

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Use case and introduction
# MAGIC
# MAGIC Northern Outfitters, a well-established online retail store specializing in clothing, winter gear, and backpacks, is a prominent user of Salesforce. They extensively utilize various clouds, including sales, service, community, and marketing, to efficiently manage customer operations. Despite allocating significant resources to marketing promotions, the current methodology is expensive and lacks precision in targeting.
# MAGIC
# MAGIC The company's objective is to transition to targeted promotions, focusing on specific products to optimize sales and improve return on investment. Customizing promotions based on individual customer preferences and interests is expected to boost conversion rates and overall customer satisfaction. Northern Outfitters places a high value on providing outstanding service to its club members, aiming to deliver a personalized experience with call center agents offering a "white glove treatment" to these customers.
# MAGIC
# MAGIC The integration of DataCloud has allowed Northern Outfitters to ingest, prepare, and consolidate customer profiles and behaviors from different Salesforce clouds and enterprise systems. This integration has led to the creation of a unified customer view, and the company plans to leverage this comprehensive customer data for strategic intelligence.
# MAGIC
# MAGIC To bridge the gap between data scientists' machine learning models and the system of engagement for sales, service, and marketing teams, Northern Outfitters is seeking a solution that seamlessly integrates data-driven insights into the day-to-day workflows of their employees. By empowering their teams with actionable insights, the company aims to enhance decision-making, improve customer interactions, and automate customer addition to marketing journeys.
# MAGIC
# MAGIC
# MAGIC The objective of this exercise is to create a predictive model for identifying customer product interests. This model will then be utilized to generate personalized experiences and offers for customers. The development of the model is based on historical data, including customer demographics, marketing engagements, and purchase history.
# MAGIC
# MAGIC The dataset comprises 1 million records, each containing observations and information about potential predictors and the products historically purchased by customers. 
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
