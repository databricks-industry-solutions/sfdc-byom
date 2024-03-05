# Databricks notebook source
# MAGIC %md 
# MAGIC You may find this series of notebooks at https://github.com/databricks-industry-solutions/sfdc-byom. For more information about this solution accelerator, visit https://www.databricks.com/solutions/accelerators/sfdc-byom.

# COMMAND ----------

# MAGIC %md
# MAGIC # Exploratory Data Analysis
# MAGIC
# MAGIC This is the exploratory data analysis notebook in our series on integrating Salesforce Data Cloud with the Databricks Data Intelligence Platform. In the preceding notebook, we successfully ingested product interest data from Salesforce Data Cloud into our Databricks environment, laying the groundwork for sophisticated data-driven insights. This notebook is dedicated to dissecting and understanding that ingested data through exploratory data analysis (EDA) techniques. 
# MAGIC
# MAGIC EDA is a critical step in the data science workflow as it allows us to uncover patterns, anomalies, and relationships in the data, providing valuable insights that inform subsequent stages of feature engineering and model development. By visualizing and summarizing our dataset, we aim to achieve a deep understanding of its characteristics and idiosyncrasies, which is essential for preparing the data for effective machine learning.
# MAGIC
# MAGIC As we proceed, we will explore various dimensions of the product interest data, employing a mix of statistical summaries and visualizations to identify trends, distributions, and potential outliers. This process will not only aid in ensuring the quality and integrity of our data but also in uncovering opportunities for feature creation that can enhance the performance of our eventual product recommendation model.
# MAGIC
# MAGIC Let's dive into the data and uncover the insights that will drive our next steps towards developing a powerful product recommendation system hosted in Databricks.

# COMMAND ----------

# DBTITLE 1,Run common setup
# MAGIC %run ./common

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load the silver table
# MAGIC
# MAGIC To get started, let's load the silver table we created in the ingestion task we just finished. Running our exploratory data analysis from data already loaded in the lakehouse let's us iterate much faster as we don't have to worry about making a connection back to Salesforce each time we want to run a query, and in the lakehouse data is optimized for running big data analytics. Also, in terms of medallion architecture, our intent is to be working from a cleansed dataset, so either silver or gold tables. If we identify any data quality issues during our EDA process, we would want to propagate those cleansing steps upstream to the bronze to silver transition.

# COMMAND ----------

# DBTITLE 1,Load and view silver product interest table
df_spark = spark.table("product_interest_silver").drop("id")
df = df_spark.toPandas()
display(df_spark)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run data profiler
# MAGIC
# MAGIC The Databricks built-in data profiling capability is a powerful tool designed to assist in exploratory data analysis (EDA) tasks. It provides an automated way to generate detailed statistical summaries and visualizations of the dataset, offering insights into its structure, quality, and characteristics. The data profiler in Databricks helps data scientists and analysts understand the data they are working with and make informed decisions about preprocessing, feature engineering, and modeling. This is the ideal place for us to begin our exploratory data analysis for the product data, because it gives us all of the following capabilities with a single line of code:
# MAGIC
# MAGIC - **Automated statistical summary:** The data profiler generates descriptive statistics, such as mean, median, standard deviation, minimum, and maximum values for each numerical column in the dataset. This summary provides a quick overview of the dataset's central tendencies, dispersion, and shape.
# MAGIC
# MAGIC - **Distribution visualizations:** The profiler generates visualizations to display the distribution of numerical variables, helping to identify potential outliers, skewness, and other important trends. These visualizations can include histograms, box plots, and density plots, among others.
# MAGIC
# MAGIC - **Categorical variable analysis:** The profiler also analyzes categorical variables by counting the frequency of each category. This information helps to understand the distribution and prevalence of different categories and can be useful for feature engineering or stratified analysis.
# MAGIC
# MAGIC - **Missing values detection:** The profiler identifies missing values in the dataset and reports the percentage of missing values for each column. This information is essential for determining the appropriate handling of missing data, such as imputation or removal.
# MAGIC
# MAGIC - **Correlation analysis:** The data profiler can calculate the correlation between numerical variables to identify any significant relationships. This analysis helps to understand the interdependencies between variables and can guide feature selection or transformation.
# MAGIC
# MAGIC - **Easy integration with Databricks environment:** The data profiler is seamlessly integrated into the Databricks environment, allowing users to execute the profiling on large-scale datasets efficiently. It leverages distributed computing capabilities to handle big data effectively.
# MAGIC
# MAGIC There are two ways to run the data profiler:
# MAGIC
# MAGIC 1. **Using the UI flow:** Any time you display a Spark dataframe in Databricks you have the option to add visualization tabs to the main table output. When you click the plus icon to add a visualization, you can also add a data profile.
# MAGIC
# MAGIC 2. **Calling via the dbutils library:** The same functionality is accessible via code by simply calling the dbutils method `dbutils.data.summarize(df)`. This will output the same results that the UI flow would produce.
# MAGIC
# MAGIC Let's try running it on our product interest silver table now using the dbutils library approach.

# COMMAND ----------

dbutils.data.summarize(df_spark)

# COMMAND ----------

# MAGIC %md
# MAGIC ## View basic descriptive statistics
# MAGIC
# MAGIC While the profiler gives us a lot of information quickly, often times we'll still want to look at individual attributes using the same approach you'd use anywhere else in the Python data analysis ecosystem.
# MAGIC
# MAGIC In this section, we delve into the fundamental statistics of our dataset to establish a foundational understanding of the product interest data. Descriptive statistics are crucial as they provide a quick summary of the central tendencies, dispersion, and shape of our dataset's distribution. We will use both Spark DataFrame and Pandas DataFrame functionalities to calculate measures such as mean, median, standard deviation, minimum, and maximum values for each numerical column. Additionally, we will examine the distribution of categorical variables by counting the frequency of each category.
# MAGIC
# MAGIC This statistical analysis serves as the first step in identifying patterns, detecting outliers, and understanding the data's overall structure. It is instrumental in guiding our data preprocessing decisions, such as handling missing values, scaling and normalizing data, and potentially identifying features that could be relevant for our predictive model.
# MAGIC
# MAGIC By scrutinizing these statistics, we aim to uncover insights that will inform the more detailed exploratory analysis and feature engineering tasks ahead, ultimately enhancing the performance of our product recommendation model. Let's proceed to analyze our dataset's descriptive statistics to gain a clear view of its characteristics.

# COMMAND ----------

# DBTITLE 1,Check how many rows and columns
# Print number of rows and columns of the dataframe
df.shape

# COMMAND ----------

# DBTITLE 1,Display the first couple of rows of the dataframe
# Preview data
display(df.head())

# COMMAND ----------

# DBTITLE 1,Check for missing values
# Check for missing values
df.isna().sum()

# COMMAND ----------

# DBTITLE 1,View numerical predictors statistics
# View numerical predictors statistics
display(df.describe().reset_index(drop=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Custom visualizations
# MAGIC
# MAGIC We can also define our own per column type visualizations and summaries (e.g., categorical and numerical) and apply those to each of the columns in our dataset for a more detailed view. Let's look at an example of this with the same product interest dataset.

# COMMAND ----------

# DBTITLE 1,Helper visualization methods
def describe_categorical(t):
    """Create descriptive statistics of categorical variables."""
    uniquecategories = len(list(t.unique()))
    print("Number of Unique Categories : " + str(uniquecategories))
    tmp = pd.DataFrame()
    tmp = t.value_counts().reset_index(name='counts').rename({'index': 'Categories'}, axis=1)
    tmp['%'] = 100 * tmp['counts'] / tmp['counts'].sum()
    print(tmp)
    tmp['percentages'] = tmp['%'].apply(lambda x: '{:.2f}%'.format(x))
    tmp.sort_values(by = '%', inplace = True, ascending = False)
    ax = tmp.plot(x="Categories", y=["counts"], kind="bar")  
    for i, val in enumerate(tmp['counts']):
        ax.text(i, val, tmp['percentages'][i], horizontalalignment='center')


def describe_continuous(t):
    """Create descriptive statistics of continous variables."""
    t.describe()
    fig, ax=plt.subplots(nrows =1,ncols=3,figsize=(10,8))
    ax[0].set_title("Distribution Plot")
    #sns.histplot(t,ax=ax[0])
    sns.kdeplot(t,fill=True, ax=ax[0])
    ax[1].set_title("Violin Plot")
    sns.violinplot(y=t,ax=ax[1], inner="quartile")
    ax[2].set_title("Box Plot")
    sns.boxplot(y=t,ax=ax[2],orient='v')  

# COMMAND ----------

# DBTITLE 1,Describe the product purchased label (our `y` variable)
describe_categorical(df['product_purchased']) 

# COMMAND ----------

# DBTITLE 1,Describe the `campaign` feature
describe_categorical(df['campaign']) 

# COMMAND ----------

# DBTITLE 1,Describe the `club member` feature
describe_categorical(df['club_member']) 

# COMMAND ----------

# DBTITLE 1,Describe the `state` feature
describe_categorical(df['state']) 

# COMMAND ----------

# DBTITLE 1,Describe the `month` feature
describe_categorical(df['month']) 

# COMMAND ----------

# DBTITLE 1,Describe the `case type return` feature
describe_categorical(df['case_type_return']) 

# COMMAND ----------

# DBTITLE 1,Describe the `case type shipment damaged` feature
describe_categorical(df['case_type_shipment_damaged'])

# COMMAND ----------

# DBTITLE 1,Describe the `case count` feature
describe_continuous(df['case_count']) 

# COMMAND ----------

# DBTITLE 1,Describe the `pages visited` feature
describe_continuous(df['pages_visited']) 

# COMMAND ----------

# DBTITLE 1,Describe the `engagement score` feature
describe_continuous(df['engagement_score'])

# COMMAND ----------

# DBTITLE 1,Describe the `tenure` feature
describe_continuous(df['tenure'])

# COMMAND ----------

# MAGIC %md
# MAGIC ## View correlation matrix
# MAGIC
# MAGIC Another common technique that is helpful during exploratory data analysis is to view a correlation matrix of the different variables in the dataset. This can be useful for a variety of reasons:
# MAGIC
# MAGIC - **Identify Relationships:** Quickly identify and visualize the strength and direction of relationships between different variables imported from Salesforce Data Cloud. This is crucial for understanding how different Salesforce fields relate to each other, which can inform data cleaning, feature selection, and predictive modeling.
# MAGIC
# MAGIC - **Data Cleaning and Preprocessing:** Spotting highly correlated variables can inform decisions about which variables to keep, combine, or discard, improving model performance and interpretation.
# MAGIC
# MAGIC - **Simplification:** By excluding duplicate correlations, the visualization becomes less cluttered, making it easier for stakeholders to interpret the results without a deep statistical background.
# MAGIC
# MAGIC - **Interactive Exploration:** In a notebook environment, this function complements interactive EDA by allowing users to dynamically explore different subsets of their data and immediately see the impact on variable relationships.
# MAGIC
# MAGIC Let's define a simple helper function to create such a correlation matrix from our dataset.

# COMMAND ----------

# DBTITLE 1,Correlation matrix helper method
def correlation_matrix(df, dropDuplicates = True):

    # Exclude duplicate correlations by masking uper right values
    if dropDuplicates:    
        mask = np.zeros_like(df, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True

    # Set background color / chart style
    sns.set_style(style = 'white')

    # Set up  matplotlib figure
    f, ax = plt.subplots(figsize=(5, 5))

    # Add diverging colormap from red to blue
    cmap = sns.diverging_palette(250, 10, as_cmap=True)

    # Draw correlation plot with or without duplicates
    if dropDuplicates:
        sns.heatmap(df, mask=mask, cmap=cmap, 
                square=True,
                linewidth=.5, cbar_kws={"shrink": .5}, ax=ax)
    else:
        sns.heatmap(df, cmap=cmap, 
                square=True,
                linewidth=.5, cbar_kws={"shrink": .5}, ax=ax)


# COMMAND ----------

# MAGIC %md
# MAGIC Now that we have the method defined, let's apply it to our dataset. 
# MAGIC
# MAGIC Note that we also need to expand out the categorical columns with one hot encoding to create dummy variables for their individual values.

# COMMAND ----------

# DBTITLE 1,View correlation matrix
# define catagorical columns to convert
cat_columns = ['state', 'campaign', 'product_purchased']

# convert all categorical variables to numeric
df_dummies =  pd.get_dummies(df , columns = cat_columns, drop_first = True)

correlation_matrix(df_dummies.corr(), dropDuplicates = False)

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC We can also use this correlation information to determine if any values are correlated to the point that they provide completely redundant information, which could severely impact our analysis and the quality of the resulting model. For this, let's define a few additional helper functions and apply those to our dataset.

# COMMAND ----------

# DBTITLE 1,Redundant pairs helper function
def get_redundant_pairs(df):
    """Get diagonal and lower triangular pairs of correlation matrix."""
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

def get_top_abs_correlations(df, n=5):
    au_corr = df.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]

print("Top absolute correlations")
print(get_top_abs_correlations(df_dummies, 5))

# COMMAND ----------

# MAGIC %md
# MAGIC The correlation between predictors is not found to be significant so we do have to drop any predictors

# COMMAND ----------

# MAGIC %md
# MAGIC ## Conclusion
# MAGIC
# MAGIC That's it for our exploratory data analysis notebook. We demonstrated a couple of different ways to visualize and summarize the example product interest dataset for our scenario, and along the way we collected a variety of additional transformations to consider (some of which have already been applied to the silver table). 
# MAGIC
# MAGIC Here's a summary of some useful transformations to consider for data preparation based on this analysis.
# MAGIC
# MAGIC 1. Remove null records
# MAGIC 2. Since the products purchased have a class imbalance we will need to balance the classes to reduce bias
# MAGIC 3. Since club member is a binary predictor treat it as categorical
# MAGIC 4. Since purchases in some states are much greater than others the states with smaller % of customers should be combined as other
# MAGIC 5. Month is should be treated as categorical
# MAGIC 6. Case counts greater than 3 can be combined into a single category
# MAGIC 7. Since Case Type Return is a binary predictor treat it as categorical
# MAGIC 8. Since Case Type Shipment Damaged is a binary predictor treat it as categorical
# MAGIC 9. Engagement score and Clicks need to be scaled
# MAGIC 10. Tenure needs to be treated as categorical and greater than 3 years can be combined into one bucket. 
# MAGIC
# MAGIC Also, please note again that while these notebooks are being presented in a linear fashion, the EDA, data cleansing, data preparation and the rest of the model creation process are often highly iterative. Many of the transformations and observations you make along the way may end up belonging upstream in the data pipeline (for instance, all the way back in Salesforce or some other data source), in the data cleansing and loading process, in the featurization process, or even within the model transformation pipeline. Keep this in mind as you adapt this notebook and the others to your production Salesforce Data Cloud and Databricks Machine Learning projects!
# MAGIC
# MAGIC When you're ready, please proceed to the next notebook: [Feature Engineering]($./04_feature_engineering).
