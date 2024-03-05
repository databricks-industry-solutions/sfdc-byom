# Databricks notebook source
# MAGIC %md 
# MAGIC You may find this series of notebooks at https://github.com/databricks-industry-solutions/sfdc-byom. For more information about this solution accelerator, visit https://www.databricks.com/solutions/accelerators/sfdc-byom.

# COMMAND ----------

# MAGIC %md
# MAGIC # Build and Train Model
# MAGIC
# MAGIC Now for the fun part! Up to this point, we've loaded data from Salesforce in a way that it can be efficiently cleaned up and combined with additional data from all around our organization. Then we did exploratory data analysis and prepared features in our feature table. We're finally ready to put those features to use and train our machine learning model.
# MAGIC
# MAGIC Let's get to it!

# COMMAND ----------

# DBTITLE 1,Run common setup
# MAGIC %run ./common

# COMMAND ----------

# MAGIC %md
# MAGIC ## Import libraries
# MAGIC
# MAGIC For this example, we're going to create an XGBoost model using the scikit-learn interface. We'll exploit the power of our Databricks cluster by conducting our hyperparameter sweep in parallel over the cluster. To help us use all this great functionality, we first need to import the relevant libraries.

# COMMAND ----------

# DBTITLE 1,Import libraries
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import cross_val_score
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
import hyperopt
from hyperopt import hp

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define transformation logic
# MAGIC
# MAGIC While feature tables are great for collecting aggregate and detailed information for specific entities, there is often a few final transformations that are better suited for transformations within the machine learning pipeline itself. Common examples are various categorical encoding methods like one-hot encoding, standardization methods and the like. That will be the case here as well, and we'll provide those transformation later as part of the scikit-learn pipeline. 
# MAGIC
# MAGIC However, there are a couple of other preprocessing steps we're going to need to fit in to adjust the dataframe slighltly to make sure we match up with what Salesforce is going to pass to our model when it calls it as part of inference down the line. These transformations will be applied outside the model pipeline, but within the model wrapper.

# COMMAND ----------

# DBTITLE 1,Custom transform helper function
# Apply custom transforms here
def transform(X):
    # Define the non-other (retained) states list
    retained_states_list = [
        'Washington', 
        'Massachusetts', 
        'California', 
        'Minnesota', 
        'Vermont', 
        'Colorado', 
        'Arizona']
        
    # Object conversions
    int_object_cols = [
        "club_member",
        "month",
        "case_type_return",
        "case_type_shipment_damaged"]
    
    # Define columns to drop
    dropped_cols = ["state", "case_count", "tenure"]

    # Convert predictor types
    for c in int_object_cols:
        X[c] = X[c].astype(int).astype(object)

    # Implement your custom formula with if statement
    # For example, if you want to create a new column based on a condition:
    X['transformed_state'] = X['state'].apply(
        lambda x: 'Other' if x not in retained_states_list else x) 
    X['transformed_cases'] = X['case_count'].apply(
        lambda x: 'No Cases' if x == 0 else '1 to 2 Cases' if x <= 2 else 'Greater than 2 Cases') 
    X['transformed_tenure'] = X['tenure'].apply(
        lambda x: 'Less than 1' if x < 1 else '1 to 2 Years' if x == 1 else '1 to 2 Years' if x == 2 else '2 to 3 Years' if x == 3 else 'Greater than 3 Years')
    
    # Remove columns to ignore
    X = X.drop(dropped_cols, axis=1)

    # Rename certain columns
    X = X.rename(columns={
        'transformed_state': 'state', 
        'transformed_cases': 'case_count', 
        'transformed_tenure': 'tenure'})
    return X 

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read feature table from Unity Catalog
# MAGIC
# MAGIC In the previous notebook, we used the feature engineering API to write feature tables to Unity Catalog.
# MAGIC
# MAGIC In this notebook, we're going to use the same `FeatureEngineeringClient` to load those features back as a training set to train our model.
# MAGIC
# MAGIC We get started in the same way, importing and instantiating the client.

# COMMAND ----------

# DBTITLE 1,Import the feature engineering library
from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup

fe = FeatureEngineeringClient()

# COMMAND ----------

# MAGIC %md
# MAGIC Feature tables work by providing the keys of the data to look up as a batch dataframe we pass to the API. We also need to provide our label. In this case, we're loading from the same silver table. In practice however, the feature tables will often be updated by some separate pipelines and the labels will likely come from a different source anyway. The main takeaway here is that we need to make sure we create this batch data frame to use to drive the feature lookups and provide the labels for our model.

# COMMAND ----------

# DBTITLE 1,Create batch dataframe with keys and labels
batch_df = (
    spark.table("product_interest_silver")
    .select("id", "product_purchased"))

batch_df.printSchema()

display(batch_df)

# COMMAND ----------

# MAGIC %md
# MAGIC Now that we have the batch dataframe, we can create our training set. 
# MAGIC
# MAGIC The training set object is created to combine the batch dataframe with the set of features to look up, as well as a mapping that tells it which lookup key in the batch dataframe should match the primary key in the feature table. In our case, it's simply the `id` field again. We didn't provide any specific feature names or renaming mapping, so this will give us all the features back from the table.

# COMMAND ----------

# DBTITLE 1,Create training set from feature lookups
feature_lookups = [
    FeatureLookup(
        table_name="product_interest_features",
        lookup_key="id")
]

training_set = fe.create_training_set(
    df=batch_df, 
    feature_lookups=feature_lookups,
    label="product_purchased")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load and split the training set
# MAGIC
# MAGIC With the training set object defined, we can now load the data and create a Pandas dataframe from it as we would for basically any other scikit-learn based model. Once we load up all the data, we can then split it into the normal train, test, validation splits and apply the transformation helper function we defined earlier.

# COMMAND ----------

df_pandas = training_set.load_df().toPandas()

# Separate features and target variable
X = df_pandas.drop("product_purchased", axis=1)
y = df_pandas["product_purchased"]

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split data into full training and held-out testing sets
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42)

# Further divide full training set it training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=42)

# Apply pre-processing logic to the splits
X_train_full = transform(X_train_full)
X_train = transform(X_train)
X_test = transform(X_test)
X_val = transform(X_val)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Additional transformations
# MAGIC
# MAGIC There are still a few columns in our dataset that need to be preprocessed just a bit. We want to apply standard scaling to all our numeric features, and one hot encoding to all our categorical features.

# COMMAND ----------

# DBTITLE 1,Define the preprocessor transform
numeric_features = [
    'engagement_score', 
    'clicks',
    'pages_visited']

categorical_features = [
    'club_member', 
    'campaign', 
    'state', 
    'month', 
    'case_count', 
    'case_type_return', 
    'case_type_shipment_damaged', 
    'tenure']

preprocessor = ColumnTransformer(
    transformers=[
        ('numeric', StandardScaler(), numeric_features),
        ('categorical', OneHotEncoder(handle_unknown="ignore"), categorical_features)])

# COMMAND ----------

# MAGIC %md
# MAGIC To use hyperopt for our hyperparameter sweep, we need to define our search space.

# COMMAND ----------

# DBTITLE 1,Define the hyperopt search space
from hyperopt.pyll import scope

search_space = {
    'classifier__n_estimators': scope.int(hp.quniform('n_estimators', 100, 1001, 100)),
    'classifier__max_depth': scope.int(hp.quniform('max_depth', 3, 9, 4)),
    'classifier__learning_rate': hp.loguniform('learning_rate', -2, 0),
    'classifier__subsample': hp.uniform('subsample', 0.8, 1.0),
    'classifier__colsample_bytree': hp.uniform('colsample_bytree', 0.8, 1.0),
    'classifier__gamma': hp.uniform('gamma', 0, 0.2),  # Range from 0 to 0.2 (inclusive) with 3 values
    'classifier__reg_alpha': hp.uniform('reg_alpha', 0, 1.0),  # Range from 0 to 1 (inclusive) with 3 values
    'classifier__reg_lambda': hp.uniform('reg_lambda', 0, 1.0),
}

# COMMAND ----------

# MAGIC %md
# MAGIC It can be helpful for debugging purposes to have a static set of hyperparameters to test the model structure against.

# COMMAND ----------

# DBTITLE 1,Define a sample set of params for debugging
# Define a set of simple parameters to run a simple trial run and for debugging
static_params = {
    'classifier__n_estimators': 100,
    'classifier__max_depth': 3,
    'classifier__learning_rate': 0.01,
    'classifier__subsample': 0.8,
    'classifier__colsample_bytree': 0.8,
    'classifier__gamma': 0.0,
    'classifier__reg_alpha': 0.0,
    'classifier__reg_lambda': 0.0
}

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define the model and pipeline
# MAGIC
# MAGIC Now that we have the search space and preprocessing logic defined, let's create the actual classifier and bundle it with the preprocessor to create a pipeline.

# COMMAND ----------

# DBTITLE 1,Define the model and pipeline
# Define the xgb classifier
xgb_classifier = xgb.XGBClassifier(objective='multi:softmax')

# Create the ML pipeline
pipeline = Pipeline(steps=[
    ('transformer', preprocessor),
    ('classifier', xgb_classifier)])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test run
# MAGIC
# MAGIC Before we do a full hyperparameter sweep and train our final model, we can do a quick test run with the static set of hyperparameters we defined earlier.

# COMMAND ----------

# DBTITLE 1,Quick test run
# Run the quick trial run (leaving in here to help debugging)
with mlflow.start_run(run_name=f"{model_name}_static_params") as test_run:
    params = static_params
    pipeline.set_params(**params)
    pipeline.fit(X_train, y_train)
    mlflow.log_params(params)
    y_hat = pipeline.predict(X_val)
    weighted_f1_score = metrics.f1_score(y_val, y_hat, average="weighted")
    accuracy_score = metrics.accuracy_score(y_val, y_hat)
    mlflow.log_metric("weighted_f1_score", weighted_f1_score)
    mlflow.log_metric("accuracy_score", accuracy_score)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define objective function for hyperopt
# MAGIC
# MAGIC In addition to the search space, we also need to define our objective function for hyperopt. This is the function that hyperopt will probe using its best choices of hyperparameters from the search space we defined. In our case, we just need to train the model using the train split we carved out earlier and then evaluate the performance of that set of hyperparameters using some metric on our validation set. In our case, we'll use a weighted f1 score to cover the multiple classes we're defining for our recommender. Note that since hyperopt provides a minimization function, but for f1 score more is better, we need to multiply our metric by -1 before we return it.
# MAGIC
# MAGIC This is also where we first bump into MLflow. Here, we log a nested run to capture the combination of the set of parameters used for this particular sub-run along with the metrics it produced. However, we don't need to capture anything else, like the model itself. Once we have the best set of hyperparameters, we'll retrain over the full training set and evaluate that using our hold-out test set.

# COMMAND ----------

# DBTITLE 1,Define hyperopt objective function
# Define objective function for hyperopt
def objective_fn(params):
    with mlflow.start_run(nested=True):
        pipeline.set_params(**params)
        mlflow.log_params(params)
        pipeline.fit(X_train, y_train)
        y_hat = pipeline.predict(X_val)
        weighted_f1_score = metrics.f1_score(y_val, y_hat, average="weighted")
        accuracy = metrics.accuracy_score(y_val, y_hat)
        mlflow.log_metric("weighted_f1_score", weighted_f1_score)
        mlflow.log_metric("accuracy", accuracy)

        # Set the loss to -1*weighted_f1_score so fmin maximizes the weighted_f1_score
        return {"status": hyperopt.STATUS_OK, "loss": -1 * weighted_f1_score}


# COMMAND ----------

# MAGIC %md
# MAGIC Also, as mentioned earlier we're going to run this hyperparameters sweep in parallel over the cluster. To do this, we'll need to tell hyperopt how many runs we want to do in parallel. While its technically possible to run your entire budget in one go, that typically won't yield the best performance outcome, as the algorithm hyperopt uses won't be able to focus its search space as the runs proceed. A decent trade-off and heuristic to use here is the square root of your total evaluation budget. Here, we'll just use a simple budget of 16 evals, which means according to the heuristic we can use parallelism of 4. This means 4 runs will happen in parallel, and hyperopt will have multiple opportunities to improve the search space over those runs.

# COMMAND ----------

# DBTITLE 1,Configure parallelism
import math

from hyperopt import SparkTrials

# Feel free to change max_evals if you want fewer/more trial runs
# note: we're assuming you're using the 16 core cluster created in RunMe
max_evals = 16
parallelism = 4  # e.g., int(math.sqrt(max_evals)) or sc.defaultParallelism

# COMMAND ----------

# MAGIC %md
# MAGIC ## Custom model wrapper
# MAGIC
# MAGIC Since our input data to our model serving endpoint will need some preprocessing applied before we feed it to our scikit-learn pipeline, we need to create a simple wrapper class to apply the same preprocessing as well as the postprocessing to the results. Autologging has already logged the model above as part of hyperparameter tuning, but here we'll log our wrapper model along with the parameters and metrics and this will be the one we'll deploy to the endpoint.

# COMMAND ----------

# Define the custom model wrapper.
class ModelWrapper(mlflow.pyfunc.PythonModel):

    def __init__(self, pipeline, label_encoder):
        self.pipeline = pipeline
        self.label_encoder = label_encoder

    def predict(self, context, model_input, params=None):
        X = transform(model_input.copy(deep=False))
        y = self.pipeline.predict(X)
        return self.label_encoder.inverse_transform(y)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Tuning and training run
# MAGIC
# MAGIC Now that we have the wrapper defined, we run our distributed hyperparameter search and then log it explicitly to the MLflow tracking experiment as well as Unity Catalog as our model registry with a call to `log_model`. Along with the model artifact, we also log the metrics and parameters we used, the signature, and sample model input to help users of the model trace back our lineage and aide reproducibility and understanding.

# COMMAND ----------

# DBTITLE 1,Tune and train the model
spark_trials = SparkTrials(parallelism=parallelism)

with mlflow.start_run(run_name=f"{model_name}_hyperopt_tuning") as run:

    # Find the best set of hyperparameters
    best_params = hyperopt.fmin(
        fn=objective_fn,
        space=search_space,
        algo=hyperopt.tpe.suggest,
        max_evals=max_evals,
        trials=spark_trials)

    params = hyperopt.space_eval(search_space, best_params)

    # Do a final training run with the best parameters
    pipeline.set_params(**params)
    pipeline.fit(X_train_full, y_train_full)
    mlflow.log_params(params)
    y_hat = pipeline.predict(X_test)

    # Overall metrics
    weighted_f1_score = metrics.f1_score(y_test, y_hat, average="weighted")
    accuracy = metrics.accuracy_score(y_test, y_hat)
    mlflow.log_metric("weighted_f1_score", weighted_f1_score)
    mlflow.log_metric("accuracy", accuracy)
    
    # Per class metrics
    cm = metrics.confusion_matrix(y_test, y_hat)
    tp = cm.diagonal()
    fp = cm.sum(axis=0) - tp
    n_classes = len(tp)
    mlflow.log_metrics({f"class_{i}_tp": tp[i] for i in range(n_classes)})
    mlflow.log_metrics({f"class_{i}_fp": fp[i] for i in range(n_classes)})
    mlflow.log_metrics({f"class_{i}_accuracy": tp[i] for i in range(n_classes)})

    # Log the model with pre and post processing logic
    mlflow.pyfunc.log_model(
        python_model=ModelWrapper(pipeline, label_encoder),
        artifact_path="model",
        signature=mlflow.models.infer_signature(X, y),
        input_example=X.sample(3),
        registered_model_name=f"{catalog_name}.{schema_name}.{model_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test the registered model
# MAGIC
# MAGIC Let's load the model back from the registry and make sure we can use it for predictions as a sanity test.

# COMMAND ----------

# DBTITLE 1,Run a sanity test on the model
client = mlflow.MlflowClient()
model_versions = client.search_model_versions(f"name='{catalog_name}.{schema_name}.{model_name}'")
latest_version = str(max(int(v.version) for v in model_versions))
latest_uri = f"models:/{catalog_name}.{schema_name}.{model_name}/{latest_version}"
loaded_model = mlflow.pyfunc.load_model(latest_uri)
sample_model_input = X.sample(3)
sample_model_output = loaded_model.predict(sample_model_input)
display(sample_model_output)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Assign champion alias
# MAGIC
# MAGIC For downstream scoring pipelines, including deployment to a model serving endpoint, we can reference the model by an alias to better communicate which is considered the intended live model.

# COMMAND ----------

# DBTITLE 1,Assign a model alias
client.set_registered_model_alias(model_name, "champion", latest_version)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Conclusion
# MAGIC
# MAGIC Congratulations! You've just created and registered a machine learning model based on product interest loaded from Salesforce Data Cloud to recommend products for users. The techniques we used here are going to be fairly typical so we're hopeful this gives you a good head start in doing something similar with your own use case. However, we're not quite done yet! Even though we have the model deployed to the registry and could apply it from there to batch and streaming workloads, to integrate with Salesforce Data Cloud we need one more step: we need to set up a real-time serving endpoint in Databricks. When you're ready to tackle this step, continue on to the next notebook: [Deploy Serving Endpoint]($./06_deploy_serving_endpoint).
