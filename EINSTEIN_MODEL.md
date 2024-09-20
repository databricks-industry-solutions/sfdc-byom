## Set up model in Einstein Studio


### 1. Log in to the org

![image](https://github.com/coreyabs-db/sfdc-byom-images/raw/main/images/deploy_model_01.png)


### 2. Navigate to ML Workspace / Einstein Studio

![image](https://github.com/coreyabs-db/sfdc-byom-images/raw/main/images/deploy_model_02.png)


### 3. Select ML Workspace

![image](https://github.com/coreyabs-db/sfdc-byom-images/raw/main/images/deploy_model_03.png)


### 4. Click New

You should see a toast message that the end point was saved successfully

![image](https://github.com/coreyabs-db/sfdc-byom-images/raw/main/images/deploy_model_04.png)


### 5. Give your model a name and click create

![image](https://github.com/coreyabs-db/sfdc-byom-images/raw/main/images/deploy_model_05.png)


### 6. Select Endpoint

![image](https://github.com/coreyabs-db/sfdc-byom-images/raw/main/images/deploy_model_06.png)


### 7. Click on add endpoint

![image](https://github.com/coreyabs-db/sfdc-byom-images/raw/main/images/deploy_model_07.png)


### 8. Enter inference url from Databricks as well as request format as dataframe split

![image](https://github.com/coreyabs-db/sfdc-byom-images/raw/main/images/deploy_model_08.png)


### 9. Select Authentication type, Auth Header= "Authorization"

![image](https://github.com/coreyabs-db/sfdc-byom-images/raw/main/images/deploy_model_09.png)


### 10. Secret Key = "Bearer <<your personal access token from Databricks>>"

![image](https://github.com/coreyabs-db/sfdc-byom-images/raw/main/images/deploy_model_10.png)


### 11. Click Save. 

You should see a toast message that the end point was saved successfully

![image](https://github.com/coreyabs-db/sfdc-byom-images/raw/main/images/deploy_model_11.png)


### 12. Select input features

![image](https://github.com/coreyabs-db/sfdc-byom-images/raw/main/images/deploy_model_12.png)


### 13. Click on Add input features

![image](https://github.com/coreyabs-db/sfdc-byom-images/raw/main/images/deploy_model_13.png)


### 14. Choose the DMO 

Choose the DMO that has all the fields for model scoring in this case it is account contact DMO.

![image](https://github.com/coreyabs-db/sfdc-byom-images/raw/main/images/deploy_model_14.png)


### 15. Click Save

![image](https://github.com/coreyabs-db/sfdc-byom-images/raw/main/images/deploy_model_15.png)


### 16. Select fields from DMO for scoring

Now start selecting the fields from the DMO for model scoring. Note that the feature API name of the field selected should match  the names the model is expecting for instance as shown in the query endpoint dialog above 

![image](https://github.com/coreyabs-db/sfdc-byom-images/raw/main/images/deploy_model_16.png)


### 17. Drag each predictor and click done one by one in the specific order

![image](https://github.com/coreyabs-db/sfdc-byom-images/raw/main/images/deploy_model_17.png)


### 18. Once you enter all the predictors in the click on save

![image](https://github.com/coreyabs-db/sfdc-byom-images/raw/main/images/deploy_model_18.png)


### 19. Next go to output Predictions

![image](https://github.com/coreyabs-db/sfdc-byom-images/raw/main/images/deploy_model_19.png)


### 20. Give the DMO a name.

This is where the output predictions will be saved

![image](https://github.com/coreyabs-db/sfdc-byom-images/raw/main/images/deploy_model_20.png)


### 21. Click save

![image](https://github.com/coreyabs-db/sfdc-byom-images/raw/main/images/deploy_model_21.png)


### 22. Enter the outcome variable API name and the json key

Note that in this case the json key is - $.predictions

![image](https://github.com/coreyabs-db/sfdc-byom-images/raw/main/images/deploy_model_22.png)


### 23. Click Save

![image](https://github.com/coreyabs-db/sfdc-byom-images/raw/main/images/deploy_model_23.png)


### 24. Now activate the model

![image](https://github.com/coreyabs-db/sfdc-byom-images/raw/main/images/deploy_model_24.png)


### 25. Once model is activated refresh it to see the predictions in the DMO

![image](https://github.com/coreyabs-db/sfdc-byom-images/raw/main/images/deploy_model_25.png)


