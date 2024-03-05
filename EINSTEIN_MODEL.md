## Set up model in Einstein Studio


### 1. Log in to the org

![image](files/sfdc_byom/images/deploy_model_01.png)


### 2. Navigate to ML Workspace / Einstein Studio

![image](files/sfdc_byom/images/deploy_model_02.png)


### 3. Select ML Workspace

![image](files/sfdc_byom/images/deploy_model_03.png)


### 4. Click New

You should see a toast message that the end point was saved successfully

![image](files/sfdc_byom/images/deploy_model_04.png)


### 5. Give your model a name and click create

![image](files/sfdc_byom/images/deploy_model_05.png)


### 6. Select Endpoint

![image](files/sfdc_byom/images/deploy_model_06.png)


### 7. Click on add endpoint

![image](files/sfdc_byom/images/deploy_model_07.png)


### 8. Enter inference url from databrisck as well as request format as dataframe split

![image](files/sfdc_byom/images/deploy_model_08.png)


### 9. Select Authentication type, Auth Header= "Authorization"

![image](files/sfdc_byom/images/deploy_model_09.png)


### 10. Secret Key = "Bearer <<your personal access token from databricks>>"

![image](files/sfdc_byom/images/deploy_model_10.png)


### 11. Click Save. 

You should see a toast message that the end point was saved successfully

![image](files/sfdc_byom/images/deploy_model_11.png)


### 12. Select input features

![image](files/sfdc_byom/images/deploy_model_12.png)


### 13. Click on Add input features

![image](files/sfdc_byom/images/deploy_model_13.png)


### 14. Choose the DMO 

Choose the DMO that has all the fields for model scoring in this case it is account contact DMO.

![image](files/sfdc_byom/images/deploy_model_14.png)


### 15. Click Save

![image](files/sfdc_byom/images/deploy_model_15.png)


### 16. Select fields from DMO for scoring

Now start selecting the fields from the DMO for model scoring. Note that the feature API name of the field selected should match  the names the model is expecting for instance as shown in the query endpoint dialog above 

![image](files/sfdc_byom/images/deploy_model_16.png)


### 17. Drag each predictor and click done one by one in the specific order

![image](files/sfdc_byom/images/deploy_model_17.png)


### 18. Once you enter all the predictors in the click on save

![image](files/sfdc_byom/images/deploy_model_18.png)


### 19. Next go to output Predictions

![image](files/sfdc_byom/images/deploy_model_19.png)


### 20. Give the DMO a name.

This is where the output predictions will be saved

![image](files/sfdc_byom/images/deploy_model_20.png)


### 21. Click save

![image](files/sfdc_byom/images/deploy_model_21.png)


### 22. Enter the outcome variable API name and the json key

Note that in this case the json key is - $.predictions

![image](files/sfdc_byom/images/deploy_model_22.png)


### 23. Click Save

![image](files/sfdc_byom/images/deploy_model_23.png)


### 24. Now activate the model

![image](files/sfdc_byom/images/deploy_model_24.png)


### 25. Once model is activated refresh it to see the predictions in the DMO

![image](files/sfdc_byom/images/deploy_model_25.png)


