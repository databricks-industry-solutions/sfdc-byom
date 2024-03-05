## Set up a connected app in Salesforce

To be able to access SalesForce Data Cloud via the connector, you'll first need a connected app in Salesforce. Here's how to create that.

### 1. Go to Salesforce setup

Log in to Salesforce and go to setup

![image](https://github.com/databricks-industry-solutions/sfdc-byom/raw/main/images/connected_app_01.png)


### 2. Open up App Manager

Search for App Manager

![image](files/sfdc_byom/images/connected_app_02.png)

When you open it, it should look like this

![image](files/sfdc_byom/images/connected_app_03.png)


### 3. Create Connected App

Click on New Connected App

![image](files/sfdc_byom/images/connected_app_04.png)

1. Give the app a name
2. Enter email
3. Check Enable OAuth settings
4. Put "https://login.salesforce.com/services/oauth2/success" in the callback url
5. Ensure that the following are selected in the scopes
    1. Manage user data via APIs (api)
    2. Access all datacloud resources
    3. Perform ANSI SQL queries on DataCloud

![image](files/sfdc_byom/images/connected_app_05.png)

Click on Save.

![image](files/sfdc_byom/images/connected_app_06.png)


### 4. Update policies

In set up go to Manage Connected App

![image](files/sfdc_byom/images/connected_app_07.png)

Click on the newly created connected app and then click on Edit Policies.

![image](files/sfdc_byom/images/connected_app_08.png)

Make sure that under oauth policies we have "Relax IP restrictions" and "Allow all users to self authorize" and then click Save.

![image](files/sfdc_byom/images/connected_app_09.png)


### 5. Set up customer keys (optional)

Click on Manage Customer Keys and provide validation code if applicable.

![image](files/sfdc_byom/images/connected_app_10.png)

Copy the keys.

![image](files/sfdc_byom/images/connected_app_11.png)


%md
### 6. Ensure Oauth and OpenId are enabled

In setup, go to Oauth and OpenId settings. Ensure all the options are turned on.

![image](files/sfdc_byom/images/connected_app_12.png)

**Note:** If you want to restrict IP's, you can set it up in the connected app. See the article [Restrict Access to Trusted IP Ranges for a Connected App](https://help.salesforce.com/s/articleView?id=sf.connected_app_edit_ip_ranges.htm&type=5) for more details.


