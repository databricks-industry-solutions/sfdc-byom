## Upload training data set into Datacloud

A synthetic dataset comprising 1 million rows was generated for this purpose, encompassing the following attributes, 

- **Club Member:** Indicates whether the customer is a club member.
- **Campaign:** Represents the campaign the customer is associated with.
- **State:** Denotes the state where the customer resides.
- **Month:** Indicates the month of purchase.
- **Case Count:** The number of cases raised by the customer.
- **Case Type Return:** Specifies whether the customer returned any product in the last year.
- **Case Type Shipment Damaged:** Indicates whether the customer experienced any shipment damage in the last year.
- **Engagement Score:** Reflects the level of customer engagement, including responses to mailing campaigns, logins to the online platform, etc.
- **Tenure:** This represents the number of years the customer has been part of NT.
- **Clicks:** The average number of clicks the customer made within one week before purchase.
- **Pages Visited:** The average number of page visits the customer made within one week before purchase.
- **Product Purchased:** Specifies the product purchased by the customer.

In a real-life scenario, DataCloud can be utilized to ingest data from various sources, employing powerful batch and streaming transformational capabilities to create a robust dataset for model training.

The dataset can be accessed here.  Afterward, you have the option to upload the CSV file to an S3 bucket.

Here are the steps to create Data Streams from S3 in Salesforce:


Log in to the org

![image](files/sfdc_byom/images/create_data_stream_01.png)


Navigate to "Data Streams" and click "New"

![image](files/sfdc_byom/images/create_data_stream_02.png)


Select "Amazon S3" and click on Next

![image](files/sfdc_byom/images/create_data_stream_03.png)


Enter S3 bucket and file details

![image](files/sfdc_byom/images/create_data_stream_04.png)


Click Next

![image](files/sfdc_byom/images/create_data_stream_05.png)


Click Next

![image](files/sfdc_byom/images/create_data_stream_06.png)


Click on Full Refresh

![image](files/sfdc_byom/images/create_data_stream_07.png)


Select Frequency = "None"

![image](files/sfdc_byom/images/create_data_stream_08.png)


Click Deploy to create data stream

![image](files/sfdc_byom/images/create_data_stream_09.png)
