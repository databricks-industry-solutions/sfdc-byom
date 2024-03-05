## Install Databricks CLI and set up secrets

1. Use the steps in the  link to install the Databricks CLI on your machine - https://docs.databricks.com/en/dev-tools/cli/install.html
2.  Follow these instructions to set up authentication between the Databricks CLI and your Databricks accounts and workspaces - https://docs.databricks.com/en/dev-tools/cli/authentication.html
3. To establish a Salesforce connection and configure connection secrets from the command line interface (CLI), execute the following commands. Keep in mind that "sfdc—byom-scope" is the scope's name, and you can assign any relevant name of your choice.

```
databricks secrets create-scope sfdc—byom-scope
databricks secrets put-secret sfdc—byom-scope sfdc-byom-cdpcrma-password —string-value <<your salesforce login password>>
databricks secrets put-secret sfdc—byom-scope sfdc-byom-cdpcrma-client-id —string-value <<your salesforce connected app client id>>
databricks secrets put-secret sfdc—byom-scope sfdc-byom-cdpcrma-client-secret —string-value <<your salesforce connected app client id>>
```