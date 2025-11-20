from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

ml_client = MLClient(
    subscription_id="0ac7b36f-d0da-40e1-9e2a-3644bc3c6d6f",
    resource_group_name="dips-ml-rg",
    workspace_name="dips-ml-workspace",
    credential=DefaultAzureCredential(),
)

# List all environments
for env in ml_client.environments.list():
    print(f"Name: {env.name}, Latest: {env.latest_version}")