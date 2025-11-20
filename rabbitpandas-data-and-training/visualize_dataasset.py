#%%
import matplotlib.pyplot as plt
from PIL import Image
import fsspec
import random

from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

# --- 0. (Optional) Update libraries ---
# You may want to run this in a cell first to ensure 
# all components are on the latest compatible versions.
# %pip install --upgrade azure-ai-ml azure-identity azureml-fsspec

# --- 1. Connect to Workspace ---
try:
    credential = DefaultAzureCredential()
    ml_client = MLClient.from_config(credential=credential)
except Exception:
    # Fallback if config.json is not present
    ml_client = MLClient(
        credential,
        subscription_id="YOUR_SUBSCRIPTION_ID",
        resource_group_name="YOUR_RESOURCE_GROUP",
        workspace_name="YOUR_ML_WORKSPACE_NAME",
    )

print(f"Connected to workspace: {ml_client.workspace_name}")


#%%
# --- 2. Get the Data Asset ---
data_asset_name = "rabbit-panda-fulldataset"
data_asset = ml_client.data.get(name=data_asset_name, label="latest")

# This is the base azureml:// path
data_asset_path = data_asset.path.rstrip('/') # Remove trailing slash if present
print(f"Data Asset azureml:// path: {data_asset_path}")

#%%
# --- 3. Read the val.txt file to find images ---
# Construct the full path to the validation file
val_file_path = f"{data_asset_path}/annotations/val.txt"
print(f"Reading validation file: {val_file_path}")


# Define the storage options, which include authentication
# fsspec will pass this to the AzureML filesystem
storage_options = {"auth": credential}

try:
    with fsspec.open(val_file_path, mode='r', **storage_options) as f:
        val_lines = f.readlines()
except Exception as e:
    print(f"Error reading file. Make sure 'val.txt' exists at the root of your Data Asset.")
    print(f"Error details: {e}")
    # Raise the error to stop execution if file not found
    raise
#%%
# --- 4. Pick a few random images to display ---
plt.figure(figsize=(15, 5))
print(f"Found {len(val_lines)} validation images. Plotting 3 random samples...")

for i in range(3):
    # Get a random line
    line = random.choice(val_lines).strip()
    img_relative_path, label = line.split(' ')
    
    # Construct the full image path
    img_full_path = f"{data_asset_path}/{img_relative_path}"
    
    print(f"Opening image: {img_full_path}")
    
    # Use fsspec.open() again, this time in read-binary ('rb') mode
    with fsspec.open(img_full_path, mode='rb', **storage_options) as img_file:
        img = Image.open(img_file).convert("RGB")
        
    # Display the image
    plt.subplot(1, 3, i + 1)
    plt.imshow(img)
    plt.title(f"Class: {'panda' if label == '0' else 'rabbit'}\n{img_relative_path.split('/')[-1]}")
    plt.axis('off')
    
plt.show()
# %%
