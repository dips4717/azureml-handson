# authentication package
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential
from azure.ai.ml.entities import AmlCompute
from azure.ai.ml import command
from azure.ai.ml import Input


try:
    credential = DefaultAzureCredential()
    # Check if given credential can get token successfully.
    credential.get_token("https://management.azure.com/.default")
except Exception as ex:
    # Fall back to InteractiveBrowserCredential in case DefaultAzureCredential not work
    credential = InteractiveBrowserCredential()


# handle to the workspace
from azure.ai.ml import MLClient

# get a handle to the workspace
ml_client = MLClient(
    subscription_id="0ac7b36f-d0da-40e1-9e2a-3644bc3c6d6f",
    resource_group_name="dips-ml-rg",
    workspace_name="dips-ml-workspace",
    credential=credential,
)

cpu_compute_target = "cpu-cluster"
try:
    # let's see if the compute target already exists
    cpu_cluster = ml_client.compute.get(cpu_compute_target)
    print(
        f"You already have a cluster named {cpu_compute_target}, we'll reuse it as is."
    )

except Exception:
    print("Creating a new cpu compute target...")

    # Let's create the Azure ML compute object with the intended parameters
    cpu_cluster = AmlCompute(
        # Name assigned to the compute cluster
        name="cpu-cluster",
        # Azure ML Compute is the on-demand VM service
        type="amlcompute",
        # VM Family
        size="STANDARD_DS3_V2",
        # Minimum running nodes when there is no job running
        min_instances=0,
        # Nodes in cluster
        max_instances=4,
        # How many seconds will the node running after the job termination
        idle_time_before_scale_down=180,
        # Dedicated or LowPriority. The latter is cheaper but there is a chance of job termination
        tier="Dedicated",
    )
    # Now, we pass the object to MLClient's create_or_update method
    cpu_cluster = ml_client.begin_create_or_update(cpu_cluster).result()

print(
    f"AMLCompute with name {cpu_cluster.name} is created, the compute size is {cpu_cluster.size}"
)

## --- 2. Create the Job ---
training_job = command(
    # local path where the code is stored
    code=".",  # Current directory contains train_custom.py
    # describe the command to run the python script, with all its parameters
    # use the syntax below to inject parameter values from code
    command="""python train_custom.py \
        --data_dir ${{inputs.data_dir}} \
        --epochs ${{inputs.epochs}} \
        --learning_rate ${{inputs.learning_rate}}
    """,
    inputs={
        "data_dir": Input(
            type="uri_folder",
            path="azureml://subscriptions/0ac7b36f-d0da-40e1-9e2a-3644bc3c6d6f/resourcegroups/dips-ml-rg/workspaces/dips-ml-workspace/datastores/rabbitpandas_datastore/paths/",
            mode="download",  # use download to make access faster, mount if dataset is larger than VM
        ),
        "epochs": 1,
        "learning_rate": 0.001,
    },
    environment="azureml://registries/azureml/environments/acpt-pytorch-2.2-cuda12.1/versions/43",
    compute="cpu-cluster",
    instance_count=1,
    display_name="using-python-sdk-to-submit-job",
    description="using-python-sdk-to-submit-job for training a torchvision model on CPU (no quota needed)",
)

# --- 3. Submit the Job ---
# submit the job
returned_job = ml_client.jobs.create_or_update(
    training_job,
    # Project's name
    experiment_name="panda-rabbit",
)

# get a URL for the status of the job
print("The url to see your live job running is returned by the sdk:")
print(returned_job.studio_url)
