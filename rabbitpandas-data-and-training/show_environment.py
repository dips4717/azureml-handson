from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

subscription_id = "0ac7b36f-d0da-40e1-9e2a-3644bc3c6d6f"
resource_group = "dips-ml-rg"
workspace = "dips-ml-workspace"

def main():
    credential = DefaultAzureCredential()
    ml_client = MLClient(
        subscription_id=subscription_id,
        resource_group_name=resource_group,
        workspace_name=workspace,
        credential=credential,
    )

    name = "AzureML-pytorch-1.10-ubuntu18.04-py38-cuda11-gpu"
    version = "latest"

    try:
        env = ml_client.environments.get(name=name, version=version)
    except Exception as e:
        print(f"Failed to get environment {name}@{version}: {e}")
        return

    print(f"Environment: {env.name}@{env.version}")
    # Common places for image info
    image = getattr(env, "image", None)
    docker = getattr(env, "docker", None)
    conda_file = getattr(env, "conda_file", None)

    print("image:", image)
    if docker:
        base_image = getattr(docker, "base_image", None)
        print("docker.base_image:", base_image)
        # Print docker dict if available
        try:
            print("docker:", docker.__dict__)
        except Exception:
            print("docker (raw):", docker)

    print("conda_file:", conda_file)

    # Fallback: print repr/dir for more debugging
    print("--- full repr ---")
    try:
        print(env)
    except Exception:
        print(repr(env))

    print("--- dir(env) ---")
    print([x for x in dir(env) if not x.startswith("__")])


if __name__ == '__main__':
    main()
