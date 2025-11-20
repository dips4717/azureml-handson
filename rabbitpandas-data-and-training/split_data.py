import os
import argparse
from sklearn.model_selection import train_test_split
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
import fsspec

def main(dataasset_name, output_dir, val_split_size, random_seed):
    # 1. Connect to Workspace
    credential = DefaultAzureCredential()
    ml_client = MLClient.from_config(credential=credential)
    print("Connected to ML Workspace.")
    
    # 2. Get Data Asset URI first
    try:
        data_asset = ml_client.data.get(name=dataasset_name, label="latest")
        data_uri = data_asset.path
        print(f"Data Asset URI: {data_uri}")
    except Exception as e:
        print(f"Error retrieving Data Asset: {e}")
        return

    # 3. Initialize FileSystem with the URI
    # FIX: We create the fs object HERE, passing the uri we just retrieved
    try:
        fs = fsspec.filesystem("azureml", uri=data_uri, credential=credential)
    except Exception as e:
        print(f"Error initializing file system: {e}")
        return
    
    print(f"Scanning directory...")
    
    image_files = []
    labels_map = {}
    label_counter = 0

    # 4. Walk through the input directory using fs.walk
    # fs.walk works exactly like os.walk but over the network
    for root, dirs, files in fs.walk(data_uri):
        if not dirs: # Leaf directory
            class_name = root.split('/')[-1]
            
            if class_name not in labels_map:
                labels_map[class_name] = label_counter
                label_counter += 1
            
            class_label = labels_map[class_name]
            print(f"  Found class: {class_name} (ID: {class_label})")

            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    # Construct full path
                    full_path = f"{root}/{file}"
                    
                    # Calculate relative path for the text file
                    # Ensure prefix ends with slash to remove it cleanly
                    prefix = data_uri if data_uri.endswith('/') else data_uri + '/'
                    relative_path = full_path.replace(prefix, "")
                    
                    image_files.append((relative_path, class_label))
    
    print(f"\nFound {len(image_files)} total images in {label_counter} classes.")
    print(f"Class mapping: {labels_map}")

    if len(image_files) == 0:
        print("Error: No images found. Check if the Data Asset path is correct.")
        return

    # 5. Split the data
    train_files, val_files = train_test_split(
        image_files, 
        test_size=val_split_size, 
        random_state=random_seed,
        stratify=[label for _, label in image_files]
    )

    # 6. Save results locally
    os.makedirs(output_dir, exist_ok=True)
    
    def write_split_file(filename, file_list):
        output_path = os.path.join(output_dir, filename)
        with open(output_path, 'w') as f:
            for img_path, label in file_list:
                f.write(f"{img_path} {label}\n")
        print(f"Successfully wrote {len(file_list)} entries to {output_path}")

    write_split_file('train.txt', train_files)
    write_split_file('val.txt', val_files)

    print(f"\nSplit files created successfully in local folder: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataasset_name', type=str, default="rabbit-panda-dataset1", help='Name of the registered Data Asset')
    parser.add_argument('--output_dir', type=str, default="outputs/", help='Local directory to save the .txt files')
    parser.add_argument('--val_split_size', type=float, default=0.2, help='Fraction of data for validation')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed for splitting')
    
    args = parser.parse_args()
    
    main(args.dataasset_name, args.output_dir, args.val_split_size, args.random_seed)