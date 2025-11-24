import argparse
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import mlflow
import mlflow.pytorch

# --- 1. Custom Dataset Definition ---
class ImageTxtDataset(Dataset):
    def __init__(self, root_dir, txt_file_path, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images (e.g., /inputs/training_data).
            txt_file_path (string): Path to the txt file with image paths and labels.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_list = []
        
        # Read the text file
        with open(txt_file_path, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if line:
                    img_path, label = line.split(' ')
                    self.image_list.append((img_path, int(label)))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        # Get image path and label
        img_relative_path, label = self.image_list[idx]
        
        # Construct full image path
        img_full_path = os.path.join(self.root_dir, img_relative_path)
        
        # Open image
        try:
            image = Image.open(img_full_path).convert("RGB")
        except FileNotFoundError:
            print(f"Error: File not found at {img_full_path}")
            return None, None
        except Exception as e:
            print(f"Error loading image {img_full_path}: {e}")
            return None, None

        # Apply transforms
        if self.transform:
            image = self.transform(image)
            
        return image, label

# --- 2. Define the CNN Model ---
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        self.model = models.resnet18(weights='IMAGENET1K_V1')
        for param in self.model.parameters():
            param.requires_grad = False
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes) # num_classes based on your data

    def forward(self, x):
        return self.model(x)

# --- 3. Define main training function ---
def main(data_dir, epochs, learning_rate):
    
    mlflow.start_run()
    mlflow.log_param("epochs", epochs)
    mlflow.log_param("learning_rate", learning_rate)

    # --- Data Loading ---
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Use the custom Dataset
    # data_dir is the mounted path, e.g., /inputs/training_data
    # The script looks for train.txt and val.txt *inside* this path
    train_txt = os.path.join(data_dir, 'annotations/train.txt')
    val_txt = os.path.join(data_dir, 'annotations/val.txt')

    print(f"Loading training data from: {train_txt}")
    print(f"Loading validation data from: {val_txt}")

    image_datasets = {
        'train': ImageTxtDataset(root_dir=data_dir, txt_file_path=train_txt, transform=data_transform),
        'validation': ImageTxtDataset(root_dir=data_dir, txt_file_path=val_txt, transform=data_transform)
    }
    
    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=16, shuffle=True, num_workers=4),
        'validation': DataLoader(image_datasets['validation'], batch_size=16, shuffle=False, num_workers=4)
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # --- Model, Loss, Optimizer ---
    # Assuming 2 classes (panda, rabbit)
    model = SimpleCNN(num_classes=2).to(device) 
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.model.fc.parameters(), lr=learning_rate)

    # --- Training Loop (Identical to before) ---
    print("Starting Training...")
    for epoch in range(epochs):
        for phase in ['train', 'validation']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                # Handle potential None from failed image loads
                if inputs is None or labels is None:
                    continue
                
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_acc = running_corrects.double() / len(image_datasets[phase])
            
            print(f"Epoch {epoch}/{epochs-1} | {phase} Loss: {epoch_loss:.4f} | {phase} Acc: {epoch_acc:.4f}")
            
            # Log metrics
            if phase == 'train':
                mlflow.log_metric("train_loss", epoch_loss, step=epoch)
                mlflow.log_metric("train_accuracy", epoch_acc.item(), step=epoch)
            else:
                mlflow.log_metric("val_loss", epoch_loss, step=epoch)
                mlflow.log_metric("val_accuracy", epoch_acc.item(), step=epoch)

    # --- Save Model ---
    os.makedirs("outputs", exist_ok=True)
    model_path = os.path.join("outputs", "model.pth")
    torch.save(model.state_dict(), model_path)
    mlflow.pytorch.log_model(model, "pytorch-model")
    
    mlflow.end_run()
    print("Training complete. Model saved to 'outputs/model.pth'")

# --- 4. Argument Parsing ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, help='Directory for the training data')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--local_debug', type=lambda x: str(x).lower() in ["1", "yes", "true"],  default=0, help='Flag for local debugging')
    args = parser.parse_args()

    if args.local_debug:
        # Obtaine acess to the workspace
        
        from azure.identity import DefaultAzureCredential
        from azure.ai.ml import MLClient

        credential = DefaultAzureCredential() 
        # get a handle to the workspace
        try:
            credential = DefaultAzureCredential()
            ml_client = MLClient.from_config(credential=credential)
        except Exception:
            ml_client = MLClient(
                subscription_id="0ac7b36f-d0da-40e1-9e2a-3644bc3c6d6f",
                resource_group_name="dips-ml-rg",
                workspace_name="dips-ml-workspace",
                credential=credential,
            )
        print(f"Connected to workspace: {ml_client.workspace_name}")
        data_asset_name = "rabbit-panda-fulldataset"
        data_asset = ml_client.data.get(name=data_asset_name, label="latest")

        # This is the base azureml:// path
        data_asset_path = data_asset.path.rstrip('/')
        args.data_dir = data_asset_path

    pass
    main(args.data_dir, args.epochs, args.learning_rate)