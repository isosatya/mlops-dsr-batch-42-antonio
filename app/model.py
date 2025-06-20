import wandb
import torch
import os
from torchvision.models import ResNet, resnet18
from torchvision.transforms import v2 as transforms
from loadotenv import load_env
import torch.nn as nn
from pathlib import Path


MODELS_DIR="models"
os.makedirs(MODELS_DIR, exist_ok=True)
MODEL_FILE_NAME = "best_model.pth"

load_env()

wandb_api_key = os.environ.get("WANDB_API_KEY")
model_path = os.environ.get("MODEL_PATH")

if wandb_api_key:
    wandb.login(key=wandb_api_key)
    print("logged in to wandb")

print("model path", model_path)

def download_artifact():
    artifact = wandb.Api().artifact(model_path, type="model")
    artifact.download(MODELS_DIR)

def get_raw_model() -> ResNet:
    """we create a resnet model with the same architecture as the one we trained"""
    architecture = resnet18(weights=None)   # we create model without pretrained weights
                                            # we do not care about them because we use our tuning weights
    architecture.fc = nn.Sequential(
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 6)
    )

    return architecture                                     

# this will download the model to the "models" directory, you should see best_model.pth inside of it 
def load_model() -> ResNet:
    download_artifact()
    model = get_raw_model()
    
    # get the trained model weights from the models directory
    mode_state_dict_path = Path(MODELS_DIR) / MODEL_FILE_NAME
    model_state_dict = torch.load(os.path.join(MODELS_DIR, MODEL_FILE_NAME),
                                  map_location=torch.device("cpu"))
    model.load_state_dict(model_state_dict, strict=True)
    model.eval()        # set the model to eval mode 
    
    # now we have the model with the trained weights
    return model

def load_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),        # size of the images we used for training
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.224, 0.224, 0.224])
    ])

