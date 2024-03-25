from pathlib import Path
import torch

def save_model(model: torch.nn.Module, 
               name: str):
    
    MODELS_PATH = Path("models/")
    MODELS_PATH.mkdir(parents=True, exist_ok=True)

    MODELS_NAME = f"{name}"+".pth"
    MODELS_SAVE_PATH = MODELS_PATH / MODELS_NAME

    print(f"Saving model...")
    torch.save(obj=model, f=MODELS_SAVE_PATH)