from typing import List
from pathlib import Path
import requests
import torch
from torchvision import transforms, io
from torch import nn
import matplotlib.pyplot as plt
from utilities.architecture import TinyVGG

class_names = ['mango', 'tomato']
device = "cuda" if torch.cuda.is_available() else "cpu"
MODELS_PATH = Path("models/")

MODELS_NAME = "plant_category.pth"
MODELS_SAVE_PATH = MODELS_PATH / MODELS_NAME

instance_model = torch.load(f=MODELS_SAVE_PATH)
# input_mango_bacterial_canker.jpeg
input_image_path = Path("input.jpeg")

instance_model.eval()


def custom_image_predictor(model: nn.Module,
                           image_path: str,
                           class_names: List[str] = None,
                           transform=None,
                           device=device,
                           ):

    # if not input_image_path.is_file():
    #     with open(input_image_path, "wb") as f:
    #         print("Downloading image...")
    #         response = requests.get('')
    #         f.write(response.content)
    # else:
    #     print("image already exists")
    # print("Done !")

    image_transform = transforms.Compose([
        transforms.Resize(size=(224, 224))
    ])

    input_image_tensor_uint8 = io.read_image(image_path) / 255
    input_image_tensor = input_image_tensor_uint8.type(torch.float32)
    input_image_tensor_transformed = image_transform(
        input_image_tensor).to(device)

    with torch.no_grad():
        # unsqueeze is done to bring in a batch size. In this case its 1.
        input_image_pred = model(input_image_tensor_transformed.unsqueeze(0))

    prediction_probabilities = torch.softmax(input_image_pred, dim=1)
    prediction_label = torch.argmax(
        prediction_probabilities, dim=1).cpu()  # **

    # Plotting our prediction
    plt.imshow(input_image_tensor_transformed.permute(1, 2, 0).cpu())
    if class_names:
        plt.title(
            f'Prediction: {class_names[prediction_label]} | Probability: {prediction_probabilities.max().cpu()}')
    plt.axis(False)
    plt.show()


custom_image_predictor(instance_model, str(input_image_path), class_names)
