from torchvision import transforms, io
import requests
from tqdm import tqdm
from timeit import default_timer as timer
import torch
from torch import nn
from torchvision import transforms, datasets
from pathlib import Path
from typing import List, Dict
import matplotlib.pyplot as plt

BATCH_SIZE = 32

data_path = Path('data/')
image_path = data_path / "plant_category"
device = "cuda" if torch.cuda.is_available() else "cpu"

image_path_list = list(image_path.rglob('*/*/*.jpg'))

train_dir = image_path / "train"
test_dir = image_path / "test"


# train_custom_transform = transforms.Compose([
#     transforms.Resize(size=(224, 224)),
#     transforms.TrivialAugmentWide(num_magnitude_bins=31),
#     transforms.ToTensor(),
# ])
train_custom_transform = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    # transforms.RandomResizedCrop(224),
    # transforms.TrivialAugmentWide(num_magnitude_bins=31),
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomVerticalFlip(),
    # transforms.RandomRotation(20),
    # transforms.ColorJitter(brightness=0.2, contrast=0.2,
    #                        saturation=0.2, hue=0.2),
    # transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.5),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
# test_custom_transform = transforms.Compose([
#     transforms.Resize(size=(224, 224)),
#     transforms.ToTensor(),
# ])

train_custom_data = datasets.ImageFolder(
    root=train_dir, transform=train_custom_transform)
test_custom_data = datasets.ImageFolder(
    root=test_dir, transform=train_custom_transform)
train_custom_dataloader = torch.utils.data.DataLoader(
    train_custom_data, batch_size=BATCH_SIZE, shuffle=True)
test_custom_dataloader = torch.utils.data.DataLoader(
    test_custom_data, batch_size=BATCH_SIZE, shuffle=False)

class_names = train_custom_data.classes
print(class_names)


class TinyVGG(nn.Module):

    def __init__(self,
                 input_shape: int,
                 hidden_units: int,
                 output_shape: int):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=2,
                      padding=0),
            nn.ReLU(),

        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=2,
                      padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2)
        )
        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=2,
                      padding=0),
            nn.ReLU(),
        )
        self.conv_block_4 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=2,
                      padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2)
        )  
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*3*3,
                      out_features=output_shape)
        )

    def forward(self, x: torch.TensorType):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.conv_block_3(x)
        x = self.conv_block_4(x)
        # print(x.shape)
        return self.classifier(x)


model_0 = TinyVGG(3, 10, len(class_names)).to(device)


def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device=device):

    model.train()

    train_loss, train_accuracy = 0, 0

    for batch, (X, y) in enumerate(dataloader):

        X, y = X.to(device), y.to(device)
        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate the loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        # Calculate accuracy metric
        y_pred_classes = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_accuracy += ((y_pred_classes == y).sum().item() /
                           len(y_pred_classes))

    train_loss /= len(dataloader)
    train_accuracy /= len(dataloader)

    return train_loss, train_accuracy


def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              optimizer: torch.optim.Optimizer):

    model.eval()

    test_loss, test_accuracy = 0, 0
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):

            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            y_pred = model(X)

            loss = loss_fn(y_pred, y)
            test_loss += loss.item()

            y_pred_label = torch.argmax(y_pred, dim=1)
            test_accuracy += ((y_pred_label == y).sum().item() /
                              len(y_pred_label))

    test_loss /= len(dataloader)
    test_accuracy /= len(dataloader)

    return test_loss, test_accuracy


loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model_0.parameters(), lr=0.001)


def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          loss_fn: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          epochs: int = 3,
          device=device):

    results = {
        "train_loss": [],
        "train_accuracy": [],
        "test_loss": [],
        "test_accuracy": [],
    }

    train_time_start = timer()

    for epoch in tqdm(range(epochs)):
        print(f" Epoch: {epoch}: \n")
        train_loss, train_accuracy = train_step(
            model, train_dataloader, loss_fn, optimizer)
        test_loss, test_accuracy = test_step(
            model, test_dataloader, loss_fn, optimizer)

        results["train_loss"].append(train_loss)
        results["train_accuracy"].append(train_accuracy)
        results["test_loss"].append(test_loss)
        results["test_accuracy"].append(test_accuracy)
        print(
            f"Train loss: {train_loss} | Train accuracy: {train_accuracy} | Test loss: {test_loss} | Test accuracy: {test_accuracy}")

    train_time_end = timer()
    print(f"Total train time: {train_time_end-train_time_start: .3f} seconds")

    return results


train_results = train(model_0, train_custom_dataloader,
                      test_custom_dataloader, loss_fn, optimizer, epochs=30)

MODELS_PATH = Path("models/")
MODELS_PATH.mkdir(parents=True, exist_ok=True)

MODELS_NAME = "plant_category.pth"
MODELS_SAVE_PATH = MODELS_PATH / MODELS_NAME

print(f"Saving model...")
torch.save(obj=model_0, f=MODELS_SAVE_PATH)


def plot_loss_curves(results: Dict[str, List[str]]):

    train_loss = results['train_loss']
    test_loss = results['test_loss']

    train_accuracy = results['train_accuracy']
    test_accuracy = results['test_accuracy']

    epochs = range(len(results['train_loss']))

    plt.figure(figsize=(15, 7))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label="train_loss")
    plt.plot(epochs, test_loss, label="test_loss")
    plt.xlabel("Epochs")
    plt.title("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracy, label="train_accuracy")
    plt.plot(epochs, test_accuracy, label="test_accuracy")
    plt.xlabel("Epochs")
    plt.title("Accuracy")
    plt.legend()

    plt.show()


plot_loss_curves(train_results)
