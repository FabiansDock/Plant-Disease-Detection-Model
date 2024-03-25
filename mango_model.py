from pathlib import Path
from utilities.architecture import TinyVGG
from utilities.plot_loss_curves import plot_loss_curves
from utilities.train import train
from utilities.save_model import save_model
import torch
from torch import nn
from torchvision import transforms, datasets

BATCH_SIZE = 32

data_path = Path('data/')
image_path = data_path / "grape_mango_tomato"
device = "cuda" if torch.cuda.is_available() else "cpu"

image_path_list = list(image_path.rglob('*/*/*.jpg'))

train_dir = image_path / "train" / "mango"
test_dir = image_path / "test" / "mango"

train_custom_transform = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    transforms.ToTensor(),
]) 

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

model_0 = TinyVGG(3, 10, len(class_names)).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model_0.parameters(), lr=0.001)


train_results = train(model_0, train_custom_dataloader,
                      test_custom_dataloader, loss_fn, optimizer, epochs=40, device=device)

save_model(model_0, name="mango_disease_category")

plot_loss_curves(train_results)
