import torch
from tqdm import tqdm
from timeit import default_timer as timer

def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device="cpu"):
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
              optimizer: torch.optim.Optimizer,
              device="cpu"):

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


def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          loss_fn: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          epochs: int = 3,
          device="cpu"):

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
            model, train_dataloader, loss_fn, optimizer, device)
        test_loss, test_accuracy = test_step(
            model, test_dataloader, loss_fn, optimizer, device)

        results["train_loss"].append(train_loss)
        results["train_accuracy"].append(train_accuracy)
        results["test_loss"].append(test_loss)
        results["test_accuracy"].append(test_accuracy)
        print(
            f"Train loss: {train_loss} | Train accuracy: {train_accuracy} | Test loss: {test_loss} | Test accuracy: {test_accuracy}")

    train_time_end = timer()
    print(f"Total train time: {train_time_end-train_time_start: .3f} seconds")

    return results