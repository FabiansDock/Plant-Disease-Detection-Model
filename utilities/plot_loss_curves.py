from typing import Dict, List
import matplotlib.pyplot as plt

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