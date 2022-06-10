import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

#
# def train(dataloader, model, loss_fn, optimizer):
#     size = len(dataloader.dataset)
#     model.train()
#     for batch, (X, y) in enumerate(dataloader):
#         X, y = X.to(device), y.to(device)
#         prediction = model(X)
#         loss = loss_fn(prediction, y)
#
#         # Back propagation
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         if batch % 100 == 0:
#             loss, current = loss.item(), batch * len(X)
#             print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
#
#
# def test(dataloader, model, loss_fn):
#     size = len(dataloader.dataset)
#     num_batches = len(dataloader)
#     model.eval()
#     miss, hit = 0, 0
#     with torch.no_grad():
#         for X, y in dataloader:
#             X, y = X.to(device), y.to(device)
#             prediction = model(X)
#             miss += loss_fn(prediction, y).item()
#             hit += (prediction.argmax(1) == y).type(torch.float).sum().item()
#     miss /= num_batches
#     hit /= size
#     print(f"Test Error: \n Accuracy: {(100 * hit):>0.1f}%, Avg loss: {miss:>8f} \n")
#
#
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


if __name__ == '__main__':
    training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=False,
        transform=ToTensor(),
    )

    testing_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=False,
        transform=ToTensor(),
    )
    #
    # batch_size = 64
    # train_data_loader = DataLoader(training_data, batch_size=batch_size)
    # test_data_loader = DataLoader(testing_data, batch_size=batch_size)
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # model = NeuralNetwork().to(device)
    # loss_fn = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    #
    # epochs = 5
    # for t in range(epochs):
    #     print(f"Epoch {t + 1}\n-------------------------------")
    #     train(train_data_loader, model, loss_fn, optimizer)
    #     test(test_data_loader, model, loss_fn)
    # print("Done!")
    # torch.save(model.state_dict(), "model.pth")

    model = NeuralNetwork()
    model.load_state_dict(torch.load("model.pth"))
    classes = [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ]


    model.eval()
    x, y = testing_data[44][0], testing_data[44][1]
    with torch.no_grad():
        prediction = model(x)
        predicted, actual = classes[prediction[0].argmax(0)], classes[y]
        print(f'Predicted: "{predicted}", Actual: "{actual}"')
