import json
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import seaborn as sns
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
import math
import copy


def load_data(data_path):
    """Loads training dataset from json file.
        :param data_path (str): Path to json file containing data
        :return X (ndarray): Inputs
        :return y (ndarray): Targets
    """

    with open(data_path, "r") as fp:
        data = json.load(fp)

    # convert lists to numpy arrays
    X = np.array(data["mfcc"])
    y = np.array(data["labels"])

    print("Data succesfully loaded!")

    return X, y


class CNNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(

            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Flatten(),
            nn.Linear(4096, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 16)
        )

    def forward(self, xb):
        return self.network(xb)


class AddIterator:

    def __init__(self, X, y, batch_size=32):
        self.batch_size = batch_size
        self.n_batches = int(math.ceil(X.shape[0] // batch_size))
        self._current = 0
        self.y = y
        self.X = X

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if self._current >= self.n_batches:
            raise StopIteration()
        k = self._current
        self._current += 1
        bs = self.batch_size
        k_bs = k * bs
        k_bs_1 = (k + 1) * bs

        return self.X[k_bs: k_bs_1], self.y[k_bs: k_bs_1]


def batches(X, y, bs=32):
    for X, y in AddIterator(X, y, bs):
        X = torch.unsqueeze(torch.Tensor(X), 1)
        y = torch.LongTensor(y)
        yield X, y


# Create the validation/test function
def test(dataloader, model):
    size = len(dataloader.dataset)
    model.eval()
    test_loss, correct = 0, 0

    with torch.no_grad():
        for batch, (X, Y) in enumerate(dataloader):
            X, Y = X.to(device), Y.to(device)
            pred = model(X)

            test_loss += cost(pred, Y).item()
            correct += (pred.argmax(1) == Y).type(torch.float).sum().item()

    test_loss /= size
    correct /= size

    print(f'\nTest Error:\nacc: {(100 * correct):>0.1f}%, avg loss: {test_loss:>8f}\n')



def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return float(torch.tensor(torch.sum(preds == labels).item() / len(preds)))



def train_model(model, criterion, optimizer, n_epochs, patience, bs):
    history = []
    best_loss = np.inf
    best_weights = None
    no_improvements = 0


    # Training loop
    torch.autograd.set_detect_anomaly(True)
    for epoch in range(n_epochs):
        acc = []
        stats = {'epoch': epoch + 1, 'total': n_epochs}

        for phase in ('train', 'val'):
            if phase == 'train':
                training = True
            else:
                training = False

            running_loss = 0

            for batch in batches(*datasets[phase], bs=bs):
                X_batch, y_batch = [b.to(device) for b in batch]
                optimizer.zero_grad()

                # compute gradients only during 'train' phase
                with torch.set_grad_enabled(training):
                    outputs = model(X_batch)  # feature_names
                    train_acc = accuracy(outputs, y_batch)
                    loss = criterion(torch.squeeze(outputs), y_batch)
                    acc.append(train_acc)

                    # don't update weights and rates when in 'val' phase
                    if training:
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item()

            epoch_loss = running_loss / dataset_sizes[phase]
            stats[phase] = epoch_loss
            stats[f"{phase}_accuracy"] = acc

            # early stopping: save weights of the best model so far
            if not training:
                if epoch_loss < best_loss:
                    print('loss improvement on epoch: %d' % (epoch + 1))
                    best_loss = epoch_loss
                    best_weights = copy.deepcopy(model.state_dict())
                    no_improvements = 0
                else:
                    no_improvements += 1

        history.append(stats)
        print('[{epoch:03d}/{total:03d}] train: {train:.9f} - val: {val:.9f}'.format(**stats))
        if no_improvements >= patience:
            print('early stopping after epoch {epoch:03d}'.format(**stats))
            break

    return best_weights, history

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = CNNet().to(device)

    # cost function used to determine best parameters
    cost = torch.nn.CrossEntropyLoss()

    # used to create optimal parameters
    learning_rate = 0.0001
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    X_train, y_train = load_data("training_data_fma_small.json")
    X_valid, y_valid = load_data("validation_data_fma_small.json")

    datasets = {'train': (X_train, y_train), 'val': (X_valid, y_valid)}
    dataset_sizes = {'train': len(X_train), 'val': len(X_valid)}

    label_dict = {7: 0, 12: 1, 6: 2, 13: 3, 5: 4}
    y_train = np.asarray([label_dict[i] for i in list(y_train)])
    y_valid = np.asarray([label_dict[i] for i in list(y_valid)])

    bs = 32
    lr = 0.01
    n_epochs = 30
    patience = 3
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr)

    weights, hist, = train_model(model, criterion, optimizer, n_epochs, patience, bs)
