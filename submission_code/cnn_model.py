from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import yaml

from sklearn import metrics


NOT_HPARAMS = [
    "self",
    "__class__",
    "fpath",
    "verbose",
    "use_cuda",
    "n_batches",
    "model_fpath",
]


def grid_show(images):
    img = torchvision.utils.make_grid(images)
    img = img / 2 + 0.5  # unnormalize
    img_arr = img.numpy()
    plt.imshow(np.transpose(img_arr, (1, 2, 0)))
    plt.show()


def conv_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)
    h = np.floor(
        ((h_w[0] + (2 * pad) - (dilation * (kernel_size[0] - 1)) - 1) / stride) + 1
    )
    w = np.floor(
        ((h_w[1] + (2 * pad) - (dilation * (kernel_size[1] - 1)) - 1) / stride) + 1
    )
    return int(h), int(w)


def conv_output_size(h_w, k_pool, out_ch, n_layers):
    for _ in range(n_layers):
        h_w = conv_output_shape(h_w, kernel_size=k_pool, stride=k_pool)
    return out_ch * np.multiply(*h_w)


class SimpleConvModel(nn.Module):
    def __init__(self, k_conv1, k_conv2, k_conv3, k_pool, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, k_conv1, padding="same")
        self.pool = nn.MaxPool2d(k_pool)
        self.conv2 = nn.Conv2d(6, 12, k_conv2, padding="same")
        self.conv3 = nn.Conv2d(12, out_ch, k_conv3, padding="same")

        out_size = conv_output_size([128, 128], k_pool, out_ch, n_layers=3)
        print("Conv. parameters:", out_size)

        self.fc1 = nn.Linear(out_size, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 13)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class BasicCNN(BaseEstimator):
    def __init__(
        self,
        learning_rate=1e-3,
        momentum=0.9,
        nb_epoch=10,
        batch_size=4,
        verbose=False,
        use_cuda=False,
        k_conv1=3,
        k_conv2=3,
        k_conv3=3,
        k_pool=3,
        out_ch=24,
        n_batches=1000,
        model_fpath=None,
    ):

        super().__init__()

        self.hyperparameters = {
            k: v for k, v in locals().items() if k not in NOT_HPARAMS
        }

        self.learning_rate = learning_rate
        self.momentum = momentum
        self.nb_epoch = nb_epoch
        self.batch_size = batch_size
        self.verbose = verbose
        self.use_cuda = False if use_cuda is None else use_cuda
        self.k_conv1 = k_conv1
        self.k_conv2 = k_conv2
        self.k_conv3 = k_conv3
        self.k_pool = k_pool
        self.out_ch = out_ch
        self.n_batches = n_batches
        self.model_fpath = model_fpath

        self.net = SimpleConvModel(k_conv1, k_conv2, k_conv3, k_pool, out_ch)

    def numpy_loader(self, X, y=None, batch_size=4):
        if isinstance(X, torch.utils.data.DataLoader):
            return X  # X is a DataLoader dataset (possibly including y)

        min_max_scaler = sklearn.preprocessing.MinMaxScaler()
        X = min_max_scaler.fit_transform(X)
        X = X * 2 - 1
        X = X.reshape(-1, 128, 128, 3)
        X = np.transpose(X, (0, 3, 1, 2))

        if y is None:
            dataset = torch.utils.data.TensorDataset(torch.Tensor(X).float())
        else:
            dataset = torch.utils.data.TensorDataset(
                torch.Tensor(X).float(), torch.Tensor(y).long()
            )
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    def _train(self, train_loader, optimizer, criterion, epoch):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = self.net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % self.n_batches == self.n_batches - 1:  # Print every n mini-batches
                print(
                    f"epoch: {epoch + 1} | batch: {i + 1:5d} | "
                    f"loss: {running_loss / self.n_batches:.3f}"
                )
                running_loss = 0.0

        return running_loss / self.n_batches

    def fit(self, X, y, X_valid=None, y_valid=None):
        """
        param X: numpy.ndarray
            shape = (num_sample, C * W * H)
            with C = 3, W = H = 128
        param y: numpy.ndarray
            shape = (num_sample, 1)
        """
        train_loader = self.numpy_loader(X, y, batch_size=self.batch_size)
        if X_valid is not None and y_valid is not None:
            valid_loader = self.numpy_loader(
                X_valid, y_valid, batch_size=self.batch_size
            )

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(
            self.net.parameters(), lr=self.learning_rate, momentum=self.momentum
        )

        losses, accuracies, epochs = [], [], []
        for epoch in range(self.nb_epoch):  # loop over the dataset multiple times
            loss = self._train(train_loader, optimizer, criterion, epoch)
            if X_valid is not None and y_valid is not None:
                accuracy = self.evaluate(*self.predict(valid_loader))
            else:
                accuracy = None

            losses.append(loss)
            accuracies.append(accuracy)
            epochs.append(epoch + 1)

        print("Finished Training")

        self.fit_info_df = pd.DataFrame(
            {"epoch": epochs, "loss": losses, "accuracy": accuracies}
        )

        self.save()
        return self

    def save(self):
        if self.model_fpath is not None:
            torch.save(self.net.state_dict(), self.model_fpath)
            with self.model_fpath.with_suffix(".yml").open("w") as f:
                yaml.dump(self.hyperparameters, f)

            self.fit_info_df.to_csv(self.model_fpath.with_suffix(".csv"))

            print(f"Model saved in {self.model_fpath}")

    def predict(self, X, net_fpath: Path = None, pred_fpath: Path = None):

        if net_fpath is not None:
            self.net.load_state_dict(torch.load(net_fpath))

        self.net.eval()

        test_loader = self.numpy_loader(X, batch_size=self.batch_size)

        predictions, labels = [], []

        with torch.no_grad():  # Not training => No need to compute gradients
            for data in test_loader:
                if len(data) == 2:
                    images, labs = data
                else:
                    (images,) = data
                    # labs = [None] * images.size()[0]
                    labs = None
                outputs = self.net(images)  # Run images through the network
                _, predicted = torch.max(outputs.data, 1)
                predictions.extend(predicted)
                if labs is not None:
                    labels.extend(labs)

        if pred_fpath is not None:
            with pred_fpath.open("w") as f:
                print(*np.array(predictions), sep="\n", file=f)

        if labels:
            return np.array(predictions), np.array(labels)

        return np.array(predictions)

    @staticmethod
    def evaluate(predictions, labels, fpath: Path = None):
        accuracy = metrics.accuracy_score(labels, predictions)
        txt = f"Accuracy: " f"{100 * accuracy:.2f}%"
        print(txt)
        if fpath is not None:
            with fpath.with_suffix(".txt").open("w") as f:
                print(txt, file=f)

        return accuracy
