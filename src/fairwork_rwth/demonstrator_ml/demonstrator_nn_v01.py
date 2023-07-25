import torch

import numpy as np
import pathlib as pl
import pandas as pd

from torch import nn, optim
from torch.utils.data import Dataset

from fairwork_rwth.utils.logger import log


class DemonstratorNeuralNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layer_1 = nn.Linear(input_dim, hidden_dim)
        self.layer_2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.layer_1(x))
        x = torch.relu(self.layer_2(x))
        x = self.layer_3(x)
        return x


class CustomDataset(Dataset):
    def __init__(self, data_directory: str | pl.Path, batch_size):
        self.data_directory = pl.Path(data_directory)
        self.filenames = list(self.data_directory.glob("*.csv"))
        self.batch_size = batch_size

        if not len(self.filenames):
            raise ValueError(f"no .csv files found in '{self.data_directory}'")

        sample_file = pd.read_csv(self.filenames[0]).to_numpy()
        self.file_shape = sample_file.shape

        self.n_y_params = 1
        self.n_x_params = self.file_shape[1] - 1

    def __len__(self):
        return int(np.ceil(
            len(self.filenames) /
            float(self.batch_size)
        ))  # Number of chunks.

    def __getitem__(self, idx):  # idx means index of the chunk.
        # In this method, we do all the preprocessing.
        # First read data from files in a chunk. Preprocess it. Extract labels. Then return data and labels.
        batch_x = self.filenames[
                  idx * self.batch_size
                  :
                  (idx + 1) * self.batch_size
                  ]  # This extracts one batch of file names from the list `filenames`.

        x_data = []
        y_data = []

        for file in batch_x:
            temp = pd.read_csv(open(file, 'r'))  # Change this line to read any other type of file
            y = temp.pop("final allocation")
            x = temp.to_numpy()
            # flatten x
            x = x.ravel()
            x_data.append(x)
            y_data.append(y.to_numpy())

        # The following condition is actually needed in Pytorch. Otherwise, for our particular example, the iterator will be an infinite loop.
        # Readers can verify this by removing this condition.
        if idx == self.__len__():
            raise IndexError

        # convert list of numpy arrays to one numpy array
        x_data = np.array(x_data)
        y_data = np.array(y_data)
        return x_data, y_data


def train_demonstrator_model(n_epochs:int) -> None:
    # list all .csv files in the training-data directory
    data_dir_path = pl.Path("./resources").joinpath("training-data").joinpath("demonstrator-v01")

    dataset = CustomDataset(data_directory=data_dir_path, batch_size=5)

    model = DemonstratorNeuralNet(
        input_dim=dataset.n_x_params,
        hidden_dim=10,
        output_dim=dataset.n_y_params
    )
    log.info(f"model: {model}")

    # define pytorch adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # define pytorch mean squared error loss function
    loss_fn = nn.MSELoss()

    # define pytorch data loader
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=5, shuffle=True)

    # define pytorch training loop
    for epoch in range(n_epochs):
        for batch_idx, (data, target) in enumerate(data_loader):
            optimizer.zero_grad()
            print(data.shape)
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 10 == 0:
                log.info(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(data_loader.dataset)} "
                         f"({100. * batch_idx / len(data_loader):.0f}%)]\tLoss: {loss.item():.6f}")


if __name__ == '__main__':
    train_demonstrator_model(n_epochs=5)
