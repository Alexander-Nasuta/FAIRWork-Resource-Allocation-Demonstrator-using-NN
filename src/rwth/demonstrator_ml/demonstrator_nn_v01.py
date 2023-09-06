import torch

import numpy as np
import pathlib as pl
import pandas as pd

from torch import nn, optim
from torch.utils.data import Dataset

from rwth.utils.logger import log


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
    def __init__(self, data_directory: str | pl.Path):
        self.data_directory = pl.Path(data_directory)
        self.filenames = list(self.data_directory.glob("*.csv"))

        if not len(self.filenames):
            raise ValueError(f"no .csv files found in '{self.data_directory}'")

        sample_file = pd.read_csv(self.filenames[0])
        # drop index column
        sample_file = sample_file.drop(sample_file.columns[0], axis=1)
        sample_y = sample_file.pop("final allocation")

        self.n_y_params, *_ = np.ravel(sample_y.to_numpy()).shape
        self.n_x_params, *_ = np.ravel(sample_file.to_numpy()).shape

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):  # idx means index of the chunk.
        # In this method, we do all the preprocessing.
        # First read data from files in a chunk. Preprocess it. Extract labels. Then return data and labels.
        file = self.filenames[idx]

        df = pd.read_csv(open(file, 'r'))
        # drop index column
        df = df.drop(df.columns[0], axis=1)
        y_data = np.ravel(df.pop("final allocation").to_numpy())
        x_data = np.ravel(df.to_numpy())

        # The following condition is actually needed in Pytorch. Otherwise, for our particular example, the iterator will be an infinite loop.
        # Readers can verify this by removing this condition.
        if idx == self.__len__():
            raise IndexError

        return x_data, y_data


def train_demonstrator_model(n_epochs: int) -> None:
    # list all .csv files in the training-data directory
    data_dir_path = pl.Path("./resources").joinpath("training-data").joinpath("demonstrator-v01")

    dataset = CustomDataset(data_directory=data_dir_path)

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
        for batch_idx, (x, y) in enumerate(data_loader):
            # x = x.float()
            # y = y.float()
            optimizer.zero_grad()
            y_pred = model(x.to(torch.float32)).to(torch.float32)
            loss = loss_fn(y_pred, y.to(torch.float32))
            loss.backward()
            optimizer.step()
            log.info(f"epoch: {epoch}, batch_idx: {batch_idx}, loss: {loss.item()}")

    # save model
    torch.save(model.state_dict(), "./resources/trained-models/demonstrator-v01/model1.pt")


def get_model() -> DemonstratorNeuralNet:
    if not pl.Path("./resources/trained-models/demonstrator-v01/model1.pt").exists():
        raise FileNotFoundError("could not find model1.pt. run 'train_demonstrator_model' first.")
    # load model

    # list all .csv files in the training-data directory
    # data_dir_path = pl.Path("./resources").joinpath("training-data").joinpath("demonstrator-v01")
    # dataset = CustomDataset(data_directory=data_dir_path)

    x_dim = 1431
    y_dim = 159

    model = DemonstratorNeuralNet(
        input_dim=x_dim,
        hidden_dim=10,
        output_dim=y_dim
    )
    model.load_state_dict(torch.load("./resources/trained-models/demonstrator-v01/model1.pt"))

    return model


def processed_prediction(datapoint: np.ndarray, n_workers) -> np.ndarray:
    model = get_model()
    # predict
    y_pred = model(torch.from_numpy(datapoint).to(torch.float32))
    # torch to numpy
    y_pred = y_pred.detach().numpy()
    # set n_workers largest values to 1, rest to 0
    y_pred = np.array([1 if i in np.argpartition(y_pred, -n_workers)[-n_workers:] else 0 for i in range(len(y_pred))])
    return y_pred


def model_prediction(datapoint: np.ndarray) -> np.ndarray:
    model = get_model()
    # predict
    y_pred = model(torch.from_numpy(datapoint).to(torch.float32))
    # torch to numpy
    y_pred = y_pred.detach().numpy()
    return y_pred


def prediction_example() -> None:
    # get random csv file from training-data directory
    files = pl.Path("./resources").joinpath("training-data").joinpath("demonstrator-v01").glob("*.csv")
    file_name = np.random.choice(list(files))

    # read csv file
    df = pd.read_csv(file_name)
    # drop index column
    df = df.drop(df.columns[0], axis=1)
    y_data = df.pop("final allocation")
    log.info(f"input_data cols: {df.columns}")
    log.info(f"input_data:")
    log.info(df.head())

    x_data = df.to_numpy()
    log.info(f"not flattered input_data as numpy array (shape: {x_data.shape}): \n{x_data}")

    # y_data = np.ravel(y_data.to_numpy())
    x_data = np.ravel(x_data)

    log.info(f"input_data as numpy array (shape: {x_data.shape}): \n{x_data}")

    # predict
    y_pred = processed_prediction(x_data, n_workers=4)
    log.info(f"y_pred: {y_pred}")


if __name__ == '__main__':
    # change working directory to the root of the project
    import os
    os.chdir(pl.Path(__file__).parent.parent.parent.parent)
    log.info(f"working directory: {os.getcwd()}")

    train_demonstrator_model(n_epochs=5)
    model = get_model()
    log.info(f"trained model: {model}, file: ./resources/trained-models/demonstrator-v01/model1.pt")
    # prediction_example()
