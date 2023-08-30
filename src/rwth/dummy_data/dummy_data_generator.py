import random
import uuid

import pandas as pd
import pathlib as pl
import numpy as np
from rwth.utils.logger import log


def generate_csv_files(
        n_files=2,
        n_workers: int = 159,
        n_final_allocations=4,
        filename_prefix: str = "dummy_",
        data_dir_path: str | pl.Path = pl.Path("./resources").joinpath("training-data").joinpath("demonstrator-v01")
) -> None:
    pl.Path(data_dir_path).mkdir(parents=True, exist_ok=True)
    for i in range(n_files):
        file_name = f"{filename_prefix}{uuid.uuid4()}.csv"
        log.info(f"[{i + 1}/{n_files}] generating dummy data...")

        w_availabilities = [random.random() > 20 / 159 for _ in range(n_workers)]
        w_availabilities_indexes = [i for i, av in enumerate(w_availabilities) if av]

        if len(w_availabilities_indexes) < n_final_allocations:
            log.warning("generated less available workers than needed. skipping this generated data entry.")
        random.shuffle(w_availabilities_indexes)
        ute_head_allocation = [False for _ in range(n_workers)]

        for _, index in zip(range(n_final_allocations), w_availabilities_indexes):
            ute_head_allocation[index] = True

        worker_preference = [random.random() if av else 0.0 for av in w_availabilities]

        final_allocation = [False for _ in range(n_workers)]
        weighted_sum = np.array([
            float(ute) * 0.6 + 0.4 * pref
            for ute, pref
            in zip(ute_head_allocation, worker_preference)
        ])

        for indx in np.argpartition(weighted_sum, -n_final_allocations)[-n_final_allocations:]:
            final_allocation[indx] = True

        data = {
            "ID": [100_001 + i for i in range(n_workers)],
            "Woker avaibale": w_availabilities,
            "Medical condtion": [random.random() > 0.5 for _ in range(n_workers)],
            "Efficiency on the line": [random.random() > 0.5 for _ in range(n_workers)],
            "Difficulty of the geometry": [random.random() > 0.5 for _ in range(n_workers)],
            "production priority": [0 for _ in range(n_workers)],
            "due date in days": [1 for _ in range(n_workers)],

            "UTE allocation": ute_head_allocation,
            "worker preference": worker_preference,

            "final allocation": final_allocation
        }
        df = pd.DataFrame(data).astype(float)
        path = pl.Path(data_dir_path).joinpath(file_name)

        log.info(f"saving df as file '{file_name}'")
        # print(df.head())
        df.to_csv(path_or_buf=path)


if __name__ == '__main__':
    log.info("make sure your working directory is the root of the project! "
             "PyCharm -> Edit configurations... -> Environment -> Working directory")
    generate_csv_files(n_files=100)
