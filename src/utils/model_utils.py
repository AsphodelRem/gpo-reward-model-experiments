from collections import OrderedDict

import pandas as pd
from datasets import Dataset, DatasetDict


def get_trainable_weights(model):
    save_dict = OrderedDict()
    state_dict = model.state_dict()
    for key, value in model.named_parameters():
        if value.requires_grad:
            if "pretrained_model." in key:
                key = key.replace("pretrained_model.", "")
            save_dict[key] = state_dict[key]
    return save_dict


def load_and_prepare_dataset(config: dict):
    dataset = pd.read_parquet(config.dataset.dataset_path)
    dataset = Dataset.from_pandas(dataset)

    if "test" not in dataset:
        train_test_split = dataset.train_test_split(
            test_size=config.dataset.train_test_split_ratio
        )
        dataset = DatasetDict(
            {"train": train_test_split["train"], "test": train_test_split["test"]}
        )

    return dataset
