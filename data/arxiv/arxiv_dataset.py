import pickle
import json
import os
from pathlib import Path

import arxiv
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


class ArxivDataset(pl.LightningDataModule):
    def __init__(
        self,
        model,
    ):
        super().__init__()
        self.data_dir = "data/arxiv/dataset"
        self.tokenize = model.tokenize
        self.collate_fn = model.collate_fn
        self.num_workers = model.hparams.num_workers
        self.batch_size = model.hparams.batch_size
        self.test_batch_size = model.hparams.test_batch_size

    def prepare_data(self):
        # Check if data_dir is a directory
        dirpath = Path(self.data_dir)
        assert dirpath.is_dir()

        modes = ["train", "val", "test"]

        # Check if dataset is already processed
        dataset = {}
        for mode in modes:
            dataset_filename = dirpath / f"arxiv_{mode}.json"

            # If dataset file exists
            if dataset_filename.is_file():
                # Load dataset
                dataset[mode] = load_json(str(dataset_filename))

        # check if there is already a dataset
        if not dataset:
            # Check if data_dir is a directory
            dirpath = Path(self.data_dir)
            assert dirpath.is_dir()

            # Load dataset with arXiv metadata
            arxiv_metadata = load_jsonl(
                str(dirpath / "arxiv-metadata-oai-snapshot.json")
            )
            arxiv_title = {
                article["id"]: article["title"]
                for article in tqdm(arxiv_metadata, desc="Extracting titles")
            }

            # Check if dataset is already processed
            for mode in modes:
                dataset_filename = dirpath / f"{mode}.txt"

                # If dataset does not exist, we can't process it
                if not dataset_filename.is_file():
                    continue
                else:
                    dataset[mode] = []

                N = 10000  # TODO: remove this
                with open(str(dataset_filename), "r") as f:
                    for line in tqdm(
                        f,
                        total=sum(1 for line in open(str(dataset_filename))),
                        desc=f"Processing {str(dataset_filename)}",
                    ):
                        entry = json.loads(line)
                        text = " ".join(entry["article_text"])
                        abstract = (
                            " ".join(entry["abstract_text"])
                            .replace("<S> ", "")
                            .replace(" </S>", "")
                        )
                        # title = query_title_from_id(entry["article_id"])
                        try:
                            title = arxiv_title[entry["article_id"]]
                        except KeyError:
                            continue

                        model_input = (
                            "<pad>title: " + title + " " + "abstract:" + abstract
                        )

                        dataset[mode].append(
                            {"id": entry["article_id"], "model_input": model_input}
                        )

                        N -= 1
                        if N == 0:
                            break

                # Save processed dataset to JSON files
                output_filename = dirpath / f"arxiv_{mode}.json"
                save_json(dataset[mode], str(output_filename))

        # Tokenize
        for mode, dataset_split in dataset.items():
            for sample in dataset_split:
                sample["input_ids"] = torch.tensor(
                    self.tokenize(sample["model_input"]), dtype=torch.long
                )

        self.dataset = dataset

    def save_data(self, N=None, output_path="./", append=None, protocol=None):
        assert isinstance(protocol, int)
        dirpath = Path(output_path)
        assert dirpath.is_dir()
        if append:
            append_str = f".{append}"
        else:
            append_str = ""

        for mode in self.dataset.keys():
            with open(
                os.path.join(dirpath, f"{mode}{append_str}.pickle"), "wb"
            ) as handle:
                if not N:
                    N = len(self.dataset[mode])
                if protocol:
                    pickle.dump(self.dataset[mode][:N], handle, protocol=protocol)
                else:
                    pickle.dump(self.dataset[mode][:N], handle)

    def train_dataloader(self):
        assert len(self.dataset["train"]) > 0
        return DataLoader(
            CustomDataset(self.dataset["train"]),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            shuffle=True,
            pin_memory=bool(torch.cuda.device_count()),
        )

    def val_dataloader(self):
        assert len(self.dataset["val"]) > 0
        return DataLoader(
            CustomDataset(self.dataset["val"]),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            shuffle=False,
            pin_memory=bool(torch.cuda.device_count()),
        )

    def test_dataloader(self):
        assert len(self.dataset["test"]) > 0
        return DataLoader(
            CustomDataset(self.dataset["test"]),
            batch_size=self.test_batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            shuffle=False,
            pin_memory=bool(torch.cuda.device_count()),
        )

    def __str__(self):
        return str(self.dataset)


def load_json(filename):
    with open(filename, "r") as f:
        data = json.load(f)
    return data


def load_jsonl(filename):
    with open(filename, "r") as f:
        data = [
            json.loads(line) for line in tqdm(f, total=2072073, desc="Loading JSONL")
        ]

    return data


def save_json(data, filename):
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)
    return


def query_title_from_id(article_id):
    if "." not in article_id:
        return None
    search = arxiv.Search(
        id_list=[article_id],
        max_results=1,
    )
    return next(search.results()).title


def search_title_from_id(article_id, arxiv_metadata):
    match = next((item for item in arxiv_metadata if item["id"] == article_id), None)
    if match is not None:
        match = match["title"]
    return match


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, idx):
        return self.dataset[idx]

        """
        {
            "input_ids": self.dataset[idx]["input_ids"].to(torch.long),
            "labels": self.dataset[idx]["output_ids"].to(torch.long),
        }
        """

    def __len__(self):
        return len(self.dataset)
