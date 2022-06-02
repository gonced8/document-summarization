from functools import partial
import json
import os
from pathlib import Path

import arxiv
from autofaiss import build_index
from faiss import read_index
import pytorch_lightning as pl
from sentence_transformers import SentenceTransformer
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
        self.tokenizer = model.tokenizer
        self.collate_fn = model.collate_fn
        self.num_workers = model.hparams.num_workers
        self.batch_size = model.hparams.batch_size
        self.test_batch_size = model.hparams.test_batch_size
        self.chunk_size = model.hparams.chunk_size
        self.n_neighbors = model.hparams.n_neighbors
        self.retrieval = model.hparams.retrieval

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

            # Set device
            device = "cuda" if torch.cuda.is_available() else "cpu"

            # Initialize kNN encoder
            if self.retrieval:
                knn_encoder = SentenceTransformer(
                    "sentence-transformers/sentence-t5-base"
                )

            # Check if dataset is already processed
            for mode in modes:
                dataset_filename = dirpath / f"{mode}.txt"

                # If dataset does not exist, we can't process it
                if not dataset_filename.is_file():
                    continue
                else:
                    dataset[mode] = []

                # TODO: remove this
                # N = 5000
                # N = 1000
                # N = 100
                with open(str(dataset_filename), "r") as f:
                    for line in tqdm(
                        f,
                        total=sum(1 for line in open(str(dataset_filename))),
                        desc=f"Processing {str(dataset_filename)}",
                    ):
                        entry = json.loads(line)

                        article_id = entry["article_id"]
                        title = entry["title"]
                        abstract = entry["abstract"]
                        text = " ".join(entry["article_text"])

                        prompt = "<pad>title: " + title + " " + "abstract: "
                        model_input = prompt + abstract

                        if self.retrieval:
                            # chunks, chunks_ids = self.get_chunks(
                            try:
                                text_tokenized, indices = self.get_chunks(
                                    article_id, text, model_input, knn_encoder, device
                                )
                            except Exception as e:
                                print(e)
                                continue
                            if text_tokenized is None:
                                continue
                        else:
                            chunks = None

                        dataset[mode].append(
                            {
                                "id": article_id,
                                "prompt": prompt,
                                "model_input": model_input,
                                # "chunks": chunks,
                                # "retrieved": chunks_ids,
                                "text_tokenized": text_tokenized.tolist(),
                                "indices": indices.tolist(),
                            }
                        )

                        # N -= 1
                        # if N == 0:
                        #    break

                # Save processed dataset to JSON files
                output_filename = dirpath / f"arxiv_{mode}.json"
                save_json(dataset[mode], str(output_filename))

        # Tokenize
        for mode, dataset_split in dataset.items():
            for sample in dataset_split:
                sample["prompt_ids"] = torch.tensor(
                    self.tokenize(sample["prompt"]), dtype=torch.int
                )[
                    :-1
                ]  # ignore eos_token
                sample["input_ids"] = torch.tensor(
                    self.tokenize(sample["model_input"]), dtype=torch.long
                )

                if self.retrieval:
                    sample["retrieved"] = torch.tensor(
                        [
                            [
                                sample["text_tokenized"][idx]
                                + sample["text_tokenized"][idx + 1]
                                for idx in neighbors
                            ]
                            for neighbors in sample["indices"]
                        ],
                        dtype=torch.int,
                    )

                if self.retrieval and mode == "test":
                    sample["text_tokenized"] = torch.tensor(sample["text_tokenized"])

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

    """
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
    """

    def test_dataloader(self):
        assert len(self.dataset["test"]) > 0
        return DataLoader(
            CustomDataset(self.dataset["test"]),
            batch_size=self.test_batch_size,
            num_workers=self.num_workers,
            collate_fn=partial(self.collate_fn, test=True),
            shuffle=False,
            pin_memory=bool(torch.cuda.device_count()),
        )

    def __str__(self):
        return str(self.dataset)

    def get_chunks(self, article_id, text, model_input, knn_encoder, device=None):
        # Make chunks from text
        text_tokenized = torch.tensor(
            self.tokenize(text, truncation=False),
            dtype=torch.long,
        )
        length = len(text_tokenized)
        pad = (self.chunk_size - length % self.chunk_size) % self.chunk_size

        text_tokenized = torch.nn.functional.pad(
            text_tokenized, (0, pad), value=self.tokenizer.pad_token_id
        )
        text_tokenized = text_tokenized.view(-1, self.chunk_size)

        # We need at least 3 chunks because (C1: C1+C2 and C2: C2+C3)
        if text_tokenized.size(0) < self.n_neighbors + 1:
            return None, None

        neighbor_chunks = self.tokenizer.batch_decode(
            text_tokenized, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )

        neighbor_embeddings = knn_encoder.encode(
            neighbor_chunks[:-1], device=device
        )  # do not include last chunk because we need pairs of chunks

        # Make chunks from model_input
        input_tokenized = torch.tensor(
            self.tokenize(model_input),
            dtype=torch.long,
        )
        length = len(input_tokenized)
        pad = (self.chunk_size - length % self.chunk_size) % self.chunk_size

        input_tokenized = torch.nn.functional.pad(
            input_tokenized, (0, pad), value=self.tokenizer.pad_token_id
        )
        input_tokenized = input_tokenized.view(-1, self.chunk_size)

        input_chunks = self.tokenizer.batch_decode(
            input_tokenized, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )

        input_embeddings = knn_encoder.encode(
            input_chunks[:-1], device=device
        )  # do not include last chunk because last chunk uses retrieval from previous

        # Build index
        index_path = os.path.join(self.data_dir, f"index/{article_id}.index")
        index_infos_path = os.path.join(self.data_dir, f"index/{article_id}_infox.json")
        if os.path.exists(index_path):
            index = read_index(index_path)
        else:
            index, _ = build_index(
                neighbor_embeddings,
                save_on_disk=True,
                index_path=index_path,
                index_infos_path=index_infos_path,
                verbose=30,
            )

        # Retrieve chunks for model input
        _, indices = index.search(
            input_embeddings,
            k=self.n_neighbors,
        )  # fetch 2 neighbors, first indices should be self

        """
        chunks = [
            [" ".join(neighbor_chunks[idx : idx + 2]) for idx in neighbors]
            for neighbors in indices
        ]

        chunks_ids = [
            [text_tokenized[idx : idx + 2].flatten().tolist() for idx in neighbors]
            for neighbors in indices
        ]
        """

        return text_tokenized, indices


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
        # return 200
        return len(self.dataset)
