import os

from .arxiv import *


def get_dataset(model):
    name = model.hparams.data_name.lower()

    if "arxiv" in name:
        return ArxivDataset(model)
    else:
        raise NotImplementedError()
