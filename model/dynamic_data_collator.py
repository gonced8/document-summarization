import torch


class DynamicDataCollator:
    def __init__(self, tokenizer):
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, batch):
        return {
            k: torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(sample[k]) for sample in batch],
                batch_first=True,
                padding_value=self.pad_token_id,
            )
            for k in batch[0]
        }
