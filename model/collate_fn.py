import torch


class EncoderDecoderCollateFn:
    def __init__(self, tokenizer):
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, samples):
        # Transpose batch
        batch = {k: [sample[k] for sample in samples] for k in samples[0]}

        # Pad samples
        # Attention mask has do be first because it uses the sample size before padding
        batch["attention_mask"] = torch.nn.utils.rnn.pad_sequence(
            [
                torch.ones_like(sample, dtype=torch.bool)
                for sample in batch["input_ids"]
            ],
            batch_first=True,
        )

        batch["input_ids"] = torch.nn.utils.rnn.pad_sequence(
            batch["input_ids"],
            batch_first=True,
            padding_value=self.pad_token_id,
        )

        batch["labels"] = torch.nn.utils.rnn.pad_sequence(
            batch["labels"],
            batch_first=True,
            padding_value=self.pad_token_id,
        )

        return batch


class DecoderCollateFn:
    def __init__(self, tokenizer):
        self.pad_token_id = tokenizer.eos_token_id

    def __call__(self, samples):
        # Transpose batch
        batch = {k: [sample[k] for sample in samples] for k in samples[0]}

        # Pad samples
        # Attention mask has do be first because it uses the sample size before padding

        """
        batch["attention_mask"] = torch.nn.utils.rnn.pad_sequence(
            [
                torch.ones_like(sample, dtype=torch.bool)
                for sample in batch["input_ids"]
            ],
            batch_first=True,
        )
        """

        batch["prompt_ids"] = torch.nn.utils.rnn.pad_sequence(
            batch["prompt_ids"],
            batch_first=True,
            padding_value=self.pad_token_id,
        )

        batch["input_ids"] = torch.nn.utils.rnn.pad_sequence(
            batch["input_ids"],
            batch_first=True,
            padding_value=self.pad_token_id,
        )

        if "retrieved" in batch:
            batch["retrieved"] = torch.nn.utils.rnn.pad_sequence(
                batch["retrieved"],
                batch_first=True,
                padding_value=self.pad_token_id,
            )

        """
        batch["labels"] = torch.nn.utils.rnn.pad_sequence(
            batch["labels"],
            batch_first=True,
            padding_value=-100,  # TODO: is this always correct?
        )
        """

        return batch


def pad_sequence(samples, padding_value=0, padding_side="right"):
    if padding_side == "left":
        max_len = max([len(sample) for sample in samples])
        batch = torch.full(
            (len(samples), max_len), fill_value=padding_value, dtype=samples[0].dtype
        )

        for i, sample in enumerate(samples):
            batch[i, -len(sample) :] = sample

        return batch
    else:
        return torch.nn.utils.rnn.pad_sequence(
            samples, True, padding_value, padding_side
        )
