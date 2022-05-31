import json
import os

import logging
from datasets import load_metric
import pytorch_lightning as pl
from retro_pytorch import RETRO
from retro_pytorch.retro_pytorch import RMSNorm
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers.optimization import AdamW

from .collate_fn import DecoderCollateFn

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class RetroSum(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)

        self.model_name = "RetroSum"
        self.original_model_name = "t5-base"

        # Initialize original model (T5)
        self.config, self.tokenizer, self.model = self.init_retro_from(
            args, self.original_model_name
        )

        self.collate_fn = DecoderCollateFn(self.tokenizer)
        self.f1_em_metric = load_metric("squad_v2")
        self.rouge_metric = load_metric("rouge")

        # self.freeze_embeds()

    def freeze_params(self, model: nn.Module):
        """Set requires_grad=False for each of model.parameters()"""
        for par in model.parameters():
            par.requires_grad = False

    def freeze_embeds(self):
        """Freeze token embeddings and positional embeddings for bart, just token embeddings for t5."""
        self.freeze_params(self.model.model.shared)
        for d in [self.model.model.encoder, self.model.model.decoder]:
            self.freeze_params(d.embed_positions)
            self.freeze_params(d.embed_tokens)

    def forward(self, input_ids, return_loss=False):
        output = self.model(seq=input_ids, retrieved=None, return_loss=return_loss)
        return output

    def training_step(self, batch, batch_idx):
        output = self.forward(batch["input_ids"], return_loss=True)
        loss = output

        self.log("train_loss", loss, batch_size=len(batch["input_ids"]))
        return loss

    """
    def validation_step(self, batch, batch_idx):
        output = self.forward(
            batch["input_ids"], batch["attention_mask"], batch["labels"]
        )
        loss = output.loss

        # Mask question tokens before decoding
        output = torch.argmax(output.logits, dim=2)
        output[batch["labels"] == -100] = self.tokenizer.bos_token_id

        predictions = self.tokenizer.batch_decode(
            output,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        # Left strip whitespaces
        predictions = [prediction.lstrip() for prediction in predictions]

        references = batch["truth_output"]

        self.rouge_metric.add_batch(predictions=predictions, references=references)

        self.log("val_loss", loss, prog_bar=True, batch_size=len(batch["input_ids"]))
        return loss

    def validation_epoch_end(self, outputs):
        rouge_score = self.rouge_metric.compute()
        rouge_score = parse_rouge_score(rouge_score)

        self.log_dict(rouge_score, prog_bar=True)
        return

    def on_epoch_start(self):
        print()

    def test_step(self, batch, batch_idx):
        min_length = min([len(sample) for sample in batch["input_ids"]])

        output = self.model.generate(
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
            min_length=min_length + 1,
            max_length=self.hparams.max_output_length,
            do_sample=True,
            no_repeat_ngram_size=self.hparams.no_repeat_ngram_size,
        )

        # Mask model_input_ids from output
        output = output[:, batch["input_ids"].size(1) :]

        # Decode
        predictions = self.tokenizer.batch_decode(
            output,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        # Left strip whitespaces
        predictions = [prediction.lstrip() for prediction in predictions]

        # Compute metrics and prepare results output
        results = []

        # F1 and Exact match
        for sample_id, model_input, reference, prediction in zip(
            batch["id"], batch["model_input"], batch["truth_output"], predictions
        ):
            results.append(
                {
                    "id": sample_id,
                    "model_input": model_input,
                    "truth_output": reference,
                    "model_answer": prediction,
                }
            )

            reference = {
                "id": id,
                "answers": {"answer_start": [0], "text": [reference]},
            }
            prediction = {
                "id": id,
                "prediction_text": prediction,
                "no_answer_probability": 0.0,
            }
            self.f1_em_metric.add(prediction=prediction, reference=reference)

        # ROUGE
        self.rouge_metric.add_batch(
            predictions=predictions, references=batch["truth_output"]
        )

        return results

    def test_epoch_end(self, outputs):
        # Compute metrics
        score = {}

        f1_em_score = self.f1_em_metric.compute()
        score["F1"] = f1_em_score["f1"] / 100
        score["Exact match"] = f1_em_score["exact"] / 100

        rouge_score = self.rouge_metric.compute()
        score["ROUGE"] = parse_rouge_score(rouge_score)

        # Merge outputs of the multiple steps
        outputs = [output for batch_outputs in outputs for output in batch_outputs]

        # Save output with scores
        if self.hparams.results_filename:
            outputs.insert(0, score)
            filename = os.path.join("results", f"{self.hparams.results_filename}.json")
            with open(filename, "w") as f:
                json.dump(outputs, f, indent=4)
                print(f"Saved test output to: {filename}")

        self.log_dict(score, prog_bar=True)

        return score

    def generate(self, input_ids, max_length=None, *args, **kargs):
        if max_length is None:
            max_length = self.hparams.max_output_length

        if not batch.is_cuda:
            batch.to(self.device)

        output = self.model.generate(
            input_ids,
            max_length,
            *args,
            **kargs,
        )

        predictions = self.tokenizer.batch_decode(
            output,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        # Left strip whitespaces
        predictions = [prediction.lstrip() for prediction in predictions]

        return predictions
    """

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.hparams.lr)

        """
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        scheduler = {
            "scheduler": transformers.get_cosine_with_hard_restarts_schedule_with_warmup(
                optimizer,
                num_warmup_setps=100,
                num_training_steps=2000,
                num_cycles=self.hparams.max_epochs,
            ),
            "interval": "step",
        }
        return [optimizer], [scheduler]
        """
        return optimizer

    def tokenize(self, sample):
        max_input_length = self.hparams.max_input_length
        max_output_length = self.hparams.max_output_length

        # Disable long sequence warning during tokenization
        logging.getLogger("transformers.tokenization_utils_base").setLevel(
            logging.ERROR
        )

        # Tokenize model input into input_ids
        input_ids = self.tokenizer(
            sample,
            truncation=True,
            max_length=max_output_length,
            return_attention_mask=False,
        )["input_ids"]

        return input_ids

    @staticmethod
    def init_retro_from(args, original_model_name):
        tokenizer = AutoTokenizer.from_pretrained(original_model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(original_model_name)

        # Initialize RETRO
        config = {
            "num_tokens": model.config.vocab_size,
            "max_seq_len": args.max_output_length,
            "enc_dim": model.config.d_model,
            "enc_depth": model.config.num_layers,
            "enc_cross_attn_layers": None,
            "dec_depth": model.config.num_decoder_layers,
            "dec_cross_attn_layers": tuple(
                range(
                    6,  # start in 6th layer
                    model.config.num_decoder_layers + 1,  # include last cca
                    3,  # every 3rd layer
                )
            ),
            "heads": model.config.num_heads,
            "dec_dim": model.config.d_model,
            "dim_head": model.config.d_kv,
            "enc_attn_dropout": model.config.dropout_rate,
            "enc_ff_dropout": model.config.dropout_rate,
            "dec_attn_dropout": model.config.dropout_rate,
            "dec_ff_dropout": model.config.dropout_rate,
            "chunk_size": 64,
            "pad_id": tokenizer.pad_token_id,
            "norm_klass": RMSNorm,
        }

        retro = RETRO(**config)

        # Copy original model parameters
        with torch.no_grad():
            # Embeddings
            retro.token_emb.weight.copy_(model.get_input_embeddings().weight)

            # Copy encoder
            for i in range(config["enc_depth"]):
                # Attention
                retro.encoder.get_parameter(f"layers.{i}.0.fn.to_q.weight").copy_(
                    model.encoder.get_parameter(
                        f"block.{i}.layer.0.SelfAttention.q.weight"
                    )
                )
                retro.encoder.get_parameter(f"layers.{i}.0.fn.to_k.weight").copy_(
                    model.encoder.get_parameter(
                        f"block.{i}.layer.0.SelfAttention.k.weight"
                    )
                )
                retro.encoder.get_parameter(f"layers.{i}.0.fn.to_v.weight").copy_(
                    model.encoder.get_parameter(
                        f"block.{i}.layer.0.SelfAttention.v.weight"
                    )
                )
                retro.encoder.get_parameter(f"layers.{i}.0.fn.to_out.weight").copy_(
                    model.encoder.get_parameter(
                        f"block.{i}.layer.0.SelfAttention.o.weight"
                    )
                )
                retro.encoder.get_parameter(f"layers.{i}.0.fn.to_out.bias").fill_(0.0)
                retro.encoder.get_parameter(f"layers.{i}.0.norm.gamma").copy_(
                    model.encoder.get_parameter(f"block.{i}.layer.0.layer_norm.weight")
                )

                # Fully-connected
                retro.encoder.get_parameter(f"layers.{i}.2.fn.ff.0.weight").copy_(
                    model.encoder.get_parameter(
                        f"block.{i}.layer.1.DenseReluDense.wi.weight"
                    )
                )
                retro.encoder.get_parameter(f"layers.{i}.2.fn.ff.0.bias").fill_(0.0)
                retro.encoder.get_parameter(f"layers.{i}.2.fn.ff.3.weight").copy_(
                    model.encoder.get_parameter(
                        f"block.{i}.layer.1.DenseReluDense.wo.weight"
                    )
                )
                retro.encoder.get_parameter(f"layers.{i}.2.fn.ff.3.bias").fill_(0.0)
                retro.encoder.get_parameter(f"layers.{i}.2.norm.gamma").copy_(
                    model.encoder.get_parameter(f"block.{i}.layer.1.layer_norm.weight")
                )

            # Copy decoder
            for i in range(config["dec_depth"]):
                # Attention
                retro.decoder.get_parameter(f"layers.{i}.0.fn.to_q.weight").copy_(
                    model.decoder.get_parameter(
                        f"block.{i}.layer.0.SelfAttention.q.weight"
                    )
                )
                retro.decoder.get_parameter(f"layers.{i}.0.fn.to_k.weight").copy_(
                    model.decoder.get_parameter(
                        f"block.{i}.layer.0.SelfAttention.k.weight"
                    )
                )
                retro.decoder.get_parameter(f"layers.{i}.0.fn.to_v.weight").copy_(
                    model.decoder.get_parameter(
                        f"block.{i}.layer.0.SelfAttention.v.weight"
                    )
                )
                retro.decoder.get_parameter(f"layers.{i}.0.fn.to_out.weight").copy_(
                    model.decoder.get_parameter(
                        f"block.{i}.layer.0.SelfAttention.o.weight"
                    )
                )
                retro.decoder.get_parameter(f"layers.{i}.0.fn.to_out.bias").fill_(0.0)
                retro.decoder.get_parameter(f"layers.{i}.0.norm.gamma").copy_(
                    model.decoder.get_parameter(f"block.{i}.layer.0.layer_norm.weight")
                )

                # Fully-connected
                retro.decoder.get_parameter(f"layers.{i}.2.fn.ff.0.weight").copy_(
                    model.decoder.get_parameter(
                        f"block.{i}.layer.2.DenseReluDense.wi.weight"
                    )
                )
                retro.decoder.get_parameter(f"layers.{i}.2.fn.ff.0.bias").fill_(0.0)
                retro.decoder.get_parameter(f"layers.{i}.2.fn.ff.3.weight").copy_(
                    model.decoder.get_parameter(
                        f"block.{i}.layer.2.DenseReluDense.wo.weight"
                    )
                )
                retro.decoder.get_parameter(f"layers.{i}.2.fn.ff.3.bias").fill_(0.0)
                retro.decoder.get_parameter(f"layers.{i}.2.norm.gamma").copy_(
                    model.decoder.get_parameter(f"block.{i}.layer.2.layer_norm.weight")
                )

        return config, tokenizer, retro


def parse_rouge_score(result):
    return {k: round(v.mid.fmeasure, 4) for k, v in result.items()}
