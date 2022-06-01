import json
import os

from einops import rearrange
from datasets import load_metric
import logging
import pytorch_lightning as pl
from retro_pytorch import RETRO
from retro_pytorch.retro_pytorch import RMSNorm
from retro_pytorch.training import *
from sentence_transformers import SentenceTransformer
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

        # Initialize kNN encoder
        if self.hparams.retrieval:
            self.knn_encoder = SentenceTransformer(
                "sentence-transformers/sentence-t5-base"
            )

        # Initialize original model (T5)
        self.config, self.tokenizer, self.model = self.init_retro_from(
            self.hparams, self.original_model_name
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

    def validation_step(self, batch, batch_idx):
        output = self.forward(batch["input_ids"], return_loss=True)
        loss = output

        self.log("val_loss", loss, prog_bar=True, batch_size=len(batch["input_ids"]))
        return loss

    def on_train_epoch_start(self):
        print()

    def test_step(self, batch, batch_idx):
        if not batch["input_ids"].is_cuda:
            batch["input_ids"].to(self.device)

        references = self.tokenizer.batch_decode(
            batch["input_ids"],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        predictions = self.generate(
            batch["prompt_ids"],
        )

        print("reference", references[0], "", sep="\n")
        print("prediction", predictions[0], "", sep="\n")

        # Compute metrics and prepare results output
        results = []

        # F1 and Exact match
        for sample_id, reference, prediction in zip(
            batch["id"], references, predictions
        ):
            results.append(
                {
                    "id": sample_id,
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
            predictions=predictions, references=batch["model_input"]
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

    def generate_retro(
        self,
        start=None,
        retrieval=False,
        filter_fn=top_k,
        filter_thres=0.9,
        temperature=1.0,
    ):
        assert filter_fn in {
            top_k,
            top_p,
        }, "filter function must be either top-k or nucleus"

        # if not prime tokens given, assume sampling from SOS token with batch size of 1

        # Constants
        SOS_ID = self.tokenizer.pad_token_id
        EOS_ID = self.tokenizer.eos_token_id
        PAD_ID = self.tokenizer.pad_token_id
        chunk_size = self.config["chunk_size"]

        if not exists(start):
            start = torch.full(
                (1, 1),
                SOS_ID,
                dtype=torch.long,
                device=self.device,
            )

        b, start_seq_len = start.shape

        # move onto same device as RETRO

        start = start.to(self.device)

        # prepare retrieval related variables

        if retrieval and start_seq_len >= chunk_size:
            seq_index = (start_seq_len // chunk_size) * chunk_size
            past_seq_chunks = rearrange(
                start[:, :seq_index], "b (n c) -> (b n) c", c=chunk_size
            )

            retrieved = self.fetch_knn_chunks_fn(past_seq_chunks)
            retrieved = rearrange(retrieved, "(b n) k c -> b n k c", b=b)
        else:
            retrieved = None

        # get starting sequence index

        out = start

        # sampling loop

        for i in range(start_seq_len - 1, self.hparams.max_output_length):
            logits = self(out)
            logits = logits[:, i]

            logits = filter_fn(logits, thres=filter_thres)
            sampled = gumbel_sample(logits, temperature=temperature, dim=-1)
            sampled = rearrange(sampled, "b -> b 1")

            out = torch.cat((out, sampled), dim=1)

            # early terminate if all EOS

            is_eos_tokens = out == EOS_ID

            if is_eos_tokens.any(dim=-1).all():

                # mask out everything after the eos tokens

                shifted_is_eos_tokens = F.pad(is_eos_tokens, (1, -1))
                mask = shifted_is_eos_tokens.float().cumsum(dim=-1) >= 1
                out = out.masked_fill(mask, PAD_ID)
                break

            # when the sequence length is a multiple of the chunk size
            # retrieve the next set of knns

            curr_seq_len = out.shape[-1]

            if retrieval and (curr_seq_len % chunk_size) == 0:
                last_chunk = rearrange(out, "b (c n) -> b c n", n=chunk_size)[:, -1]

                knn_chunks = self.fetch_knn_chunks_fn(last_chunk)

                # concat retrieved knn chunks to all retrieved
                # to be sent to Retro for chunked cross attention at the next iteration

                knn_chunks = rearrange(knn_chunks, "b k r -> b 1 k r")
                retrieved = safe_cat(retrieved, knn_chunks, dim=1)

                print(f"retrieved at {curr_seq_len} / {self.max_seq_len}")

        return out

    def generate(self, input_ids, *args, **kargs):
        output = self.generate_retro(
            input_ids,
            *args,
            **kargs,
        )

        predictions = self.tokenizer.batch_decode(
            output,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        # Strip whitespaces
        predictions = [prediction.strip() for prediction in predictions]

        return predictions

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

    def tokenize(self, sample, truncation=True):
        max_input_length = self.hparams.max_input_length
        max_output_length = self.hparams.max_output_length

        # Disable long sequence warning during tokenization
        logging.getLogger("transformers.tokenization_utils_base").setLevel(
            logging.ERROR
        )

        # Tokenize model input into input_ids
        input_ids = self.tokenizer(
            sample,
            truncation=truncation,
            max_length=max_output_length if truncation else None,
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
            "chunk_size": args.chunk_size,
            "pad_id": tokenizer.pad_token_id,
            "norm_klass": RMSNorm,
        }

        retro = RETRO(**config)

        # Copy original model parameters
        if args.mode == "train":
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
                    retro.encoder.get_parameter(f"layers.{i}.0.fn.to_out.bias").fill_(
                        0.0
                    )
                    retro.encoder.get_parameter(f"layers.{i}.0.norm.gamma").copy_(
                        model.encoder.get_parameter(
                            f"block.{i}.layer.0.layer_norm.weight"
                        )
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
                        model.encoder.get_parameter(
                            f"block.{i}.layer.1.layer_norm.weight"
                        )
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
                    retro.decoder.get_parameter(f"layers.{i}.0.fn.to_out.bias").fill_(
                        0.0
                    )
                    retro.decoder.get_parameter(f"layers.{i}.0.norm.gamma").copy_(
                        model.decoder.get_parameter(
                            f"block.{i}.layer.0.layer_norm.weight"
                        )
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
                        model.decoder.get_parameter(
                            f"block.{i}.layer.2.layer_norm.weight"
                        )
                    )

        return config, tokenizer, retro


def parse_rouge_score(result):
    return {k: round(v.mid.fmeasure, 4) for k, v in result.items()}
