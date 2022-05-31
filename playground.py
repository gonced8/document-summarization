from retro_pytorch import RETRO
from retro_pytorch.retro_pytorch import RMSNorm
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")

config = {
    "num_tokens": model.config.vocab_size,
    "max_seq_len": 256,
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
# 0 1 2 3 4 5 6* 7 8 9* 10 11 12*

# Copy T5 parameters
with torch.no_grad():
    # Embeddings
    retro.token_emb.weight.copy_(model.get_input_embeddings().weight)

    # TODO: fc layers have bias but t5 doesn't

    # Copy encoder
    for i in range(config["enc_depth"]):
        # Attention
        retro.encoder.get_parameter(f"layers.{i}.0.fn.to_q.weight").copy_(
            model.encoder.get_parameter(f"block.{i}.layer.0.SelfAttention.q.weight")
        )
        retro.encoder.get_parameter(f"layers.{i}.0.fn.to_k.weight").copy_(
            model.encoder.get_parameter(f"block.{i}.layer.0.SelfAttention.k.weight")
        )
        retro.encoder.get_parameter(f"layers.{i}.0.fn.to_v.weight").copy_(
            model.encoder.get_parameter(f"block.{i}.layer.0.SelfAttention.v.weight")
        )
        retro.encoder.get_parameter(f"layers.{i}.0.fn.to_out.weight").copy_(
            model.encoder.get_parameter(f"block.{i}.layer.0.SelfAttention.o.weight")
        )
        retro.encoder.get_parameter(f"layers.{i}.0.fn.to_out.bias").fill_(0.0)
        retro.encoder.get_parameter(f"layers.{i}.0.norm.gamma").copy_(
            model.encoder.get_parameter(f"block.{i}.layer.0.layer_norm.weight")
        )

        # Fully-connected
        retro.encoder.get_parameter(f"layers.{i}.2.fn.ff.0.weight").copy_(
            model.encoder.get_parameter(f"block.{i}.layer.1.DenseReluDense.wi.weight")
        )
        retro.encoder.get_parameter(f"layers.{i}.2.fn.ff.0.bias").fill_(0.0)
        retro.encoder.get_parameter(f"layers.{i}.2.fn.ff.3.weight").copy_(
            model.encoder.get_parameter(f"block.{i}.layer.1.DenseReluDense.wo.weight")
        )
        retro.encoder.get_parameter(f"layers.{i}.2.fn.ff.3.bias").fill_(0.0)
        retro.encoder.get_parameter(f"layers.{i}.2.norm.gamma").copy_(
            model.encoder.get_parameter(f"block.{i}.layer.1.layer_norm.weight")
        )

    # Copy decoder
    for i in range(config["dec_depth"]):
        # Attention
        retro.decoder.get_parameter(f"layers.{i}.0.fn.to_q.weight").copy_(
            model.decoder.get_parameter(f"block.{i}.layer.0.SelfAttention.q.weight")
        )
        retro.decoder.get_parameter(f"layers.{i}.0.fn.to_k.weight").copy_(
            model.decoder.get_parameter(f"block.{i}.layer.0.SelfAttention.k.weight")
        )
        retro.decoder.get_parameter(f"layers.{i}.0.fn.to_v.weight").copy_(
            model.decoder.get_parameter(f"block.{i}.layer.0.SelfAttention.v.weight")
        )
        retro.decoder.get_parameter(f"layers.{i}.0.fn.to_out.weight").copy_(
            model.decoder.get_parameter(f"block.{i}.layer.0.SelfAttention.o.weight")
        )
        retro.decoder.get_parameter(f"layers.{i}.0.fn.to_out.bias").fill_(0.0)
        retro.decoder.get_parameter(f"layers.{i}.0.norm.gamma").copy_(
            model.decoder.get_parameter(f"block.{i}.layer.0.layer_norm.weight")
        )

        # Fully-connected
        retro.decoder.get_parameter(f"layers.{i}.2.fn.ff.0.weight").copy_(
            model.decoder.get_parameter(f"block.{i}.layer.2.DenseReluDense.wi.weight")
        )
        retro.decoder.get_parameter(f"layers.{i}.2.fn.ff.0.bias").fill_(0.0)
        retro.decoder.get_parameter(f"layers.{i}.2.fn.ff.3.weight").copy_(
            model.decoder.get_parameter(f"block.{i}.layer.2.DenseReluDense.wo.weight")
        )
        retro.decoder.get_parameter(f"layers.{i}.2.fn.ff.3.bias").fill_(0.0)
        retro.decoder.get_parameter(f"layers.{i}.2.norm.gamma").copy_(
            model.decoder.get_parameter(f"block.{i}.layer.2.layer_norm.weight")
        )
