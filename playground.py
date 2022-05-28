from retro_pytorch import RETRO
from transformers import AutoTokenizer, AutoModelWithLMHead

tokenizer = AutoTokenizer.from_pretrained("t5-base")
model = AutoModelWithLMHead.from_pretrained("t5-base")

encoder = model.get_encoder()
decoder = model.get_decoder()

config = model.config
config = {
    "num_tokens": config["vocab_size"],
    "max_seq_len": config["task_specific_params"]["summarization"]["max_length"],
    "enc_dim": config["d_model"],
    "enc_depth": config["num_layers"],
    "enc_cross_attn_layers": None,
    "dec_depth": config["num_decoder_layers"]
    + 1
    + (config["num_decoder_layers"] - 6) // 3,
    "dec_cross_attn_layers": (
        6 + i for i in range(1 + (config["num_decoder_layers"] - 6) // 3)
    ),
    "heads": config["num_heads"],
    "dec_dim": config["d_model"],
    "dim_head": config["d_kv"],
    "enc_attn_dropout": 0,
    "enc_ff_dropout": 0,
    "dec_attn_dropout": 0,
    "dec_ff_dropout": 0,
    "chunk_size": 64,
    "pad_id": 0,
    "enc_scale_residual": None,
    "dec_scale_residual": None,
    "nom_klass": None,
    "gated_rmsnorm": False,
    "use_deepnet": False,
}

# 0 1 2 3 4 5 C0 6 7 8 C1 9 10 11 C2
