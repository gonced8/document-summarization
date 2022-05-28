import json

from retro_pytorch.retrieval import doc_text_to_chunks_and_seq_indices

filename = "data/arxiv/documents/0704.0021.json"

with open(filename, "r") as f:
    data = json.load(f)

text = data["text"]

chunk_size = 64
seq_len = 2048

chunks, seq = doc_text_to_chunks_and_seq_indices(
    doc_text=text, chunk_size=chunk_size, seq_len=seq_len
)

print(chunks.shape)
print(seq.shape)
print(seq)
