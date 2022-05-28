Introduction

With the growth of publicly available text data, the summarization of such contents is essential for their usefulness. A text summary must convey important information from the original text and present a smaller, more manageable, size. The task of automatic text summarization produces a concise and fluent text summary while preserving key information and overall meaning.

Approaches to automatic text summarization can be divided into extractive and abstractive summarization. While the extractive approach produces a summary that is comprised entirely of excerpts from the original text, the abstractive approach generates an output that may contain content that is entirely original. Both approaches have seen significant improvements in recent years by using models based on the Transformer architecture. In particular, the fluency of these language models has allowed for state-of-the-art results for abstractive summarization.

However, Transformers' quadratic memory and time complexities with respect to the sequence length make them very expensive to use, especially with long sequences, as required by document-level summarization. Recent approaches explore different attention mechanisms that are able to reduce the quadratic cost, allowing to process longer sequences. Additionally, retrieval-enhanced language models exhibit useful memorization qualities while being more efficient than plain models. Although less explored, retrieval has been used to enhance an abstractive summarization model, improving its performance.

Our work will address the problem of document-level summarization by studying how the aforementioned techniques can be used to improve the automatic summarization of very long texts. In particular, we will use the arXiv dataset, consisting of several scientific papers and the corresponding abstracts. The results obtained with Efficient Transformers will be reproduced and used as baselines. Then, we propose a novel retrieval-enhanced approach based on the architecture which reduces the cost of generating a summary of the entire document by processing smaller chunks.


Related Work

The Transformer architecture introduced in 2017 established, within sequence modeling, an alternative to. In fact, by processing sentences as a whole using attention mechanisms and positional embeddings, Transformers avoid processing the input recurrently, facilitating parallelization as well as handling long-context dependencies. 

Long document summarization

Since most common Transformer models are pretrained for inputs of 256-1024 tokens, and fine-tuning them for longer sizes is computationally expensive, they seem unsuitable for the task of summarizing entire documents. However, three different approaches to the standard Transformer that allow for long-document summarization have been proposed: 1) divide-and-conquer, 2) hierarchical attention mechanisms, and 3) sparse attention mechanisms.

The first approach builds upon the idea that long-document summarization can be decomposed into shorter summarization problems, in which the task is tackled in a section-wise manner. Considering that manually adapting training data to accommodate this methodology would not be feasible, Gidiotis and Tsoumakas designed a method to enable training in such a manner: rather than manually summarizing each section of the document, the process is performed automatically using Divide-ANd-ConquER (DANCER). This methodology is used to create artificial pairs of sections and abstract segments for training, which are applied to a well-known encoder-decoder Transformer architecture, PEGASUS. Although this approach makes the model generalizable to theoretically infinite documents, it fails to incorporate context from the other sections of the document. Furthermore, it does not manage duplicate information when the set of summaries is concatenated.

Hierarchical attention, first introduced in the context of sequence classification, explores the ambivalent relevance of each token according to the context they are in. The hierarchical attention mechanism incorporates two levels of attention mechanisms, one at the sequence level and another at the word level. As such, the first level can identify which sequences of tokens (within a sentence) are potentially relevant, significantly limiting the number of individual tokens that need to be processed by the second level (full attention pattern). This mechanism was transposed to long document summarization by Rohde et al. with state-of-the-art results, although for input sequences limited to approximately 3k (due to memory constraints). 

Finally, sparse attention mechanisms directly tackle the issue of time and memory quadratic complexity with sequence length. Instead of using a full attention pattern, primacy is given to the local context (local attention window), while also incorporating some global attention elements that provide access to the global context. This sparsity approach provides a considerable context of the full sequence while significantly decreasing complexity. Beltagy et al. and Zaheer et al. propose drop-in replacements for the standard attention mechanisms, reporting results for the standard Transformer and PEGASUS architectures, respectively. Similarly, Guo et al. extends the original T5 architecture with an attention sparsity pattern, applied to the encoder layer only. 

While all approaches achieved state-of-the-art performances on the arXiv dataset, not all models are designed to handle the same input length, as illustrated in Table. Considering shorter input lengths as a limitation for the specific task of document-length summarization, the LongT5 approach proposed in reports the most satisfactory results in both domains (performance and input length).  

Summarization datasets

Guo et al. showcased six datasets for text summarization. These datasets can be divided into two groups: the first, constituted by the CNN/Daily Mail, MediaSum, and Multi-News datasets, relates to news articles and media sources; the second, constituted by the PubMed, arXiv, and BigPatent datasets, relates to scientific and technical documents. Naturally, the first includes shorter documents, with an average input length of 1,797 tokens, while the second group includes longer documents, averaging 6,931 tokens (obtained with the SentencePiece tokenizer).

Guo et al. gather the summarization results of many state-of-the-art models, which are presented in Table, along with a few details of the models. These are evaluated using the automatic metric and considered baselines for this work.


Efficient Transformer

A review of state-of-the-art approaches (Section) indicates that Transformer-based models with sparse attention mechanisms are particularly well-suited for the task of summarizing long sequences. Given the notable results reported by Guo et al., our work focuses on the LongT5 model.

The LongT5 model aims to tackle the issue of the quadratic complexity of traditional attention mechanisms. The proposed approach uses a mechanism as an alternative to the attention pattern of the original T5 encoder architecture. As illustrated in Figure, this pattern gives primacy to neighboring tokens (through the use of a sliding window) while, at the same time, incorporating global context through a set of dynamically constructed global tokens (Figure). This effectively reduces the time and memory complexity of input encoding from O(n x n) to O(n x (r + n/k)) (where n is the input length, r is the width of the local window, and n/k is number of global tokens). Since the output size in a document summarization is considerably more manageable than its input size, this attention mechanism is not as important for the decoder component, therefore, LongT5 simply leverages the original decoder from T5.

Document-level summarization using LongT5

When applied to the task of document-level summarization using the arXiv dataset, the input of the Transformer will be the entire document text (excluding everything before the Introduction and after the Conclusion) and the ground truth summary will be the paper Abstract -- similar to the illustration in Figure.

As a first approach, a pretrained implementation of the LongT5 (LongT5-TGlobal-Large - 16k input) was fine-tuned with the aforementioned arXiv dataset. 


Retrieval-Enhanced Approach for Summarization

Instead of relying only on learned weights for memorization, combining neural networks with explicit memories (e.g., through retrieval from a repository) is a possible way to decrease the number of model parameters while obtaining comparable performance. Historically, information retrieval was performed using bag-of-words representations and functions like TF-IDF and BM25. More recently, neural models trained to encode text into dense representations are able to capture implicit semantics, with retrieval methods exploring these representations in dual-encoder or cross-encoder settings. 

One example of coupling an external memory with a neural model for text generation is the kNN-LM, which builds a key-value database of context-token pairs and calculates the next-token probability by interpolating a Transformer with a distribution calculated from the retrieved k nearest neighbors.
RAG combines inputs and text retrieved using a dual-encoder, feeding both to a decoder for generation. FiD assumes a similar approach, scaling better to larger numbers of retrieved passages.
Combining kNN-LM, DPR, and FiD, RETRO retrieves chunks of text (neighbors) whose dense representations are then processed independently in an encoder, and attended in a operation in a decoder. By processing the input in chunks, RETRO avoids computing the quadratic attention over the entire document, by computing it only over the chunks that the retrieval component considered relevant.

Our proposed approach is to use a RETRO-based model to generate a document summary, retrieving from a set of chunks obtained only from that document. Without retrieval, the decoder would generate a summary-like text from a given prompt (e.g., paper title) -- Figure. However, the generated text would be very imprecise/incorrect since the decoder would not have any information besides the prompt. With retrieval, chunks of the generated text are used to sequentially retrieve neighbors from the document text, which are encoded and attended to in the operation in the decoder -- Figure. With this approach, the decoder will be able to incorporate the information from the document during the generation of the summary. Figure illustrates three different approaches to the task of document summarization of a scientific paper.

RETRO-fitting a baseline model

Although a RETRO-based model could be trained from scratch for abstractive summarization, extending baseline models into RETRO models offers a more efficient alternative -- RETRO-fitting. Starting from a pretrained Transformer, it is augmented with nearest-neighbor retrieval, a neighbor encoder, and chunked cross-attention layers. During training, all parameters are frozen except the neighbor encoder and the chunked cross-attention, ensuring that the original model performance is maintained without retrieval.

Since RETRO works with chunks of a fixed size, the RETRO-fitting implementation is simpler if the pretrained Transformer utilizes the same tokenizer as the encoder used for the nearest-neighbor search, such that the number of tokens is the same throughout. In the original paper, RETRO tokenizes the dataset using SentencePiece, but performs nearest-neighbor search using, which was originally implemented using WordPiece tokenization. Thus, we assume that the authors pretrained a-like model using SentencePiece and, consequently, our design will have some differences.

In detail, our implementation uses the encoder and decoder models of a pretrained T5 model. As in the original paper, it starts by tokenizing the dataset using SentencePiece and making up chunks of 64 tokens. Then, the nearest-neighbor search is performed using a frozen Sentence-T5 encoder to compute dense vectors (d=768) for each chunk of text. The nearest neighbors are retrieved using AutoFaiss. The retrieved neighbors are encoded using the T5 encoder and attended to in a T5 decoder augmented with chunked cross-attention layers, implemented using the (unofficial) RETRO - Pytorch library. As for the chunked cross-attention operations, these are introduced in every 3\textsuperscript{rd} layer, starting from 6, of the 12-layer T5 decoder.


Experiments

Experimental Setup

We focus on the arXiv dataset, which consists of scientific papers from the corresponding repository. Being scientific papers, these documents follow a common structure: initial description of the problem, methodology, experiments/results, and conclusions. A publicly available compilation of 215K docs was curated by Cohan et al. and was used in this work. In this compilation, each paper entry is represented in a JSON object with the following elements: article id, abstract text, article text, section names, and sections. Some dataset statistics are shown in Table. 

To automatically evaluate the summarization performance, we use the ROUGE-1, ROUGE-2, and ROUGE-L metrics. These work by measuring the overlap of n-grams between the generated and reference summaries. Since automatic metrics often do not correlate well with human judgment, we also use BERTScore, which exploits pretrained models to measure semantic equivalence.

