# Arabic Language Modeling Comparison

This repository implements and compares different architectures for Arabic language modeling, including traditional N-gram models, LSTM-based RNNs, Transformers with Flash Attention, and Mamba State Space Models (SSM). It uses the Arabic Wikipedia dataset (6.1 GB), the same in [this Kaggle notebook](https://www.kaggle.com/code/abedkhooli/arabic-bert-ak) for training Arabic BERT models. The project is devided into 2 parts:
1. Embedding Notebook
2. Models Notebook


## 1. Embedding Notebook

The notebook includes two approaches for generating word embeddings for Arabic text:

1. **Custom Skip-Gram Model:**
   - Built using PyTorch with a focus on flexible tokenization and training pipelines.
   - Multi-layer architecture for refining embeddings.

2. **Gensim Word2Vec:**
   - Pre-built functionalities for efficient embedding generation.
   - Preferred for its superior performance and reduced computation time.

### Key Notes:
- Arabic tokenization is handled via the BERT WordPiece tokenizer.
- Gensim embeddings were ultimately adopted for better quality and sampling.


## 2. Models Notebook

This section describes four Arabic language models with the same BERT tokenizer, word embeddings and ~14.7M (excluding embeddings) parameters for fair comparison:

1. **N-gram Model**
   - Bigram (N=2) with basic smoothing.
   - Serves as a baseline traditional statistical approach.

2. **LSTM-based RNN**
   - 7-layer architecture with word2vec embeddings.
   - Uses shared embedding layers initialized with pre-trained vectors.

3. **Transformer with Flash Attention**
   - 6 transformer layers with Flash Attention for efficiency.
   - Expansion ratio: 2.66.

4. **Mamba State Space Model (SSM)**
   - 6 Mamba layers with a state dimension of 64.
   - Convolution width: 4, expansion factor: 3.




## Configuration

```python
CONTEXT_SIZE = 4
EMBEDDING_DIM = 512
MAX_LENGTH = 64
BATCH_SIZE = 256
TRAIN_SPLIT = 0.9
```

## Training Notes

### General Setup
- AdamW optimizer
- Linear learning rate schedule
- Weight decay: 0.002
- Training/Validation split: 90/10
- Sequence length: 64 with stride 32 (effective 2x epochs)

### Model-Specific Notes

#### Transformer
- Uses Flash Attention for efficient training
- Learning rate: 7e-4 with embedding lr = main_lr * 0.2

#### Mamba
- Requires lower learning rate (1e-4) for stability
- Uses embedding scaling (sqrt(dim) * 2)
- Embedding learning rate = main_lr * 0.2
- Without these adjustments, training can result in NaN losses

### Implementation Challenges
   - Parallel file processing with encode_batch caused stability issues
   - Resolution: Set num_workers=0 and enable TOKENIZERS_PARALLELISM
   - Possible memory/thread contention with encode_batch method



## Results

| Model       | Perplexity |
|------------|------------|
| N-gram     | 690,866,710.01 |
| RNN        | 95.09 |
| Transformer| 93.87 |
| Mamba      | 94.16 |

**Training Observations**
   - Transformer showed more natural text generation
   - RNN and Mamba occasionally produced repetitive punctuation
   - All NN models achieved similar perplexity scores


## Dependencies

- PyTorch
- tokenizers
- gensim
- mamba-ssm
- tqdm
- numpy

## Usage

Check the Jupyter notebook for detailed implementation and training procedures.

## Citations

If you use this code, please cite:
```
@misc{arabic_language_modeling,
  author = {NaifMersal},
  title = {Arabic Language Modeling},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/NaifMersal/Arabic-Language-Modeling},
}

```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
