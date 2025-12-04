# cVAE-BioBERT Python Package

Conditional Variational Autoencoder (cVAE) with BioBERT embeddings for gene expression analysis.

## Overview

This package implements a cVAE that uses BioBERT to encode metadata and sample descriptions, conditioning the gene expression VAE on these embeddings.

## Installation

```bash
pip install -e .
```

## Usage

```python
from cvae_biobert import CvaeBiobert

# Initialize the model
model = CvaeBiobert(
    geneCount=2000,
    latentDim=32,
    conditionDim=768
)

# Prepare data (from R via reticulate)
# geneMatrix: numpy array of shape (n_samples, n_genes)
# metadataTexts: list of text strings describing each sample

# Train the model
model.Fit(geneMatrix, metadataTexts, epochs=100)

# Get embeddings
embeddings = model.Transform(geneMatrix, metadataTexts)
```

## Architecture

1. **BioBERT Encoder**: Tokenizes and encodes metadata text into condition vectors
2. **cVAE Encoder**: Maps gene expression + condition to latent space
3. **cVAE Decoder**: Reconstructs gene expression from latent space + condition

## License

MIT
