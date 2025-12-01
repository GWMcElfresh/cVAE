# cVAE-BioBERT

A hybrid R/Python package that combines Conditional Variational Autoencoders (cVAE) with BioBERT embeddings for gene expression analysis of Seurat objects.

## Overview

This package provides:
- **BioBERT Integration**: Tokenizes metadata and sample descriptions from Seurat object columns
- **Conditional VAE**: Conditions gene expression VAE on BioBERT embeddings
- **Seurat Integration**: Seamlessly works with pre-processed Seurat objects
- **Hybrid Architecture**: R interface with Python machine learning backend

## Installation

### Python Package

```bash
cd python
pip install -e .
```

### R Package

```r
# Install from source
install.packages("R", repos = NULL, type = "source")

# Or use devtools
devtools::install("R")
```

## Quick Start

### R Usage (Recommended)

```r
library(cvaeBiobert)

# Initialize Python environment
InitPythonEnv()

# Run cVAE on Seurat object
# By default, uses Seurat's normalized layer (scale.data)
seuratObj <- RunCvaeBiobert(
  seuratObj,
  metadataColumns = c("celltype", "condition", "sample_description"),
  latentDim = 32,
  epochs = 100
)

# Get embeddings
embeddings <- GetCvaeEmbeddings(seuratObj)

# Get training metadata
metadata <- GetCvaeMetadata(seuratObj)
```

### Python Usage (Direct)

```python
from cvae_biobert import CvaeBiobert

# Initialize model
model = CvaeBiobert(
    geneCount=2000,
    latentDim=32,
    conditionDim=768
)

# Prepare data
# geneMatrix: numpy array (n_samples, n_genes)
# metadataTexts: list of text strings for conditioning

# Train and transform
embeddings = model.FitTransform(geneMatrix, metadataTexts, epochs=100)
```

## Architecture

```
Metadata Text       Gene Expression
      │                    │
      ▼                    │
  ┌─────────┐              │
  │ BioBERT │              │
  │ Encoder │              │
  └────┬────┘              │
       │                   │
       ▼                   ▼
   Condition ──────► cVAE Encoder
   Vector                  │
       │                   ▼
       │              Latent Space (z)
       │                   │
       ▼                   ▼
   Condition ──────► cVAE Decoder
   Vector                  │
                           ▼
                   Reconstructed
                   Gene Expression
```

## Naming Conventions

Following the specified conventions:
- **Functions**: PascalCase (e.g., `RunCvaeBiobert`, `InitPythonEnv`)
- **Public variables**: camelCase (e.g., `geneCount`, `latentDim`)
- **Private variables**: snake_case (e.g., `_hidden_dims`, `_encoder`)

## Package Structure

```
cVAE/
├── python/
│   ├── cvae_biobert/
│   │   ├── models/
│   │   │   ├── biobert_encoder.py  # BioBERT tokenization
│   │   │   └── cvae.py             # cVAE model
│   │   └── utils/
│   │       └── data_utils.py       # Data preparation
│   └── tests/
│       └── test_cvae.py
├── R/
│   ├── R/
│   │   ├── init.R          # Python environment setup
│   │   ├── prepare_data.R  # Seurat data preparation
│   │   └── run_cvae.R      # Main cVAE functions
│   └── tests/
│       └── testthat/
└── README.md
```

## Requirements

### Python
- Python >= 3.9
- PyTorch >= 2.0.0
- Transformers >= 4.30.0
- NumPy, Pandas, SciPy, scikit-learn

### R
- R >= 4.0.0
- reticulate >= 1.28
- Seurat >= 4.0.0
- SeuratObject >= 4.0.0

## License

MIT