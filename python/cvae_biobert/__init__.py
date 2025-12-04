"""
cVAE-BioBERT: Conditional VAE with BioBERT embeddings for gene expression analysis.

This package provides a conditional variational autoencoder that uses BioBERT
to encode metadata and sample descriptions, conditioning the VAE on these
embeddings for gene expression analysis.
"""

from cvae_biobert.models.cvae import CvaeBiobert
from cvae_biobert.models.biobert_encoder import BiobertEncoder
from cvae_biobert.utils.data_utils import PrepareGeneData, PrepareSeuratData

__version__ = "0.1.0"
__all__ = [
    "CvaeBiobert",
    "BiobertEncoder",
    "PrepareGeneData",
    "PrepareSeuratData",
]
