"""
cVAE-BioBART: Conditional VAE with BioBART embeddings for gene expression analysis.

This package provides a conditional variational autoencoder that uses BioBART
to encode metadata and sample descriptions, conditioning the VAE on these
embeddings for gene expression analysis, with text generation capability.
"""

from cvae_biobart.models.cvae import CvaeBiobert
from cvae_biobart.models.biobart_encoder import BiobartEncoder
from cvae_biobart.utils.data_utils import PrepareGeneData, PrepareSeuratData

__version__ = "0.2.0"
__all__ = [
    "CvaeBiobert",
    "BiobartEncoder",
    "PrepareGeneData",
    "PrepareSeuratData",
]

