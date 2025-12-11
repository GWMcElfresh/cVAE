"""
Data utilities for preparing gene expression data and Seurat objects.

This module provides functions for preparing data from Seurat objects
for use with the cVAE-BioBART model.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


def PrepareGeneData(
    geneMatrix: np.ndarray,
    normalizeMethod: str = "log1p",
    scaleToUnit: bool = True,
    isNormalized: bool = False,
) -> np.ndarray:
    """
    Prepare gene expression data for VAE input.
    
    Prefers to use Seurat's normalized layer (scale.data) directly when available.
    
    Args:
        geneMatrix: Gene expression matrix (samples x genes)
        normalizeMethod: Normalization method ('log1p', 'zscore', 'none')
        scaleToUnit: Whether to scale values to [0, 1] range
        isNormalized: If True, data is already normalized (e.g., from Seurat's scale.data)
                     and normalization steps are skipped
        
    Returns:
        Processed gene expression matrix
    """
    _data = geneMatrix.astype(np.float32).copy()
    
    # Skip normalization if data is already normalized (e.g., from Seurat's scale.data)
    if not isNormalized:
        if normalizeMethod == "log1p":
            # Log normalize: log(1 + x)
            _data = np.log1p(_data)
        elif normalizeMethod == "zscore":
            # Z-score normalize per gene
            _mean = np.mean(_data, axis=0, keepdims=True)
            _std = np.std(_data, axis=0, keepdims=True)
            _std[_std == 0] = 1.0  # Avoid division by zero
            _data = (_data - _mean) / _std
        elif normalizeMethod != "none":
            raise ValueError(f"Unknown normalization method: {normalizeMethod}")
    
    if scaleToUnit and not isNormalized:
        # Scale to [0, 1] per gene
        # Note: Scaling is skipped when isNormalized=True because data from
        # Seurat's scale.data is already normalized and scaled
        _min_val = np.min(_data, axis=0, keepdims=True)
        _max_val = np.max(_data, axis=0, keepdims=True)
        _range = _max_val - _min_val
        _range[_range == 0] = 1.0  # Avoid division by zero
        _data = (_data - _min_val) / _range
    
    return _data


def PrepareSeuratData(
    assayData: np.ndarray,
    metadata: pd.DataFrame,
    textColumns: Optional[List[str]] = None,
    separator: str = " | ",
    normalizeMethod: str = "log1p",
    scaleToUnit: bool = True,
    isNormalized: bool = False,
) -> Tuple[np.ndarray, List[str]]:
    """
    Prepare data from Seurat object components for cVAE-BioBART.
    
    This function is called from R via reticulate to process Seurat
    object data for the Python model. Prefers to use Seurat's normalized
    layer (scale.data) directly when available.
    
    Args:
        assayData: Gene expression matrix from Seurat assay (samples x genes)
        metadata: DataFrame containing sample metadata
        textColumns: List of metadata column names to use for text conditioning.
                    If None, uses all string/object columns.
        separator: Separator between metadata fields in combined text
        normalizeMethod: Gene expression normalization method
        scaleToUnit: Whether to scale gene expression to [0, 1]
        isNormalized: If True, data is already normalized (e.g., from Seurat's scale.data)
        
    Returns:
        Tuple of (processed_gene_matrix, metadata_texts)
    """
    # Process gene expression
    processedGenes = PrepareGeneData(
        assayData,
        normalizeMethod=normalizeMethod,
        scaleToUnit=scaleToUnit,
        isNormalized=isNormalized,
    )
    
    # Determine which columns to use for text
    if textColumns is None:
        # Use all string/object columns
        textColumns = [
            col for col in metadata.columns
            if metadata[col].dtype == object or str(metadata[col].dtype) == "string"
        ]
    
    # Validate columns exist
    _missing = [col for col in textColumns if col not in metadata.columns]
    if _missing:
        raise ValueError(f"Missing metadata columns: {_missing}")
    
    # Build metadata text strings
    metadataTexts = []
    for idx in range(len(metadata)):
        parts = []
        for col in textColumns:
            val = metadata[col].iloc[idx]
            if pd.notna(val) and str(val).strip():
                parts.append(f"{col}: {val}")
        
        text = separator.join(parts) if parts else "unknown sample"
        metadataTexts.append(text)
    
    return processedGenes, metadataTexts


def BuildMetadataText(
    metadata: pd.DataFrame,
    textColumns: List[str],
    separator: str = " | ",
) -> List[str]:
    """
    Build metadata text strings from DataFrame.
    
    Args:
        metadata: DataFrame containing sample metadata
        textColumns: List of column names to include
        separator: Separator between fields
        
    Returns:
        List of combined text strings
    """
    texts = []
    for idx in range(len(metadata)):
        parts = []
        for col in textColumns:
            if col in metadata.columns:
                val = metadata[col].iloc[idx]
                if pd.notna(val) and str(val).strip():
                    parts.append(f"{col}: {val}")
        
        text = separator.join(parts) if parts else "unknown sample"
        texts.append(text)
    
    return texts


def FormatResults(
    latentEmbeddings: np.ndarray,
    reconstructedExpression: Optional[np.ndarray] = None,
    trainingHistory: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Format cVAE results for return to R.
    
    Args:
        latentEmbeddings: Latent space embeddings (samples x latent_dim)
        reconstructedExpression: Reconstructed gene expression (optional)
        trainingHistory: Training history dictionary (optional)
        
    Returns:
        Dictionary formatted for R consumption
    """
    results = {
        "embeddings": latentEmbeddings,
        "nSamples": latentEmbeddings.shape[0],
        "latentDim": latentEmbeddings.shape[1],
    }
    
    if reconstructedExpression is not None:
        results["reconstructed"] = reconstructedExpression
    
    if trainingHistory is not None:
        results["history"] = trainingHistory
    
    return results

