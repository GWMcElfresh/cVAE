"""BioBERT encoder for tokenizing and embedding metadata text."""

from typing import List, Optional, Union

import numpy as np
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer


class BiobertEncoder(nn.Module):
    """
    BioBERT encoder for converting text metadata to embeddings.
    
    Uses BioBERT to tokenize and encode metadata text from Seurat object
    columns and sample descriptions into dense vector representations.
    
    Naming convention:
    - Public methods: PascalCase
    - Public attributes: camelCase  
    - Private attributes: snake_case
    """
    
    # Public attributes (camelCase)
    modelName: str
    embeddingDim: int
    maxLength: int
    device: torch.device
    
    def __init__(
        self,
        modelName: str = "dmis-lab/biobert-base-cased-v1.2",
        maxLength: int = 512,
        device: Optional[str] = None,
        freezeEncoder: bool = True,
    ):
        """
        Initialize BioBERT encoder.
        
        Args:
            modelName: Name or path of the BioBERT model to use
            maxLength: Maximum sequence length for tokenization
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
            freezeEncoder: Whether to freeze BioBERT parameters during training
        """
        super().__init__()
        
        self.modelName = modelName
        self.maxLength = maxLength
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Private attributes (snake_case)
        self._tokenizer = AutoTokenizer.from_pretrained(modelName)
        self._encoder = AutoModel.from_pretrained(modelName)
        self._freeze_encoder = freezeEncoder
        
        if freezeEncoder:
            for param in self._encoder.parameters():
                param.requires_grad = False
        
        self.embeddingDim = self._encoder.config.hidden_size
        self._encoder.to(self.device)
    
    def Tokenize(
        self,
        texts: Union[str, List[str]],
    ) -> dict:
        """
        Tokenize input text(s) using BioBERT tokenizer.
        
        Args:
            texts: Single text string or list of text strings
            
        Returns:
            Dictionary containing tokenized inputs
        """
        if isinstance(texts, str):
            texts = [texts]
        
        encoded = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.maxLength,
            return_tensors="pt",
        )
        
        return {k: v.to(self.device) for k, v in encoded.items()}
    
    def forward(
        self,
        texts: Union[str, List[str]],
    ) -> torch.Tensor:
        """
        Encode text(s) to embeddings using BioBERT.
        
        Args:
            texts: Single text string or list of text strings
            
        Returns:
            Tensor of shape (batch_size, embedding_dim) containing CLS embeddings
        """
        encoded = self.Tokenize(texts)
        
        with torch.set_grad_enabled(not self._freeze_encoder):
            outputs = self._encoder(**encoded)
        
        # Use CLS token embedding as the text representation
        cls_embeddings = outputs.last_hidden_state[:, 0, :]
        
        return cls_embeddings
    
    def Encode(
        self,
        texts: Union[str, List[str]],
    ) -> np.ndarray:
        """
        Encode text(s) to numpy array embeddings.
        
        Args:
            texts: Single text string or list of text strings
            
        Returns:
            Numpy array of shape (batch_size, embedding_dim)
        """
        self.eval()
        with torch.no_grad():
            embeddings = self.forward(texts)
        
        return embeddings.cpu().numpy()
    
    def CombineMetadata(
        self,
        metadataColumns: dict,
        separator: str = " | ",
    ) -> List[str]:
        """
        Combine multiple metadata columns into single text strings.
        
        Args:
            metadataColumns: Dictionary mapping column names to lists of values
            separator: String to use between column values
            
        Returns:
            List of combined text strings, one per sample
        """
        if not metadataColumns:
            raise ValueError("metadataColumns cannot be empty")
        
        # Get number of samples from first column
        _first_key = next(iter(metadataColumns))
        _n_samples = len(metadataColumns[_first_key])
        
        combined_texts = []
        for i in range(_n_samples):
            parts = []
            for col_name, values in metadataColumns.items():
                val = values[i]
                if val is not None and str(val).strip():
                    parts.append(f"{col_name}: {val}")
            
            combined_text = separator.join(parts)
            combined_texts.append(combined_text if combined_text else "unknown sample")
        
        return combined_texts
