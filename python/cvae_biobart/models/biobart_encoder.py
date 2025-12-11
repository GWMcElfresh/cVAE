"""BioBART encoder/decoder for tokenizing, embedding, and generating text."""

from typing import List, Optional, Union

import numpy as np
import torch
from torch import nn
from transformers import BartForConditionalGeneration, AutoTokenizer


class BiobartEncoder(nn.Module):
    """
    BioBART encoder/decoder for text embeddings and generation.
    
    Uses BioBART to:
    1. Encode metadata text to embeddings (for VAE conditioning)
    2. Decode embeddings back to text (for interpretation)
    
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
        modelName: str = "GanjinZero/biobart-base",
        maxLength: int = 128,
        device: Optional[str] = None,
        freezeEncoder: bool = True,
    ):
        """
        Initialize BioBART encoder/decoder.
        
        Args:
            modelName: Name or path of the BioBART model to use
            maxLength: Maximum sequence length for tokenization
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
            freezeEncoder: Whether to freeze encoder parameters during training
        """
        super().__init__()
        
        self.modelName = modelName
        self.maxLength = maxLength
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Load full BioBART model (encoder + decoder)
        self._tokenizer = AutoTokenizer.from_pretrained(modelName)
        self._model = BartForConditionalGeneration.from_pretrained(modelName)
        self._freeze_encoder = freezeEncoder
        
        if freezeEncoder:
            for param in self._model.model.encoder.parameters():
                param.requires_grad = False
        
        self.embeddingDim = self._model.config.d_model  # 768 for base
        self._model.to(self.device)
    
    def Tokenize(
        self,
        texts: Union[str, List[str]],
    ) -> dict:
        """
        Tokenize input text(s) using BioBART tokenizer.
        
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
        Encode text(s) to embeddings using BioBART encoder.
        
        Args:
            texts: Single text string or list of text strings
            
        Returns:
            Tensor of shape (batch_size, embedding_dim) containing pooled embeddings
        """
        encoded = self.Tokenize(texts)
        
        with torch.set_grad_enabled(not self._freeze_encoder):
            encoder_outputs = self._model.model.encoder(**encoded)
        
        # Use mean pooling over non-padding tokens for embedding
        hidden_states = encoder_outputs.last_hidden_state
        attention_mask = encoded["attention_mask"].unsqueeze(-1)
        sum_hidden = (hidden_states * attention_mask).sum(dim=1)
        count = attention_mask.sum(dim=1)
        pooled = sum_hidden / count
        
        return pooled
    
    def GetEncoderHiddenStates(
        self,
        texts: Union[str, List[str]],
    ) -> torch.Tensor:
        """
        Get full encoder hidden states (for decoder cross-attention).
        
        Args:
            texts: Single text string or list of text strings
            
        Returns:
            Tensor of shape (batch_size, seq_len, embedding_dim)
        """
        encoded = self.Tokenize(texts)
        
        with torch.set_grad_enabled(not self._freeze_encoder):
            encoder_outputs = self._model.model.encoder(**encoded)
        
        return encoder_outputs.last_hidden_state
    
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
    
    def DecodeFromEmbeddings(
        self,
        embeddings: torch.Tensor,
        maxNewTokens: int = 50,
        topK: int = 50,
        topP: float = 0.92,
        temperature: float = 0.7,
        repetitionPenalty: float = 1.2,
        noRepeatNgramSize: int = 3,
        numSequencePositions: int = 8,
        doSample: bool = True,
    ) -> List[str]:
        """
        Generate text from embeddings using BioBART decoder.
        
        Uses top-k/top-p (nucleus) sampling for more diverse and natural generation.
        
        Note: Since we're passing a single pooled embedding vector (not a sequence
        of encoder hidden states), we expand it to multiple positions to give the
        decoder more context for cross-attention.
        
        Args:
            embeddings: Tensor of shape (batch_size, embedding_dim) or 
                       (batch_size, seq_len, embedding_dim)
            maxNewTokens: Maximum number of tokens to generate
            topK: Top-k sampling threshold (common: 50)
            topP: Top-p (nucleus) sampling threshold (common: 0.9-0.95)
            temperature: Sampling temperature (lower = more focused, common: 0.7-1.0)
            repetitionPenalty: Penalty for repeating tokens (>1.0 reduces repetition)
            noRepeatNgramSize: N-gram size to avoid repeating
            numSequencePositions: Number of positions to expand single embedding to
            doSample: Whether to use sampling (True) or greedy decoding (False)
            
        Returns:
            List of generated text strings
        """
        self.eval()
        
        # Ensure embeddings are on correct device
        embeddings = embeddings.to(self.device)
        
        # If embeddings are 2D (batch, dim), expand to 3D (batch, seq_len, dim)
        if embeddings.dim() == 2:
            # Expand single embedding to multiple positions for better cross-attention
            # This gives the decoder more "slots" to attend to
            embeddings = embeddings.unsqueeze(1).expand(-1, numSequencePositions, -1)
        
        batch_size = embeddings.shape[0]
        
        # Create decoder start tokens
        decoder_start_token_id = self._model.config.decoder_start_token_id
        if decoder_start_token_id is None:
            decoder_start_token_id = self._model.config.bos_token_id
        
        decoder_input_ids = torch.full(
            (batch_size, 1),
            decoder_start_token_id,
            dtype=torch.long,
            device=self.device
        )
        
        # Generate using the model with custom encoder outputs
        with torch.no_grad():
            # Create encoder outputs object
            from transformers.modeling_outputs import BaseModelOutput
            encoder_outputs = BaseModelOutput(last_hidden_state=embeddings)
            
            generated_ids = self._model.generate(
                encoder_outputs=encoder_outputs,
                decoder_input_ids=decoder_input_ids,
                max_new_tokens=maxNewTokens,
                # Top-k/Top-p (nucleus) sampling
                do_sample=doSample,
                top_k=topK,
                top_p=topP,
                temperature=temperature,
                # Repetition control
                repetition_penalty=repetitionPenalty,
                no_repeat_ngram_size=noRepeatNgramSize,
                # Token handling
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )
        
        # Decode to text
        generated_texts = self._tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )
        
        return generated_texts
    
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

