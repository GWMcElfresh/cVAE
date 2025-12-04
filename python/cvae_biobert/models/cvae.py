"""
Conditional Variational Autoencoder (cVAE) with BioBERT conditioning.

This module implements a cVAE for gene expression data that is conditioned
on BioBERT embeddings of metadata text.
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from cvae_biobert.models.biobert_encoder import BiobertEncoder


class Encoder(nn.Module):
    """
    Encoder network for the cVAE.
    
    Maps gene expression data + condition to latent space parameters (mean, log_var).
    """
    
    def __init__(
        self,
        inputDim: int,
        conditionDim: int,
        hiddenDims: List[int],
        latentDim: int,
        dropoutRate: float = 0.1,
    ):
        """
        Initialize encoder.
        
        Args:
            inputDim: Dimension of gene expression input
            conditionDim: Dimension of condition vector (BioBERT embedding)
            hiddenDims: List of hidden layer dimensions
            latentDim: Dimension of latent space
            dropoutRate: Dropout rate for regularization
        """
        super().__init__()
        
        self.inputDim = inputDim
        self.conditionDim = conditionDim
        self.latentDim = latentDim
        
        # Build encoder layers
        _layers = []
        _in_dim = inputDim + conditionDim
        
        for _h_dim in hiddenDims:
            _layers.extend([
                nn.Linear(_in_dim, _h_dim),
                nn.BatchNorm1d(_h_dim),
                nn.ReLU(),
                nn.Dropout(dropoutRate),
            ])
            _in_dim = _h_dim
        
        self._encoder = nn.Sequential(*_layers)
        
        # Latent space projections
        self._fc_mu = nn.Linear(hiddenDims[-1], latentDim)
        self._fc_logvar = nn.Linear(hiddenDims[-1], latentDim)
    
    def forward(
        self,
        x: torch.Tensor,
        condition: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input to latent space parameters.
        
        Args:
            x: Gene expression tensor of shape (batch_size, input_dim)
            condition: Condition tensor of shape (batch_size, condition_dim)
            
        Returns:
            Tuple of (mu, log_var) tensors for the latent distribution
        """
        # Concatenate input with condition
        _combined = torch.cat([x, condition], dim=1)
        
        _hidden = self._encoder(_combined)
        
        mu = self._fc_mu(_hidden)
        logVar = self._fc_logvar(_hidden)
        
        return mu, logVar


class Decoder(nn.Module):
    """
    Decoder network for the cVAE.
    
    Maps latent space + condition back to gene expression space.
    """
    
    def __init__(
        self,
        latentDim: int,
        conditionDim: int,
        hiddenDims: List[int],
        outputDim: int,
        dropoutRate: float = 0.1,
    ):
        """
        Initialize decoder.
        
        Args:
            latentDim: Dimension of latent space
            conditionDim: Dimension of condition vector
            hiddenDims: List of hidden layer dimensions (reversed from encoder)
            outputDim: Dimension of gene expression output
            dropoutRate: Dropout rate for regularization
        """
        super().__init__()
        
        self.latentDim = latentDim
        self.conditionDim = conditionDim
        self.outputDim = outputDim
        
        # Build decoder layers
        _layers = []
        _in_dim = latentDim + conditionDim
        
        for _h_dim in hiddenDims:
            _layers.extend([
                nn.Linear(_in_dim, _h_dim),
                nn.BatchNorm1d(_h_dim),
                nn.ReLU(),
                nn.Dropout(dropoutRate),
            ])
            _in_dim = _h_dim
        
        self._decoder = nn.Sequential(*_layers)
        
        # Output layer
        self._output = nn.Linear(hiddenDims[-1], outputDim)
    
    def forward(
        self,
        z: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        """
        Decode latent representation to gene expression.
        
        Args:
            z: Latent tensor of shape (batch_size, latent_dim)
            condition: Condition tensor of shape (batch_size, condition_dim)
            
        Returns:
            Reconstructed gene expression tensor
        """
        _combined = torch.cat([z, condition], dim=1)
        _hidden = self._decoder(_combined)
        output = self._output(_hidden)
        
        return output


class CvaeBiobert(nn.Module):
    """
    Conditional Variational Autoencoder with BioBERT conditioning.
    
    This model combines a cVAE for gene expression data with BioBERT
    embeddings of metadata text as the conditioning signal.
    
    Naming convention:
    - Public methods: PascalCase
    - Public attributes: camelCase
    - Private attributes: snake_case
    """
    
    # Public attributes (camelCase)
    geneCount: int
    latentDim: int
    conditionDim: int
    device: torch.device
    
    def __init__(
        self,
        geneCount: int,
        latentDim: int = 32,
        hiddenDims: Optional[List[int]] = None,
        conditionDim: int = 768,
        biobertModel: str = "dmis-lab/biobert-base-cased-v1.2",
        dropoutRate: float = 0.1,
        device: Optional[str] = None,
        freezeBiobert: bool = True,
    ):
        """
        Initialize cVAE-BioBERT model.
        
        Args:
            geneCount: Number of genes in expression matrix
            latentDim: Dimension of VAE latent space
            hiddenDims: List of hidden layer dimensions (default: [512, 256, 128])
            conditionDim: Dimension of BioBERT condition embeddings
            biobertModel: Name or path of BioBERT model
            dropoutRate: Dropout rate for regularization
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
            freezeBiobert: Whether to freeze BioBERT parameters
        """
        super().__init__()
        
        self.geneCount = geneCount
        self.latentDim = latentDim
        self.conditionDim = conditionDim
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        if hiddenDims is None:
            hiddenDims = [512, 256, 128]
        
        # Private attributes (snake_case)
        self._hidden_dims = hiddenDims
        self._dropout_rate = dropoutRate
        
        # BioBERT encoder
        self._biobert = BiobertEncoder(
            modelName=biobertModel,
            device=str(self.device),
            freezeEncoder=freezeBiobert,
        )
        
        # Projection layer if BioBERT embedding dim != conditionDim
        if self._biobert.embeddingDim != conditionDim:
            self._condition_proj = nn.Linear(
                self._biobert.embeddingDim,
                conditionDim,
            )
        else:
            self._condition_proj = nn.Identity()
        
        # VAE components
        self._encoder = Encoder(
            inputDim=geneCount,
            conditionDim=conditionDim,
            hiddenDims=hiddenDims,
            latentDim=latentDim,
            dropoutRate=dropoutRate,
        )
        
        self._decoder = Decoder(
            latentDim=latentDim,
            conditionDim=conditionDim,
            hiddenDims=list(reversed(hiddenDims)),
            outputDim=geneCount,
            dropoutRate=dropoutRate,
        )
        
        self._is_fitted = False
        self._training_history: List[Dict] = []
        
        self.to(self.device)
    
    def _reparameterize(
        self,
        mu: torch.Tensor,
        logVar: torch.Tensor,
    ) -> torch.Tensor:
        """
        Reparameterization trick for sampling from latent distribution.
        
        Args:
            mu: Mean of latent distribution
            logVar: Log variance of latent distribution
            
        Returns:
            Sampled latent vector
        """
        std = torch.exp(0.5 * logVar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def _compute_loss(
        self,
        x: torch.Tensor,
        recon: torch.Tensor,
        mu: torch.Tensor,
        logVar: torch.Tensor,
        beta: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute VAE loss (reconstruction + KL divergence).
        
        Args:
            x: Original input
            recon: Reconstructed input
            mu: Latent mean
            logVar: Latent log variance
            beta: Weight for KL divergence term
            
        Returns:
            Tuple of (total_loss, recon_loss, kl_loss)
        """
        # Reconstruction loss (MSE)
        _recon_loss = nn.functional.mse_loss(recon, x, reduction="mean")
        
        # KL divergence loss
        _kl_loss = -0.5 * torch.mean(
            1 + logVar - mu.pow(2) - logVar.exp()
        )
        
        _total_loss = _recon_loss + beta * _kl_loss
        
        return _total_loss, _recon_loss, _kl_loss
    
    def forward(
        self,
        geneExpression: torch.Tensor,
        condition: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the cVAE.
        
        Args:
            geneExpression: Gene expression tensor (batch_size, gene_count)
            condition: Pre-computed condition tensor (batch_size, condition_dim)
            
        Returns:
            Tuple of (reconstructed, mu, log_var)
        """
        mu, logVar = self._encoder(geneExpression, condition)
        z = self._reparameterize(mu, logVar)
        reconstructed = self._decoder(z, condition)
        
        return reconstructed, mu, logVar
    
    def EncodeCondition(
        self,
        metadataTexts: Union[str, List[str]],
    ) -> torch.Tensor:
        """
        Encode metadata text to condition vectors using BioBERT.
        
        Args:
            metadataTexts: Single text string or list of text strings
            
        Returns:
            Condition tensor of shape (batch_size, condition_dim)
        """
        biobert_emb = self._biobert(metadataTexts)
        condition = self._condition_proj(biobert_emb)
        return condition
    
    def Fit(
        self,
        geneMatrix: np.ndarray,
        metadataTexts: List[str],
        epochs: int = 100,
        batchSize: int = 32,
        learningRate: float = 1e-3,
        beta: float = 1.0,
        betaWarmup: int = 10,
        validationSplit: float = 0.1,
        verbose: bool = True,
    ) -> Dict:
        """
        Train the cVAE-BioBERT model.
        
        Args:
            geneMatrix: Gene expression matrix (n_samples, n_genes)
            metadataTexts: List of metadata text strings for conditioning
            epochs: Number of training epochs
            batchSize: Training batch size
            learningRate: Learning rate for optimizer
            beta: Final weight for KL divergence term
            betaWarmup: Number of epochs for beta warmup
            validationSplit: Fraction of data for validation
            verbose: Whether to show progress bar
            
        Returns:
            Dictionary containing training history
        """
        if geneMatrix.shape[1] != self.geneCount:
            raise ValueError(
                f"Expected {self.geneCount} genes, got {geneMatrix.shape[1]}"
            )
        
        if len(metadataTexts) != geneMatrix.shape[0]:
            raise ValueError(
                f"Number of texts ({len(metadataTexts)}) must match "
                f"number of samples ({geneMatrix.shape[0]})"
            )
        
        # Convert data to tensors
        _gene_tensor = torch.tensor(geneMatrix, dtype=torch.float32)
        
        # Pre-compute BioBERT embeddings for efficiency
        self._biobert.eval()
        with torch.no_grad():
            _condition_tensor = self.EncodeCondition(metadataTexts)
        _condition_tensor = _condition_tensor.detach()
        
        # Split into train/validation
        _n_samples = geneMatrix.shape[0]
        _n_val = int(_n_samples * validationSplit)
        _indices = torch.randperm(_n_samples)
        
        _train_idx = _indices[_n_val:]
        _val_idx = _indices[:_n_val]
        
        _train_genes = _gene_tensor[_train_idx].to(self.device)
        _train_cond = _condition_tensor[_train_idx]
        
        _train_dataset = TensorDataset(_train_genes, _train_cond)
        _train_loader = DataLoader(
            _train_dataset,
            batch_size=batchSize,
            shuffle=True,
        )
        
        if _n_val > 0:
            _val_genes = _gene_tensor[_val_idx].to(self.device)
            _val_cond = _condition_tensor[_val_idx]
        
        # Optimizer
        _optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=learningRate,
        )
        
        # Training loop
        self._training_history = []
        
        _epoch_iter = tqdm(range(epochs), desc="Training") if verbose else range(epochs)
        
        for epoch in _epoch_iter:
            self.train()
            
            # Beta warmup
            _current_beta = beta * min(1.0, (epoch + 1) / betaWarmup)
            
            _epoch_loss = 0.0
            _epoch_recon = 0.0
            _epoch_kl = 0.0
            _n_batches = 0
            
            for _batch_genes, _batch_cond in _train_loader:
                _optimizer.zero_grad()
                
                _recon, _mu, _logvar = self.forward(_batch_genes, _batch_cond)
                _loss, _recon_loss, _kl_loss = self._compute_loss(
                    _batch_genes, _recon, _mu, _logvar, _current_beta
                )
                
                _loss.backward()
                _optimizer.step()
                
                _epoch_loss += _loss.item()
                _epoch_recon += _recon_loss.item()
                _epoch_kl += _kl_loss.item()
                _n_batches += 1
            
            _avg_loss = _epoch_loss / _n_batches
            _avg_recon = _epoch_recon / _n_batches
            _avg_kl = _epoch_kl / _n_batches
            
            # Validation
            _val_loss = None
            if _n_val > 0:
                self.eval()
                with torch.no_grad():
                    _recon, _mu, _logvar = self.forward(_val_genes, _val_cond)
                    _val_loss, _, _ = self._compute_loss(
                        _val_genes, _recon, _mu, _logvar, _current_beta
                    )
                    _val_loss = _val_loss.item()
            
            self._training_history.append({
                "epoch": epoch + 1,
                "trainLoss": _avg_loss,
                "trainReconLoss": _avg_recon,
                "trainKlLoss": _avg_kl,
                "valLoss": _val_loss,
                "beta": _current_beta,
            })
            
            if verbose:
                _desc = f"Loss: {_avg_loss:.4f}"
                if _val_loss is not None:
                    _desc += f" | Val: {_val_loss:.4f}"
                _epoch_iter.set_postfix_str(_desc)
        
        self._is_fitted = True
        
        return {
            "history": self._training_history,
            "finalTrainLoss": self._training_history[-1]["trainLoss"],
            "finalValLoss": self._training_history[-1]["valLoss"],
        }
    
    def Transform(
        self,
        geneMatrix: np.ndarray,
        metadataTexts: List[str],
    ) -> np.ndarray:
        """
        Transform gene expression data to latent space.
        
        Args:
            geneMatrix: Gene expression matrix (n_samples, n_genes)
            metadataTexts: List of metadata text strings
            
        Returns:
            Latent embeddings as numpy array (n_samples, latent_dim)
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before transform")
        
        self.eval()
        
        _gene_tensor = torch.tensor(geneMatrix, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            _condition = self.EncodeCondition(metadataTexts)
            _mu, _ = self._encoder(_gene_tensor, _condition)
        
        return _mu.cpu().numpy()
    
    def FitTransform(
        self,
        geneMatrix: np.ndarray,
        metadataTexts: List[str],
        **fitKwargs,
    ) -> np.ndarray:
        """
        Fit the model and transform data in one step.
        
        Args:
            geneMatrix: Gene expression matrix (n_samples, n_genes)
            metadataTexts: List of metadata text strings
            **fitKwargs: Additional arguments for Fit method
            
        Returns:
            Latent embeddings as numpy array
        """
        self.Fit(geneMatrix, metadataTexts, **fitKwargs)
        return self.Transform(geneMatrix, metadataTexts)
    
    def Reconstruct(
        self,
        geneMatrix: np.ndarray,
        metadataTexts: List[str],
    ) -> np.ndarray:
        """
        Reconstruct gene expression through the VAE.
        
        Args:
            geneMatrix: Gene expression matrix (n_samples, n_genes)
            metadataTexts: List of metadata text strings
            
        Returns:
            Reconstructed gene expression as numpy array
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before reconstruction")
        
        self.eval()
        
        _gene_tensor = torch.tensor(geneMatrix, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            _condition = self.EncodeCondition(metadataTexts)
            _recon, _, _ = self.forward(_gene_tensor, _condition)
        
        return _recon.cpu().numpy()
    
    def SaveModel(self, path: str) -> None:
        """
        Save model state to file.
        
        Args:
            path: Path to save model
        """
        torch.save({
            "model_state_dict": self.state_dict(),
            "config": {
                "geneCount": self.geneCount,
                "latentDim": self.latentDim,
                "conditionDim": self.conditionDim,
                "hiddenDims": self._hidden_dims,
                "dropoutRate": self._dropout_rate,
            },
            "is_fitted": self._is_fitted,
            "training_history": self._training_history,
        }, path)
    
    @classmethod
    def LoadModel(cls, path: str, device: Optional[str] = None) -> "CvaeBiobert":
        """
        Load model from file.
        
        Args:
            path: Path to saved model
            device: Device to load model to
            
        Returns:
            Loaded CvaeBiobert model
        """
        checkpoint = torch.load(path, map_location=device)
        config = checkpoint["config"]
        
        model = cls(
            geneCount=config["geneCount"],
            latentDim=config["latentDim"],
            conditionDim=config["conditionDim"],
            hiddenDims=config["hiddenDims"],
            dropoutRate=config["dropoutRate"],
            device=device,
        )
        
        model.load_state_dict(checkpoint["model_state_dict"])
        model._is_fitted = checkpoint["is_fitted"]
        model._training_history = checkpoint["training_history"]
        
        return model
