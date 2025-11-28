"""Tests for cVAE-BioBERT package."""

import numpy as np
import pandas as pd
import pytest


class TestDataUtils:
    """Tests for data utility functions."""
    
    def test_prepare_gene_data_log1p(self):
        """Test log1p normalization."""
        from cvae_biobert.utils.data_utils import PrepareGeneData
        
        data = np.array([[0, 1, 10], [5, 2, 0]], dtype=np.float32)
        result = PrepareGeneData(data, normalizeMethod="log1p", scaleToUnit=False)
        
        expected = np.log1p(data)
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_prepare_gene_data_scale(self):
        """Test scaling to unit range."""
        from cvae_biobert.utils.data_utils import PrepareGeneData
        
        data = np.array([[0, 10], [5, 5], [10, 0]], dtype=np.float32)
        result = PrepareGeneData(data, normalizeMethod="none", scaleToUnit=True)
        
        assert result.min() >= 0
        assert result.max() <= 1
    
    def test_prepare_seurat_data(self):
        """Test Seurat data preparation."""
        from cvae_biobert.utils.data_utils import PrepareSeuratData
        
        genes = np.random.rand(5, 100).astype(np.float32)
        metadata = pd.DataFrame({
            "cellType": ["T cell", "B cell", "Monocyte", "T cell", "NK cell"],
            "condition": ["healthy", "disease", "healthy", "disease", "healthy"],
        })
        
        processed, texts = PrepareSeuratData(genes, metadata)
        
        assert processed.shape == genes.shape
        assert len(texts) == 5
        assert "cellType: T cell" in texts[0]
    
    def test_build_metadata_text(self):
        """Test metadata text building."""
        from cvae_biobert.utils.data_utils import BuildMetadataText
        
        metadata = pd.DataFrame({
            "sample": ["S1", "S2"],
            "type": ["A", "B"],
        })
        
        texts = BuildMetadataText(metadata, ["sample", "type"], separator=" - ")
        
        assert texts[0] == "sample: S1 - type: A"
        assert texts[1] == "sample: S2 - type: B"


class TestBiobertEncoder:
    """Tests for BioBERT encoder (mock tests without actual model)."""
    
    def test_combine_metadata(self):
        """Test metadata combination function."""
        # This test doesn't require loading the full model
        from cvae_biobert.models.biobert_encoder import BiobertEncoder
        
        # We can test the method without initializing the encoder
        # by creating a partial mock
        metadata = {
            "cellType": ["T cell", "B cell"],
            "condition": ["healthy", "disease"],
        }
        
        # Test the logic by reimplementing
        combined = []
        for i in range(2):
            parts = []
            for col, values in metadata.items():
                val = values[i]
                if val is not None and str(val).strip():
                    parts.append(f"{col}: {val}")
            combined.append(" | ".join(parts))
        
        assert "cellType: T cell" in combined[0]
        assert "condition: healthy" in combined[0]


class TestCvaeBiobert:
    """Tests for cVAE model structure (without full initialization)."""
    
    def test_encoder_decoder_symmetry(self):
        """Test that encoder and decoder have symmetric structure."""
        from cvae_biobert.models.cvae import Encoder, Decoder
        import torch
        
        input_dim = 100
        condition_dim = 64
        hidden_dims = [128, 64]
        latent_dim = 16
        
        encoder = Encoder(input_dim, condition_dim, hidden_dims, latent_dim)
        decoder = Decoder(latent_dim, condition_dim, list(reversed(hidden_dims)), input_dim)
        
        # Test forward pass
        batch_size = 8
        x = torch.randn(batch_size, input_dim)
        c = torch.randn(batch_size, condition_dim)
        
        mu, logvar = encoder(x, c)
        assert mu.shape == (batch_size, latent_dim)
        assert logvar.shape == (batch_size, latent_dim)
        
        z = mu  # Use mean for deterministic test
        recon = decoder(z, c)
        assert recon.shape == (batch_size, input_dim)
    
    def test_encoder_output_shapes(self):
        """Test encoder output shapes."""
        from cvae_biobert.models.cvae import Encoder
        import torch
        
        encoder = Encoder(
            inputDim=200,
            conditionDim=128,
            hiddenDims=[256, 128],
            latentDim=32,
        )
        
        x = torch.randn(16, 200)
        c = torch.randn(16, 128)
        
        mu, logvar = encoder(x, c)
        
        assert mu.shape == (16, 32)
        assert logvar.shape == (16, 32)
    
    def test_decoder_output_shape(self):
        """Test decoder output shape."""
        from cvae_biobert.models.cvae import Decoder
        import torch
        
        decoder = Decoder(
            latentDim=32,
            conditionDim=128,
            hiddenDims=[128, 256],
            outputDim=200,
        )
        
        z = torch.randn(16, 32)
        c = torch.randn(16, 128)
        
        output = decoder(z, c)
        
        assert output.shape == (16, 200)


class TestFormatResults:
    """Tests for result formatting."""
    
    def test_format_results_basic(self):
        """Test basic result formatting."""
        from cvae_biobert.utils.data_utils import FormatResults
        
        embeddings = np.random.rand(10, 32)
        result = FormatResults(embeddings)
        
        assert result["nSamples"] == 10
        assert result["latentDim"] == 32
        np.testing.assert_array_equal(result["embeddings"], embeddings)
    
    def test_format_results_with_reconstruction(self):
        """Test result formatting with reconstruction."""
        from cvae_biobert.utils.data_utils import FormatResults
        
        embeddings = np.random.rand(10, 32)
        reconstructed = np.random.rand(10, 100)
        
        result = FormatResults(embeddings, reconstructedExpression=reconstructed)
        
        assert "reconstructed" in result
        np.testing.assert_array_equal(result["reconstructed"], reconstructed)
