"""
Model loading and management utilities for MCP scripts.

Provides lazy loading and caching of SPIRED-Stab models to minimize startup time.
"""

import sys
import torch
from pathlib import Path
from typing import Tuple, Any, Dict, Optional


class ModelCache:
    """Global model cache to avoid reloading models."""

    def __init__(self):
        self.cache = {}
        self.current_device = None

    def get_models(self, device: str = "cpu"):
        """Get cached models or load them if not cached."""
        if self.current_device == device and "models" in self.cache:
            return self.cache["models"]

        # Load models
        models = _load_spired_models(device)
        self.cache["models"] = models
        self.current_device = device
        return models

    def clear(self):
        """Clear the model cache."""
        self.cache.clear()
        self.current_device = None


# Global model cache instance
_model_cache = ModelCache()


def get_spired_models(device: str = "cpu") -> Tuple[Any, Any, Any, Any, Any, Any]:
    """
    Lazy load SPIRED-Stab models to minimize startup time.

    Args:
        device: Device to load models on (e.g., 'cuda:0', 'cpu')

    Returns:
        Tuple of (model, esm2_650M, esm2_3B, esm2_batch_converter, getStabDataTest, tqdm)

    Raises:
        ImportError: If models cannot be loaded
        FileNotFoundError: If model files are not found
    """
    return _model_cache.get_models(device)


def _load_spired_models(device: str = "cpu") -> Tuple[Any, Any, Any, Any, Any, Any]:
    """
    Internal function to load SPIRED-Stab models.

    Args:
        device: Device to load models on

    Returns:
        Tuple of loaded models and utilities
    """
    # Determine paths
    script_dir = Path(__file__).parent.parent
    project_root = script_dir.parent
    scripts_dir = project_root / "scripts"

    # Add scripts directory to path to import the model
    sys.path.insert(0, str(scripts_dir))

    try:
        from src.model import SPIRED_Stab
        from src.utils_train_valid import getStabDataTest
        import tqdm

        print(f"Loading SPIRED-Stab models on device: {device}")

        # Load SPIRED-Stab model
        model = SPIRED_Stab(device_list=[device, device, device, device])
        model_path = scripts_dir / "data" / "model" / "SPIRED-Stab.pth"

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        model.load_state_dict(torch.load(str(model_path), map_location=device))
        model.to(device)
        model.eval()
        print("✓ SPIRED-Stab model loaded")

        # Load ESM-2 650M model
        print("Loading ESM-2 650M model...")
        esm2_650M, _ = torch.hub.load('facebookresearch/esm:main', 'esm2_t33_650M_UR50D')
        esm2_650M.to(device)
        esm2_650M.eval()
        print("✓ ESM-2 650M model loaded")

        # Load ESM-2 3B model
        print("Loading ESM-2 3B model...")
        esm2_3B, esm2_alphabet = torch.hub.load('facebookresearch/esm:main', 'esm2_t36_3B_UR50D')
        esm2_3B.to(device)
        esm2_3B.eval()
        esm2_batch_converter = esm2_alphabet.get_batch_converter()
        print("✓ ESM-2 3B model loaded")

        print("All models loaded successfully!")
        return model, esm2_650M, esm2_3B, esm2_batch_converter, getStabDataTest, tqdm

    except ImportError as e:
        raise ImportError(f"Failed to import required modules. Make sure scripts directory is accessible: {e}")
    except Exception as e:
        raise ImportError(f"Failed to load models: {e}")


def clear_model_cache():
    """Clear the global model cache."""
    global _model_cache
    _model_cache.clear()


def check_model_files(scripts_dir: Optional[Path] = None) -> Dict[str, bool]:
    """
    Check if model files exist.

    Args:
        scripts_dir: Path to scripts directory (optional)

    Returns:
        Dictionary with model file existence status
    """
    if scripts_dir is None:
        script_dir = Path(__file__).parent.parent
        project_root = script_dir.parent
        scripts_dir = project_root / "scripts"

    model_path = scripts_dir / "data" / "model" / "SPIRED-Stab.pth"

    return {
        "spired_stab_model": model_path.exists(),
        "model_path": str(model_path),
        "scripts_dir_exists": scripts_dir.exists()
    }