"""
Shared ESM model loader to avoid redundant model loading across different SPIRED tools.

This module provides a singleton-style model manager that loads ESM models once
and reuses them across all three SPIRED APIs (run_spired, run_spired_stab, run_spired_fitness).

Model Requirements by Tool:
- run_spired (structure prediction): ESM2-650M, ESM2-3B, ESM1v (all 5)
- run_spired_stab (stability prediction): ESM2-650M, ESM2-3B only (NO ESM1v)
- run_spired_fitness (fitness landscape): ESM2-650M, ESM2-3B, ESM1v (all 5)
"""

import torch
from typing import Dict, Any, Tuple
from loguru import logger


class ESMModelManager:
    """Singleton manager for ESM models to avoid redundant loading."""

    _instance = None
    _esm2_loaded = False
    _esm1v_loaded = False
    _models: Dict[str, Any] = {}
    _current_device = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ESMModelManager, cls).__new__(cls)
        return cls._instance

    def _load_esm2_models(self, device: str) -> None:
        """Load ESM-2 models (650M and 3B)."""
        if self._esm2_loaded and self._current_device == device:
            logger.info(f"Reusing already loaded ESM2 models on {device}")
            return

        logger.info(f"Loading ESM2 models on device: {device}")

        # Load ESM-2 650M model
        logger.info("Loading ESM-2 650M model...")
        esm2_650M, _ = torch.hub.load("facebookresearch/esm:main", "esm2_t33_650M_UR50D")
        esm2_650M = esm2_650M.to(device)
        esm2_650M.eval()
        self._models["esm2_650M"] = esm2_650M
        logger.info("ESM-2 650M model loaded")

        # Load ESM-2 3B model
        logger.info("Loading ESM-2 3B model...")
        esm2_3B, esm2_alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t36_3B_UR50D")
        esm2_3B = esm2_3B.to(device)
        esm2_3B.eval()
        esm2_batch_converter = esm2_alphabet.get_batch_converter()
        self._models["esm2_3B"] = esm2_3B
        self._models["esm2_alphabet"] = esm2_alphabet
        self._models["esm2_batch_converter"] = esm2_batch_converter
        logger.info("ESM-2 3B model loaded")

        self._esm2_loaded = True
        self._current_device = device

    def _load_esm1v_models(self, device: str) -> None:
        """Load ESM-1v models (all 5)."""
        if self._esm1v_loaded and self._current_device == device:
            logger.info(f"Reusing already loaded ESM1v models on {device}")
            return

        logger.info("Loading 5 ESM-1v models...")
        esm1v_1, _ = torch.hub.load("facebookresearch/esm:main", "esm1v_t33_650M_UR90S_1")
        esm1v_2, _ = torch.hub.load("facebookresearch/esm:main", "esm1v_t33_650M_UR90S_2")
        esm1v_3, _ = torch.hub.load("facebookresearch/esm:main", "esm1v_t33_650M_UR90S_3")
        esm1v_4, _ = torch.hub.load("facebookresearch/esm:main", "esm1v_t33_650M_UR90S_4")
        esm1v_5, esm1v_alphabet = torch.hub.load("facebookresearch/esm:main", "esm1v_t33_650M_UR90S_5")

        esm1v_1 = esm1v_1.to(device)
        esm1v_2 = esm1v_2.to(device)
        esm1v_3 = esm1v_3.to(device)
        esm1v_4 = esm1v_4.to(device)
        esm1v_5 = esm1v_5.to(device)

        esm1v_1.eval()
        esm1v_2.eval()
        esm1v_3.eval()
        esm1v_4.eval()
        esm1v_5.eval()

        esm1v_batch_converter = esm1v_alphabet.get_batch_converter()

        self._models["esm1v_1"] = esm1v_1
        self._models["esm1v_2"] = esm1v_2
        self._models["esm1v_3"] = esm1v_3
        self._models["esm1v_4"] = esm1v_4
        self._models["esm1v_5"] = esm1v_5
        self._models["esm1v_alphabet"] = esm1v_alphabet
        self._models["esm1v_batch_converter"] = esm1v_batch_converter

        self._esm1v_loaded = True
        logger.info("All 5 ESM-1v models loaded")

    def _normalize_device(self, device: str) -> str:
        """Normalize device string and verify CUDA availability."""
        if device.startswith("cuda"):
            # Extract device index if present, default to cuda:0
            if ":" not in device:
                device = "cuda:0"
            # Verify CUDA is available
            if not torch.cuda.is_available():
                logger.warning(f"CUDA device '{device}' requested but CUDA not available, falling back to CPU")
                device = "cpu"
        return device

    def _check_device_change(self, device: str) -> None:
        """Check if device changed and clear models if needed."""
        if self._current_device is not None and self._current_device != device:
            logger.info(f"Device changed from {self._current_device} to {device}, clearing cached models")
            self._models.clear()
            self._esm2_loaded = False
            self._esm1v_loaded = False
            self._current_device = None

    def get_models_for_spired(self, device: str = "cuda:0") -> Tuple:
        """
        Get models needed for run_spired (structure prediction).

        Requires: ESM2-650M, ESM2-3B, ESM1v (all 5)
        """
        device = self._normalize_device(device)
        self._check_device_change(device)

        logger.info("Loading models for SPIRED structure prediction (ESM2 + ESM1v)")
        self._load_esm2_models(device)
        self._load_esm1v_models(device)

        return (
            self._models["esm2_3B"],
            self._models["esm2_650M"],
            self._models["esm1v_1"],
            self._models["esm1v_2"],
            self._models["esm1v_3"],
            self._models["esm1v_4"],
            self._models["esm1v_5"],
            self._models["esm1v_batch_converter"],
            self._models["esm1v_alphabet"],
            self._models["esm2_batch_converter"],
        )

    def get_models_for_spired_stab(self, device: str = "cuda:0") -> Tuple:
        """
        Get models needed for run_spired_stab (stability prediction).

        Requires: ESM2-650M, ESM2-3B only (NO ESM1v needed - saves memory!)
        """
        device = self._normalize_device(device)
        self._check_device_change(device)

        logger.info("Loading models for SPIRED stability prediction (ESM2 only, skipping ESM1v)")
        self._load_esm2_models(device)
        # NOTE: ESM1v models are NOT loaded for stability prediction

        return (
            self._models["esm2_3B"],
            self._models["esm2_650M"],
            self._models["esm2_batch_converter"],
        )

    def get_models_for_spired_fitness(self, device: str = "cuda:0") -> Tuple:
        """
        Get models needed for run_spired_fitness (fitness landscape).

        Requires: ESM2-650M, ESM2-3B, ESM1v (all 5)
        """
        device = self._normalize_device(device)
        self._check_device_change(device)

        logger.info("Loading models for SPIRED fitness landscape prediction (ESM2 + ESM1v)")
        self._load_esm2_models(device)
        self._load_esm1v_models(device)

        return (
            self._models["esm2_3B"],
            self._models["esm2_650M"],
            self._models["esm1v_1"],
            self._models["esm1v_2"],
            self._models["esm1v_3"],
            self._models["esm1v_4"],
            self._models["esm1v_5"],
            self._models["esm1v_batch_converter"],
            self._models["esm1v_alphabet"],
            self._models["esm2_batch_converter"],
        )

    @property
    def current_device(self) -> str:
        """Get the current device models are loaded on."""
        return self._current_device

    @property
    def is_esm2_loaded(self) -> bool:
        """Check if ESM2 models are loaded."""
        return self._esm2_loaded

    @property
    def is_esm1v_loaded(self) -> bool:
        """Check if ESM1v models are loaded."""
        return self._esm1v_loaded

    def get_loaded_models_info(self) -> str:
        """Get information about which models are loaded."""
        info = []
        if self._esm2_loaded:
            info.append("ESM2-650M, ESM2-3B")
        if self._esm1v_loaded:
            info.append("ESM1v (1-5)")
        if not info:
            return "No models loaded"
        return f"Loaded on {self._current_device}: " + ", ".join(info)


# Global instance
esm_manager = ESMModelManager()
