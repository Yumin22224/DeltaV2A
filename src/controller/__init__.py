"""Audio controller for DSP parameter prediction."""
from .model import AudioController
from .trainer import ControllerTrainer, train_controller

__all__ = ["AudioController", "ControllerTrainer", "train_controller"]
