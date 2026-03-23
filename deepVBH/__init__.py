__all__ = [
    "DeepVBHE3Model",
    "DeepVBHBaseline",
    "DeepVBHStepDataset",
]


def __getattr__(name: str):
    if name == "DeepVBHStepDataset":
        from .dataset import DeepVBHStepDataset

        return DeepVBHStepDataset
    if name == "DeepVBHE3Model":
        from .e3nn_model import DeepVBHE3Model

        return DeepVBHE3Model
    if name == "DeepVBHBaseline":
        from .model import DeepVBHBaseline

        return DeepVBHBaseline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
