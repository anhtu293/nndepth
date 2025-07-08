import os
from typing import Callable, Any, Optional
import torch.distributed as dist


def is_dist_initialized() -> bool:
    """
    Check if the training is distributed and initialized.
    """
    return dist.is_initialized()


def is_main_process() -> bool:
    """
    Check if the current process is the main process.
    """
    return (dist.is_initialized() and dist.get_rank() == 0) or not dist.is_initialized()


def get_rank() -> int:
    """
    Get the rank of the current process.
    """
    if not is_dist_initialized():
        return 0
    return dist.get_rank()


def run_on_main_process(func: Callable) -> Optional[Any]:
    """
    Run a function on the main process.
    """
    def _wrapper(*args, **kwargs):
        if is_main_process():
            return func(*args, **kwargs)
        else:
            return None
    return _wrapper
