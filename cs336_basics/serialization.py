import os
import torch
from typing import Union, BinaryIO, IO


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: Union[str, os.PathLike, BinaryIO, IO[bytes]]
):
    """
    Save model, optimizer, and iteration state to a checkpoint file.
    
    Args:
        model (torch.nn.Module): The model to serialize.
        optimizer (torch.optim.Optimizer): The optimizer to serialize.
        iteration (int): The current iteration number.
        out (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to save to.
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration
    }
    torch.save(checkpoint, out)


def load_checkpoint(
    src: Union[str, os.PathLike, BinaryIO, IO[bytes]],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer
) -> int:
    """
    Load model, optimizer, and iteration state from a checkpoint file.
    
    Args:
        src (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to load from.
        model (torch.nn.Module): The model to restore state to.
        optimizer (torch.optim.Optimizer): The optimizer to restore state to.
        
    Returns:
        int: The iteration number that was saved in the checkpoint.
    """
    checkpoint = torch.load(src, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint['iteration']
