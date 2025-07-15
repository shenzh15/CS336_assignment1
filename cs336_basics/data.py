import numpy as np
import torch
from typing import Tuple
import numpy.typing as npt


def get_batch(
    dataset: npt.NDArray, 
    batch_size: int, 
    context_length: int, 
    device: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample language modeling batches from a tokenized dataset.
    
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.
    
    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.
    
    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels (input sequences shifted right by 1).
    """
    # Ensure we have enough data for sampling
    if len(dataset) < context_length + 1:
        raise ValueError(
            f"Dataset length ({len(dataset)}) must be at least context_length + 1 ({context_length + 1})"
        )
    max_start_idx = len(dataset) - context_length
    start_indices = np.random.randint(0, max_start_idx, size=batch_size)
    input_sequences = np.zeros((batch_size, context_length), dtype=dataset.dtype)
    target_sequences = np.zeros((batch_size, context_length), dtype=dataset.dtype)
    for i, start_idx in enumerate(start_indices):
        input_sequences[i] = dataset[start_idx:start_idx + context_length]
        target_sequences[i] = dataset[start_idx + 1:start_idx + context_length + 1]
    input_tensor = torch.from_numpy(input_sequences).long().to(device)
    target_tensor = torch.from_numpy(target_sequences).long().to(device)
    return input_tensor, target_tensor
