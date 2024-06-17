import os
import platform
from typing import Optional

import torch
from transformers import PreTrainedTokenizerBase


def is_apple_silicon():
    if platform.system() == 'Darwin' and os.uname().machine == 'arm64':
        return True
    return False


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif is_apple_silicon():
        return torch.device('mps')
    return torch.device('cpu')


def test_accuracy(outputs, labels):
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == labels).sum().item()
    return correct / len(predicted)


def get_word_embeddings(text: str,
                        tokenizer: PreTrainedTokenizerBase,
                        model: torch.nn.Module,
                        device: Optional[torch.device]) -> torch.Tensor:
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    if device is not None:
        inputs.to(device)
    outputs = model(**inputs)
    embedding = outputs.last_hidden_state[-1].squeeze()
    return embedding.mean(axis=0)
