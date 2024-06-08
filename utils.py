import os
import platform

import torch


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
