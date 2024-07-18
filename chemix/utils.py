import torch

_LARGE_NUM = torch.finfo(torch.float32).max
_LARGENEG_NUM = torch.finfo(torch.float32).min


def masked_sum(x, mask):
    x_masked = torch.einsum('ijk, ij -> ijk', x, mask)
    return torch.einsum('ijk->ik', x_masked)


def masked_mean(x, mask):
    x_masked = torch.einsum('ijk, ij -> ijk', x, mask)
    counts = mask.sum(dim=1, keepdim=True)
    mean = torch.einsum('ijk->ik', x_masked) / counts
    return mean


def masked_variance(x, mask):
    # Assumes ddf=0
    x_masked = torch.einsum('ijk, ij -> ijk', x, mask)
    counts = mask.sum(dim=1, keepdim=True)
    mean = torch.einsum('ijk->ik', x_masked) / counts
    squared_diff = (x - mean.unsqueeze(1))**2
    diff_masked = torch.einsum('ijk, ij -> ijk', squared_diff, mask)
    var = torch.einsum('ijk->ik', diff_masked) / counts
    return var


def masked_min(x, mask):
    x_masked = torch.where(mask.unsqueeze(-1), x, _LARGE_NUM)
    min_values, _ = torch.min(x_masked, dim=1)
    return min_values


def masked_max(x, mask):
    x_masked = torch.where(mask.unsqueeze(-1), x, _LARGENEG_NUM)
    min_values, _ = torch.max(x_masked, dim=1)
    return min_values