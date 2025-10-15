import torch



def normalize_to_range(tensor, range_min=-1, range_max=1):
    """
    Normalize a tensor to a specific range [range_min, range_max].
    
    Args:
        tensor (torch.Tensor): Input tensor to normalize.
        range_min (float): Minimum value of the target range (default: -1).
        range_max (float): Maximum value of the target range (default: 1).
    
    Returns:
        torch.Tensor: Normalized tensor.
    """
    x_min = torch.amin(tensor, dim=None)
    x_max = torch.amax(tensor, dim=None)

    if x_max == x_min:
        return torch.full_like(tensor, range_min)  # All values are the same
    else:
        return range_min + (tensor - x_min) * (range_max - range_min) / (x_max - x_min)
