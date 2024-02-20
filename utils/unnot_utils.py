import torch


def torch_safe_division(a, b):
    c = torch.full_like(a, fill_value=0.0)
    mask = (b != 0)
    # finally perform division
    c[mask] = a[mask] / b[mask]
    return c

def cosine_sim(a, b):
    if len(a.shape) == 1:
        numer = (a * b).sum()
        denom = (torch.norm(a) * torch.norm(b))
    else:
        numer = (a * b).sum(dim=1)
        denom = (torch.norm(a, dim=1) * torch.norm(b, dim=1))
    return torch_safe_division(numer, denom)  
