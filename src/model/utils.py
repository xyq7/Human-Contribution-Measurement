import torch

def get_compute_capability():
    if not torch.cuda.is_available():
        raise ValueError("CUDA is not available on this device!")

    capability_str = torch.cuda.get_device_capability()
    capability = float(f"{capability_str[0]}.{capability_str[1]}")
    return capability

def check_bf16_support():
    capability = get_compute_capability()
    if capability >= 8.0:
        return True
    return False