import torch

def is_singular(A) -> bool:
    return A[0, 0] * A[1, 1] == A[0,1] * A[1,0]

def transform_matrix(A, T, S) -> torch.Tensor:
    """
    Perform the change-of-basis transform T⁻¹ A S and round to 3 decimals using PyTorch.
    Inputs A, T, S can be Python lists, NumPy arrays, or torch Tensors.
    Returns a 2×2 tensor or tensor(-1.) if T or S is singular.
    """
    A_t = torch.as_tensor(A, dtype=torch.float)
    T_t = torch.as_tensor(T, dtype=torch.float)
    S_t = torch.as_tensor(S, dtype=torch.float)
    # Your implementation here
    if (is_singular(S_t) or is_singular(T_t)):
        return torch.tensor(-1.)
    T_inverse = torch.linalg.inv(T_t)
    return torch.round(T_inverse @ A @ S, decimals=3)
