from typing import Callable

import torch


def _interleave(a, b):
    """
    Merge 2 tensors on the last dimension by interleaving elements:
    Args:
        a: torch.tensor(..., L)
            a = [1,2,3]
        b: torch.tensor(..., L)
            b = [4,5,6]
    Returns:
        res: torch.tensor(..., 2L)
            res = [1,4,2,5,3,6]
    """
    assert a.shape[-1] == b.shape[-1] or a.shape[-1] == b.shape[-1] + 1
    add = False
    dim = len(b.shape) - 1
    if a.shape[-1] == b.shape[-1] + 1:
        b = torch.cat([b, torch.zeros_like(b[..., 0].unsqueeze(dim=-1))], dim=dim)
        add = True
    dims = list(a.shape)
    dims[-1] = a.shape[-1] + b.shape[-1]
    res = torch.zeros(size=dims, dtype=a.dtype, device=a.device)
    mask = torch.arange(start=0, end=dims[-1], step=2, device=a.device)
    res[..., mask] = a
    res[..., mask + 1] = b
    if add:
        return res[..., :-1]
    return res


def _combine(fn, a_flat, b_flat):
    """
    fn each element of a with each element of b.
    TODO: Can parallelize here
    Args:
        fn: Callable
            Associative binary function to apply on last dimension of a_flat and b_flat
        a_flat: torch.tensor(..., L)
            a = [1,2,3]
        b_flat: torch.tensor(..., L)
            b = [4,5,6]
    Return:
        res: torch.tensor(..., L)
            Reduced a_flat and b_flat
            res = [fn(1,4), fn(2,5), fn(3,6)]
    """
    fn_map = torch.vmap(fn, in_dims=(-1, -1))
    out = fn_map(a_flat, b_flat)
    # move the first dimension to last
    # vmap puts the vectorisation dimensions in front
    dims = list(range(a_flat.dim()))
    dims = [x + 1 for x in dims]
    dims[-1] = 0
    out = out.permute(dims)
    return out


def associative_scan(fn: Callable, elems):
    """
    Implement associative scan for a binary function fn.
    See notion for and explanation of the algorithm
    Args:
        fn: Callable
            Associative function to apply on last dimension of a_flat and b_flat
        elems: torch.tensor(..., L)
            a = [1,2,3]
    Return:
        res: torch.tensor(..., L)
            Prefix scan of the array.
            res[-1] = Identity fn
            res[i] = fn(res[i-1], a[i])
    """
    num_elems = elems.shape[-1]
    if num_elems < 2:
        return elems

    # compute sums of pairs on the branch level of the binary tree
    reduced_elems = _combine(fn, elems[..., 0:-1:2], elems[..., 1::2])
    # compute sum of sums
    odd_elements = associative_scan(fn, reduced_elems)
    # even elements = from_left + node sum (sums in between parent.left and current.left
    # as with recursion we only get subtrees
    if num_elems % 2 == 0:
        even_elems = _combine(fn, odd_elements[..., :-1], elems[..., 2::2])
    else:
        even_elems = _combine(fn, odd_elements, elems[..., 2::2])
    # merge odd and even
    # the first element of the scan is the same as the first element of the orignal
    dim = len(elems.shape) - 1
    even_elems = torch.cat([elems[..., 0].unsqueeze(dim=-1), even_elems], dim=dim)
    return _interleave(even_elems, odd_elements)
