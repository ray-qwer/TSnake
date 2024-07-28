import torch
import math
# import rpe_index_cpp

def rpe_index(input, index):
    '''Y[b, h, i, j] = input[b, h, i, index[i, j]]

    Parameters
    ----------
    input: torch.Tensor, float32
        The shape is (B, H, L_query, num_buckets)
    index: torch.Tensor, int32
        The shape is (L_query, L_key)

    where B is the batch size, and H is the number of attention heads.

    Returns
    -------
    Y: torch.Tensor, float32
        The shape is (B, H, L_query, L_key)
    '''
    L_query, L_key = index.shape
    num_buckets = input.size(-1)
    B = len(input)
    offset = torch.arange(0, L_query * num_buckets, num_buckets).view(-1, 1).to(index.device)
    return input.flatten(2)[:, :, (index + offset).flatten()].view(B, -1, L_query, L_key)


# EXPECTED_VERSION = "1.2.0"
# assert rpe_index_cpp.version() == EXPECTED_VERSION, \
#         f"""Unmatched `rpe_index_cpp` version: {rpe_index_cpp.version()}, expected version: {EXPECTED_VERSION}
# Please re-build the package `rpe_ops`."""


# class RPEIndexFunction(torch.autograd.Function):
#     '''Y[b, h, i, j] = input[b, h, i, index[i, j]]'''
#     @staticmethod
#     def forward(ctx, input, index):
#         '''
#         Y[b, h, i, j] = input[b, h, i, index[i, j]]

#         Parameters
#         ----------
#         input: torch.Tensor, float32
#             The shape is (B, H, L_query, num_buckets)
#         index: torch.Tensor, int32
#             The shape is (L_query, L_key)

#         where B is the batch size, and H is the number of attention heads.

#         Returns
#         -------
#         Y: torch.Tensor, float32
#             The shape is (B, H, L_query, L_key)
#         '''

#         num_buckets = input.size(-1)
#         ctx.save_for_backward(index)
#         ctx.input_shape = input.shape
#         forward_fn = rpe_index_cpp.forward_cpu if \
#             input.device.type == 'cpu' else rpe_index_cpp.forward_gpu
#         output = forward_fn(input, index)
#         return output

#     @staticmethod
#     def backward(ctx, grad_output):
#         '''
#           - Inputs
#               grad_output: float32 (B, H, L_query, L_key)
#           - Outputs
#               grad_input: float32 (B, H, L_query, num_buckets)
#         '''
#         index = ctx.saved_tensors[0]
#         if ctx.needs_input_grad[0]:
#             grad_input = grad_output.new_zeros(ctx.input_shape)
#             backward_fn = rpe_index_cpp.backward_cpu if \
#                 grad_output.device.type == 'cpu' else rpe_index_cpp.backward_gpu
#             backward_fn(grad_input, grad_output, index)
#             return grad_input, None
#         return None, None

@torch.no_grad()
def piecewise_index(relative_position, alpha, beta, gamma, dtype):
    """
    This function is to reassign the index for those out of alpha, make it smoother
    """
    """piecewise index function defined in Eq. (18) in our paper.

    Parameters
    ----------
    relative_position: torch.Tensor, dtype: long or float
        The shape of `relative_position` is (L, L).
    alpha, beta, gamma: float
        The coefficients of piecewise index function.

    Returns
    -------
    idx: torch.Tensor, dtype: long
        A tensor indexing relative distances to corresponding encodings.
        `idx` is a long tensor, whose shape is (L, L) and each element is in [-beta, beta].
    """
    rp_abs = relative_position.abs()
    mask = rp_abs <= alpha
    not_mask = ~mask
    rp_out = relative_position[not_mask]
    rp_abs_out = rp_abs[not_mask]
    y_out = (torch.sign(rp_out) * (alpha +
                                   torch.log(rp_abs_out / alpha) /
                                   math.log(gamma / alpha) *
                                   (beta - alpha)).round().clip(max=beta)).to(dtype)

    idx = relative_position.clone()
    if idx.dtype in [torch.float32, torch.float64]:
        # round(x) when |x| <= alpha
        idx = idx.round().to(dtype)

    # assign the value when |x| > alpha
    idx[not_mask] = y_out
    return idx

@torch.no_grad()
def _rp_2d_euclidean(diff, **kwargs):
    """2D RPE with Euclidean method.

    Parameters
    ----------
    diff: torch.Tensor
        The shape of `diff` is (L, L, 2),
        where L is the sequence length,
        and 2 represents a 2D offset (row_offset, col_offset).

    Returns
    -------
    index: torch.Tensor, dtype: long
        index to corresponding encodings.
        The shape of `index` is (L, L),
        where L is the sequence length.
    """
    dis = diff.square().sum(2).float().sqrt().round()
    return piecewise_index(dis, **kwargs)
    
if __name__ == '__main__':
    import numpy as np
    import time
    B = 1
    H = 1
    L_query = 10
    L_key = L_query
    num_buckets = 10

    x = torch.randn(B, H, L_query, num_buckets)

    index = torch.randint(low=0, high=num_buckets, size=(L_query, L_key))
    index = index.to(torch.int)
    # print(rpe_index(x, index))
    offset = torch.arange(0, L_query * num_buckets, num_buckets).view(-1, 1)
    print(index)
    def test(x, index, offset):
        tic = time.time()
        x1 = x.clone()
        x1.requires_grad = True
        x2 = x.clone()
        x2.requires_grad = True

        y = rpe_index(x1, index)
#         y = RPEIndexFunction.apply(x1, index)
        gt_y = x2.flatten(2)[:, :, (index + offset).flatten()
                             ].view(B, H, L_query, L_key)
        np.testing.assert_almost_equal(
            gt_y.detach().cpu().numpy(), y.detach().cpu().numpy())

        mask = torch.randn(gt_y.shape, device=x.device)
        (gt_y * mask).sum().backward()
        (y * mask).sum().backward()

        print("X1:", x1.grad.cpu().numpy().flatten().sum())
        print("X2:", x2.grad.cpu().numpy().flatten().sum())
        np.testing.assert_almost_equal(
            x1.grad.cpu().numpy(), x2.grad.cpu().numpy(), decimal=5)
        print("Test over", x.device)
        print("Cost:", time.time() - tic)
    test(x, index, offset)
    if torch.cuda.is_available():
        test(x.cuda(), index.cuda(), offset.cuda())