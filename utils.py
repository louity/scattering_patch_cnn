import math
import torch
import torch.nn as nn

def print_and_write(log, logfile=None):
    print(log)
    if logfile is not None:
        with open(logfile, 'a') as f:
            f.write(log + '\n')


def compute_sparse_code_conv(input_batch, dictionary, lambda_1, maxiter=100, tol=1e-3, L=None):
    """
        Computes a sparse code alpha of the input_batch x in the dictionary D solving the
        Lasso problem :
            min_alpha || x - D alpha ||_2**2 + lambda_1 ||alpha||_1
        using the FISTA algorithm.

        Inspired by https://gist.github.com/agramfort/ac52a57dc6551138e89b.

        Parameters:
            input_batch: torch tensor, size (batch_size, input_size)
                input_batch of which we want to compute the sparse code

            dictionary: torch Tensor, size (input_size, dict_size)
               dictionary to compute the sparse code

            lambda_1: float
                regulariation parameter in front of the l1 norm

            maxiter: int, default 1000
                maxmum number of iterations of the FISTA algorithm

            tol: float < 1, optional
                stop criterion tolerance

        Returns:
            alpha: torch Tensor, size (batch_size, dict_size)
               the sparse code of the input batch in the dictionary D
    """
    dict_size = dictionary.size(1)
    batch_size, M, N = input_batch.size(0), input_batch.size(2), input_batch.size(3)
    alpha = input_batch.new_zeros(batch_size, dict_size, M, N)
    t = 1
    z = alpha.clone()

    if len(dictionary.size()) == 2:
        dictionary = dictionary.view(dictionary.size(0), dictionary.size(1), 1, 1).contiguous()

    if L is None:
        L = torch.symeig(torch.mm(dictionary[..., 0, 0].t(), dictionary[..., 0, 0]))[0][-1]

    for i_iter in range(maxiter):
        alpha_old = alpha.clone()
        D_z = nn.functional.conv2d(z, dictionary)
        z = z + nn.functional.conv2d(input_batch - D_z, dictionary.transpose(0, 1).contiguous()) / L
        alpha = nn.functional.softshrink(z, lambda_1 / L)
        t0 = t
        t = (1. + math.sqrt(1. + 4. * t ** 2)) / 2.
        z = alpha + ((t0 - 1.) / t) * (alpha - alpha_old)

        diff = ((alpha - alpha_old).norm(p=1, dim=1) / (alpha.norm(p=1, dim=1) + 1e-10))

        if tol is not None and diff.max() < tol:
            break

    D_alpha = nn.functional.conv2d(alpha, dictionary)
    l2_loss = 0.5 * ((input_batch - D_alpha)**2).sum(dim=1).mean()
    l1_loss = lambda_1 * alpha.norm(p=1, dim=1).mean()

    return alpha, l2_loss, l1_loss
