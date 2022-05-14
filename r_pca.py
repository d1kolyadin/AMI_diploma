from __future__ import division, print_function

import numpy as np

try:
    from pylab import plt
except ImportError:
    print('Unable to import pylab. R_pca.plot_fit() will not work.')

try:
    # Python 2: 'xrange' is the iterative version
    range = xrange
except NameError:
    # Python 3: 'range' is iterative - no need for 'xrange'
    pass

import string
import scipy.sparse


class R_pca:

    def __init__(self, D, mu=None, lmbda=None):
        self.D = D
        self.S = np.zeros(self.D.shape)
        self.Y = np.zeros(self.D.shape)

        if mu:
            self.mu = mu
        else:
            self.mu = np.prod(self.D.shape) / (4 * np.linalg.norm(self.D, ord=1))

        self.mu_inv = 1 / self.mu

        if lmbda:
            self.lmbda = lmbda
        else:
            self.lmbda = 1 / np.sqrt(np.max(self.D.shape))

    @staticmethod
    def frobenius_norm(M):
        return np.linalg.norm(M, ord='fro')

    @staticmethod
    def shrink(M, tau):
        return np.sign(M) * np.maximum((np.abs(M) - tau), np.zeros(M.shape))

    def svd_threshold(self, M, tau):
        U, S, V = np.linalg.svd(M, full_matrices=False)
        return np.dot(U, np.dot(np.diag(self.shrink(S, tau)), V))

    def fit(self, tol=None, max_iter=1000, iter_print=100, verbose=True):
        iter = 0
        err = np.Inf
        Sk = self.S
        Yk = self.Y
        Lk = np.zeros(self.D.shape)

        if tol:
            _tol = tol
        else:
            _tol = 1E-7 * self.frobenius_norm(self.D)

        #this loop implements the principal component pursuit (PCP) algorithm
        #located in the table on page 29 of https://arxiv.org/pdf/0912.3599.pdf
        while (err > _tol) and iter < max_iter:
            Lk = self.svd_threshold(
                self.D - Sk + self.mu_inv * Yk, self.mu_inv)                            #this line implements step 3
            Sk = self.shrink(
                self.D - Lk + (self.mu_inv * Yk), self.mu_inv * self.lmbda)             #this line implements step 4
            Yk = Yk + self.mu * (self.D - Lk - Sk)                                      #this line implements step 5
            err = self.frobenius_norm(self.D - Lk - Sk)
            iter += 1
            if verbose and ((iter % iter_print) == 0 or iter == 1 or iter > max_iter or err <= _tol):
                print('iteration: {0}, error: {1}'.format(iter, err))

        self.L = Lk
        self.S = Sk
        return Lk, Sk

    def plot_fit(self, size=None, tol=0.1, axis_on=True):

        n, d = self.D.shape

        if size:
            nrows, ncols = size
        else:
            sq = np.ceil(np.sqrt(n))
            nrows = int(sq)
            ncols = int(sq)

        ymin = np.nanmin(self.D)
        ymax = np.nanmax(self.D)
        print('ymin: {0}, ymax: {1}'.format(ymin, ymax))

        numplots = np.min([n, nrows * ncols])
        plt.figure()

        for n in range(numplots):
            plt.subplot(nrows, ncols, n + 1)
            plt.ylim((ymin - tol, ymax + tol))
            plt.plot(self.L[n, :] + self.S[n, :], 'r')
            plt.plot(self.L[n, :], 'b')
            if not axis_on:
                plt.axis('off')

def tt_svd(tensor, eps, max_rank):
    """
    Input
        tensor: np array
        eps: desired difference in frobenius norm between tensor and TT approximation
        max_rank: upper hard limit on each TT rank (it has priority over eps)

    Output
        carriages: list of cores that give TT decomposition of tensor
    """

    remaining = tensor
    d = len(tensor.shape)
    N = tensor.size
    r = 1

    eps = eps / np.sqrt(d - 1) #потому что ошибка в tt_svd составляет
    #sqrt(sum_{k <= d - 1} квадрат ошибки в svd для A_k) = sqrt(d - 1) * ошибка в каждом svd

    carriages = []

    for k in range(d - 1):
        matrix_to_svd = remaining.reshape((r * tensor.shape[k], N // tensor.shape[k]), order='F')
        u, sigmas, vt = np.linalg.svd(matrix_to_svd, full_matrices=False)

        curr_r = min(sigmas.size, max_rank)
        error_squared = np.sum(np.square(sigmas[curr_r:]))
        while curr_r >= 1 and error_squared + np.square(sigmas[curr_r - 1]) < np.square(eps):
            error_squared = error_squared + np.square(sigmas[curr_r - 1])
            curr_r -= 1

        carriages.append(u[:,:curr_r].reshape((r, tensor.shape[k], curr_r), order='F'))
        remaining = np.diag(sigmas[:curr_r]) @ vt[:curr_r,:]
        N = N // tensor.shape[k]
        r = curr_r

    carriages.append(remaining.reshape((r, tensor.shape[-1], 1), order='F'))

    return carriages

class R_pca_tensorised:

    def __init__(self, D, mu=None, lmbda=None):
        self.D = D
        self.S = np.zeros(self.D.shape)
        self.Y = np.zeros(self.D.shape)
        self.r = min(self.D.shape)

        if mu:
            self.mu = mu
        else:
            self.mu = np.prod(self.D.shape) / (4 * np.linalg.norm(self.D, ord=1))

        self.mu_inv = 1 / self.mu

        if lmbda:
            self.lmbda = lmbda
        else:
            self.lmbda = 1 / np.sqrt(np.max(self.D.shape))
            
        self.alphabet = string.ascii_letters 

    @staticmethod
    def frobenius_norm(M):
        return np.linalg.norm(M, ord='fro')

    @staticmethod
    def shrink(M, tau):
        return np.sign(M) * np.maximum((np.abs(M) - tau), np.zeros(M.shape))

    def svd_threshold(self, M, tau):
        U, S, V = np.linalg.svd(M, full_matrices=False)
        S_shrinked = self.shrink(S, tau)
        rank = (S_shrinked > 0).sum()
        return np.dot(U, np.dot(np.diag(S_shrinked), V)), rank
    
    def tt_rounding_step(
            self,
            tensor,
            mode,
            first_modes,
            prev_ranks,
            tau
        ):
        
        A = tensor
        r_prev = 1
        carriages = []
        for k in range(mode):
            A = A.reshape((r_prev * first_modes[k], -1), order='F')
            if k + 1 < mode:
                if A.shape[0] < A.shape[1]:
                    u, sigmas, vt = np.linalg.svd(A, full_matrices=False)
                else:
                    u, sigmas, vt = np.linalg.svd(A, full_matrices=True)
                r_next = prev_ranks[k]
                carriages.append(
                    u[:,:r_next].reshape((r_prev, first_modes[k], r_next), order='F')
                )
                A = np.dot(np.diag(sigmas[:r_next]), vt[:r_next,:])
                r_prev = r_next
            else:
                svd_thr, r_next = self.svd_threshold(A, tau)
                carriages.append(
                    svd_thr.reshape((r_prev, first_modes[k], -1), order='F')
                )
        carriages[0] = np.squeeze(carriages[0])
        
        #будем надеяться, что d не будет превышать 50
        einsum_str = self.alphabet[0] + self.alphabet[1]
        position = 1
        for k in range(mode - 1):
            einsum_str += ','
            einsum_str += self.alphabet[position]
            einsum_str += self.alphabet[position + 1]
            einsum_str += self.alphabet[position + 2]
            position += 2
        
        result = np.einsum(einsum_str, *carriages, order='F')
        
        #print([t.shape for t in carriages])
        #print(result.shape)
        
        return result, r_next
        

    def fit_mode(
            self, 
            mode,
            first_modes,
            prev_ranks,
            tol=None,
            max_iter=1000,
            iter_print=100,
            verbose=True
        ):
        iter = 0
        err = np.Inf
        Sk = self.S
        Yk = self.Y
        Lk = np.zeros(self.D.shape)
        
        assert mode == len(first_modes) == len(prev_ranks) + 1
        
        if tol:
            _tol = tol
        else:
            _tol = 1E-7 * self.frobenius_norm(self.D)

        while (err > _tol) and iter < max_iter:
            A = self.D - Sk + self.mu_inv * Yk
            A = A.reshape(first_modes + [-1], order='F')
            
            #print("Before calling tt_rouning_step:", A.shape)
            
            Lk, rk = self.tt_rounding_step(
                A,
                mode,
                first_modes,
                prev_ranks,
                self.mu_inv
            )
            Lk = Lk.reshape(self.D.shape, order='F')
            
            Sk = self.shrink(
                self.D - Lk + (self.mu_inv * Yk), self.mu_inv * self.lmbda)             #this line implements step 4
            Yk = Yk + self.mu * (self.D - Lk - Sk)                                      #this line implements step 5
            err = self.frobenius_norm(self.D - Lk - Sk)
            iter += 1
            if verbose and ((iter % iter_print) == 0 or iter == 1 or iter > max_iter or err <= _tol):
                print('iteration: {0}, error: {1}'.format(iter, err))
                

        self.L = Lk
        self.S = Sk
        self.r = rk
        return Lk, Sk, rk
    
def wtt_rpca_preprocessing(
    input_vector,
    d,
    modes,
    lambda_scale=1.
):
    ranks = []
    sparse_parts = []
    A = input_vector
    prod_modes = input_vector.size
    
    assert prod_modes == np.prod(modes)
    assert len(modes) == d
    
    for k in range(d):
        
        print("Current step", k)
        
        A = A.reshape((-1, prod_modes // modes[k]), order='F')
        rpca = R_pca_tensorised(A)
        rpca.lmbda *= lambda_scale
        L, S, r = rpca.fit_mode(
            k + 1,
            modes[:k + 1],
            ranks,
            max_iter=2,
            verbose=False
        )
        if S.shape[0] < S.shape[1]:
            sparse_parts.append(scipy.sparse.csr_matrix(S))
        else:
            sparse_parts.append(scipy.sparse.csc_matrix(S))

        A = L
        prod_modes //= modes[k]
        
        if k + 1 < d:
            ranks.append(r)
        
    return A.flatten(order='F'), sparse_parts, ranks


    