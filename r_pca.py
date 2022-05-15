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

def wtt_rpca_v1(
    input_vector,
    d,
    modes,
    ranks=None,
    eps=None,
    lambda_scale=1.0,
    verbose=True,
):
    
    filters = []
    prod_modes = input_vector.size
    
    assert len(modes) == d
    if ranks is not None:
        assert len(ranks) == d - 1
    if eps is not None:
        assert 0 <= eps <= 1
    assert prod_modes == np.prod(modes)
        
    true_ranks = []
    
    r_prev = 1
    A = input_vector
    for k in range(d):
        A = A.reshape((r_prev * modes[k], prod_modes // modes[k]), order='F')
        
        #делаем разложение A = L + S, L --- малоранговая, S --- разреженная
        #затем: L = U Sigma V^T --- SVD для L. Ненулевых сингулярных чисел будет мало (надеемся)
        #U^T A = Sigma V^T + U^T S. Старшие строки оставляем для дальнейшей работы.
        #Надеемся, что младшие строки U^T S тоже будут разреженными...
        
        rpca = R_pca(A) 
        
        rpca.lmbda = rpca.lmbda * lambda_scale #делаю уклон в сторону sparse'овости
        
        if verbose:
            print("Step", k, "out of", d)
        
        L, S = rpca.fit(
            max_iter=4000,
            iter_print=400,
            verbose=verbose
        )
        
        if A.shape[0] <= A.shape[1]:
            u, sigmas, vt = np.linalg.svd(L, full_matrices=False)
        else:
            u, sigmas, vt = np.linalg.svd(L, full_matrices=True)
            
        r_given = None if ranks is None else (1 if k == d - 1 else ranks[k])
        r_eps = min(A.shape) if eps is None else max(1, (sigmas > eps * sigmas[0]).sum())
        if r_given is not None:
            r_cur = min(r_given, r_eps)
        else:
            r_cur = r_eps
        
        filters.append(u)
        
        if verbose:
            
            print(
                "Low-rank check:",
                "\nr_cur = ", r_cur,
                "\n#singular values = ", sigmas.size,
                "\n#nnz singlular values = ", (sigmas > 1e-10).sum(),
                sep=''
            )

            print(
                "Sparsity check:",
                "\nS.size = ", S.size,
                "\nnnz(S) = ", np.count_nonzero(S),
                "\nnnz(u.T @ S) = ", np.count_nonzero(u.T @ S),
                sep=''
            )

        assert u.shape[0] == u.shape[1] == r_prev * modes[k]
        if k < d - 1:
            assert r_cur <= r_prev * modes[k]

        if k < d - 1:
            A = (u.T @ A)[:r_cur,:]
            #A = (u.T @ (L + S))[:r_cur,:]
            prod_modes //= modes[k]
            true_ranks.append(r_cur)
            r_prev = r_cur
    
    return filters, true_ranks

def wtt_rpca_v2(
    input_vector,
    d,
    modes,
    ranks=None,
    eps=None,
    lambda_scale=1.0,
    verbose=True,
):
    
    filters = []
    sparse_parts = []
    prod_modes = input_vector.size
    
    assert len(modes) == d
    if ranks is not None:
        assert len(ranks) == d - 1
    if eps is not None:
        assert 0 <= eps <= 1
    assert prod_modes == np.prod(modes)
        
    true_ranks = []
    
    r_prev = 1
    A = input_vector
    for k in range(d):
        A = A.reshape((r_prev * modes[k], prod_modes // modes[k]), order='F')
        
        #делаем разложение A = L + S, L --- малоранговая, S --- разреженная
        #затем: L = U Sigma V^T --- SVD для L. Ненулевых сингулярных чисел будет мало (надеемся)
        #сохраняем S целиком, L раскладываем как обычно
        
        rpca = R_pca(A) 
        
        rpca.lmbda = rpca.lmbda * lambda_scale #делаю уклон в сторону sparse'овости
        
        if verbose:
            print("Step", k, "out of", d)
        
        L, S = rpca.fit(
            max_iter=4000,
            iter_print=400,
            verbose=verbose
        )
        
        if A.shape[0] <= A.shape[1]:
            u, sigmas, vt = np.linalg.svd(L, full_matrices=False)
        else:
            u, sigmas, vt = np.linalg.svd(L, full_matrices=True)
            
        r_given = None if ranks is None else (1 if k == d - 1 else ranks[k])
        r_eps = min(A.shape) if eps is None else max(1, (sigmas > eps * sigmas[0]).sum())
        if r_given is not None:
            r_cur = min(r_given, r_eps)
        else:
            r_cur = r_eps
        
        filters.append(u)
        sparse_parts.append(scipy.sparse.csr_matrix(S))
        #скорее всего, строк меньше, чем столбцов, так что csr
        
        if verbose:
            print(
                "Low-rank check:",
                "\nr_cur = ", r_cur,
                "\n#singular values = ", sigmas.size,
                "\n#nnz singlular values = ", (sigmas > 1e-10).sum(),
                sep=''
            )
            print(
                "Sparsity check:",
                "\nS.size = ", S.size,
                "\nnnz(S) = ", np.count_nonzero(S),
                sep=''
            )

        assert u.shape[0] == u.shape[1] == r_prev * modes[k]
        if k < d - 1:
            assert r_cur <= r_prev * modes[k]

        if k < d - 1:
            A = (u.T @ L)[:r_cur,:]
            prod_modes //= modes[k]
            true_ranks.append(r_cur)
            r_prev = r_cur
    
    return filters, sparse_parts, true_ranks

def wtt_apply_rpca_v2(input_vector, d, filters, sparse_parts, modes, ranks):
    prod_modes = input_vector.size
    
    assert len(filters) == d
    assert len(sparse_parts) == d
    assert len(modes) == d
    assert len(ranks) == d - 1
    assert prod_modes == np.prod(modes)
        
    tails = []
    A = input_vector
    r_prev = 1
    for k in range(d):
        A = A.reshape((r_prev * modes[k], prod_modes // modes[k]), order='F')
        A = np.asarray(A - sparse_parts[k])
        A = filters[k].T @ A

        assert A.shape[0] == r_prev * modes[k]
        if k < d - 1:
            assert ranks[k] <= r_prev * modes[k]
                
        if k < d - 1:
            tails.append(A[ranks[k]:,:])
            A = A[:ranks[k],:]
            prod_modes //= modes[k]
            r_prev = ranks[k]
        
    result = A
    for k in range(d - 2, -1, -1):        
        result = np.vstack([
            result.reshape((ranks[k], prod_modes), order='F'),
            tails[k]
        ])
        prod_modes *= modes[k]
    
    return result.flatten(order='F')

def iwtt_apply_rpca_v2(input_vector, d, filters, sparse_parts, modes, ranks):
    prod_modes = input_vector.size
    
    assert len(filters) == d
    assert len(sparse_parts) == d
    assert len(modes) == d
    assert len(ranks) == d - 1
    assert prod_modes == np.prod(modes)
        
    tails = []
    A = input_vector
    r_prev = 1
    for k in range(d):
        A = A.reshape((r_prev * modes[k], prod_modes // modes[k]), order='F')

        assert A.shape[0] == r_prev * modes[k]
        if k < d - 1:
            assert ranks[k] <= r_prev * modes[k]
                
        if k < d - 1:
            tails.append(A[ranks[k]:,:])
            A = A[:ranks[k],:]
            prod_modes //= modes[k]
            r_prev = ranks[k]
        
    #prod_modes == modes[-1] в конце
    result = A
    for k in range(d - 1, -1, -1):
        r_prev = 1 if k == 0 else ranks[k - 1]
        if k == d - 1:
            result = (filters[k] @ result)
        else:
            result = (filters[k] @ np.vstack([
                result,
                tails[k]
            ]))
        result = np.asarray(result + sparse_parts[k])
        result = result.reshape((r_prev, prod_modes), order='F')
        prod_modes *= modes[k]
    
    return result.flatten(order='F')

def wtt_rpca_preprocessing_v1(
    input_vector,
    d,
    modes,
    lambda_scale=1.0,
    verbose=True,
):
    
    sparse_parts = []
    prod_modes = input_vector.size
    
    assert len(modes) == d
    assert prod_modes == np.prod(modes)
    
    A = input_vector
    for k in range(d):
        A = A.reshape((-1, prod_modes // modes[k]), order='F')

        rpca = R_pca(A) 
        rpca.lmbda = rpca.lmbda * lambda_scale      
        L, S = rpca.fit(
            max_iter=4000,
            iter_print=400,
            verbose=verbose
        )
        if A.shape[0] <= A.shape[1]:
            sparse_parts.append(scipy.sparse.csr_matrix(S))
        else:
            sparse_parts.append(scipy.sparse.csc_matrix(S)) 
        A = L
        prod_modes //= modes[k]

        if verbose:
            print("Step", k, "out of", d, "(preprocessing)")
            print(
                "Sparsity check:",
                "\nS.size = ", S.size,
                "\nnnz(S) = ", np.count_nonzero(S),
                sep=''
            )

    smoothed_signal = A.flatten(order='F')
    return smoothed_signal, sparse_parts

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
            r_up,
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
                if r_up is not None and r_up < r_next:
                    r_next = r_up
        carriages[0] = np.squeeze(carriages[0])
        
        #print("shapes of summands:", *[s.shape for s in carriages])

        result = carriages[0]
        for c in carriages[1:]:
            result = np.dot(result, c.reshape((c.shape[0], -1), order='F'))
            result = result.reshape((-1, c.shape[2]), order='F')
        
        return result, r_next

    def fit_mode(
            self, 
            mode,
            first_modes,
            prev_ranks,
            r_up=None,
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
            
            Lk, rk = self.tt_rounding_step(
                A,
                mode,
                first_modes,
                prev_ranks,
                r_up,
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
    
def wtt_rpca_preprocessing_v2(
    input_vector,
    d,
    modes,
    upper_ranks=None,
    lambda_scale=1.,
    tol=None,
    max_iter=1000
):
    ranks = []
    sparse_parts = []
    A = input_vector
    prod_modes = input_vector.size
    
    assert prod_modes == np.prod(modes)
    assert len(modes) == d
    if upper_ranks is not None:
        assert len(upper_ranks) == d - 1
    
    for k in range(d):
        
        #print("Current step", k)
        
        A = A.reshape((-1, prod_modes // modes[k]), order='F')
        rpca = R_pca_tensorised(A)
        rpca.lmbda *= lambda_scale
        r_up = None if upper_ranks is None else (1 if k + 1 == d else upper_ranks[k])
        L, S, r = rpca.fit_mode(
            k + 1,
            modes[:k + 1],
            ranks,
            r_up=r_up,
            tol=tol,
            max_iter=max_iter,
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


    