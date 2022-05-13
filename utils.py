import numpy as np
from r_pca import R_pca
import scipy.sparse
    
def values(func, left, right, n):
    return func(np.linspace(left, right, n))

def wtt_filter(input_vector, d, modes, ranks=None, eps=None):
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
        if A.shape[0] <= A.shape[1]:
            u, sigmas, vt = np.linalg.svd(A, full_matrices=False)
        else:
            u, sigmas, vt = np.linalg.svd(A, full_matrices=True)
            
        r_given = None if ranks is None else (1 if k == d - 1 else ranks[k])
        r_eps = min(A.shape) if eps is None else max(1, (sigmas > eps * sigmas[0]).sum())
        if r_given is not None:
            r_cur = min(r_given, r_eps)
        else:
            r_cur = r_eps
        
        filters.append(u)

        assert u.shape[0] == u.shape[1] == r_prev * modes[k]
        if k < d - 1:
            assert r_cur <= r_prev * modes[k]

        if k < d - 1:
            A = (u.T @ A)[:r_cur,:]
            prod_modes //= modes[k]
            true_ranks.append(r_cur)
            r_prev = r_cur
    
    return filters, true_ranks

def wtt_apply(input_vector, d, filters, modes, ranks):
    prod_modes = input_vector.size
    
    assert len(filters) == d
    assert len(modes) == d
    assert len(ranks) == d - 1
    assert prod_modes == np.prod(modes)
        
    tails = []
    A = input_vector
    r_prev = 1
    for k in range(d):
        A = A.reshape((r_prev * modes[k], prod_modes // modes[k]), order='F')
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

def iwtt_apply(input_vector, d, filters, modes, ranks):
    prod_modes = input_vector.size
    
    assert len(filters) == d
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
            result = (filters[k] @ result).reshape((r_prev, prod_modes), order='F')
        else:
            result = (filters[k] @ np.vstack([
                result,
                tails[k]
            ])).reshape((r_prev, prod_modes), order='F')
        prod_modes *= modes[k]
    
    return result.flatten(order='F')

def matrix_to_vector(A, d, modes):
    assert A.shape[0] == A.shape[1] == np.prod(modes)
    assert len(modes) == d

    result = np.reshape(A, modes + modes, order='F')
    axes_transpose = []
    for i in range(d):
        axes_transpose.append(i)
        axes_transpose.append(d + i)
    result = np.transpose(result, axes_transpose)
    new_modes = [m ** 2 for m in modes]
    result = np.reshape(result, new_modes, order='F')
    return result.flatten(order='F'), new_modes

def vector_to_matrix(v, d, modes):
    assert v.size == np.prod(modes)
    assert len(modes) == d
    
    new_modes = [int(np.sqrt(m)) for m in modes]
    assert np.all(modes == np.square(new_modes))

    result = np.reshape(v, new_modes + new_modes, order='F')
    axes_transpose = []
    for i in range(d):
        axes_transpose.append(2 * i)
    for i in range(d):
        axes_transpose.append(2 * i + 1)
    result = np.transpose(result, axes_transpose)
    result = np.reshape(result, (np.prod(new_modes), np.prod(new_modes)), order='F')
    return result, new_modes

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
                "\nnnz(S) = ", np.count_nonzero(S), #можно так, поскольку S-часть по-честному разреженная
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
                "\nnnz(S) = ", np.count_nonzero(S), #можно так, поскольку S-часть по-честному разреженная
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

def wtt_rpca_v3(
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
    
    #сначала делаем предобработку за счёт RPCA
    #затем --- полученную по итогу малоранговую часть раскладываем как обычно
    
    A = input_vector
    for k in range(d):
        A = A.reshape((-1, prod_modes // modes[k]), order='F')

        rpca = R_pca(A) 
        rpca.lmbda = rpca.lmbda * lambda_scale #делаю уклон в сторону sparse'овости        
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
    filters, true_ranks = wtt_filter(smoothed_signal, d, modes, ranks, eps)
    
    return filters, sparse_parts, true_ranks

def wtt_apply_rpca_v3(input_vector, d, filters, sparse_parts, modes, ranks):
    prod_modes = input_vector.size
    
    assert len(filters) == d
    assert len(sparse_parts) == d
    assert len(modes) == d
    assert len(ranks) == d - 1
    assert prod_modes == np.prod(modes)
    
    A = input_vector
    for k in range(d):
        A = A.reshape((-1, prod_modes // modes[k]), order='F')
                
        A = np.asarray(A - sparse_parts[k])
        prod_modes //= modes[k]
        
    smoothed_signal = A.flatten(order='F')
    coeffs = wtt_apply(smoothed_signal, d, filters, modes, ranks)
    return coeffs

def iwtt_apply_rpca_v3(input_vector, d, filters, sparse_parts, modes, ranks):
    prod_modes = input_vector.size
    
    assert len(filters) == d
    assert len(sparse_parts) == d
    assert len(modes) == d
    assert len(ranks) == d - 1
    assert prod_modes == np.prod(modes)
    
    A = iwtt_apply(input_vector, d, filters, modes, ranks)
    for k in range(d - 1, -1, -1):
        A = A.reshape((prod_modes, -1), order='F')
        A = np.asarray(A + sparse_parts[k])
        prod_modes //= modes[k]
        
    return A.flatten(order='F')

def subtract_sparse_parts(
    input_vector,
    d,
    modes,
    sparse_parts
):
    prod_modes = input_vector.size
    A = input_vector

    assert prod_modes == np.prod(modes)
    assert len(modes) == len(sparse_parts) == d
    
    for k in range(d):
        A = A.reshape((-1, prod_modes // modes[k]), order='F')
        assert A.shape == sparse_parts[k].shape
        A = np.asarray(A - sparse_parts[k])
        prod_modes //= modes[k]
    
    return A.flatten(order='F')

def add_sparse_parts(
    input_vector,
    d,
    modes,
    sparse_parts
):
    prod_modes = input_vector.size
    A = input_vector

    assert prod_modes == np.prod(modes)
    assert len(modes) == len(sparse_parts) == d
    
    for k in range(d - 1, -1, -1):
        A = A.reshape((prod_modes, -1), order='F')
        assert A.shape == sparse_parts[k].shape
        A = np.asarray(A + sparse_parts[k])
        prod_modes //= modes[k]
    
    return A.flatten(order='F')