import numpy as np
import r_pca
import scipy.sparse

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
        
    #prod_modes == modes[-1] ?? ??????????
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
        
        #???????????? ???????????????????? A = L + S, L --- ????????????????????????, S --- ??????????????????????
        #??????????: L = U Sigma V^T --- SVD ?????? L. ?????????????????? ?????????????????????? ?????????? ?????????? ???????? (????????????????)
        #U^T A = Sigma V^T + U^T S. ?????????????? ???????????? ?????????????????? ?????? ???????????????????? ????????????.
        #????????????????, ?????? ?????????????? ???????????? U^T S ???????? ?????????? ????????????????????????...
        
        rpca = r_pca.R_pca(A) 
        
        rpca.lmbda = rpca.lmbda * lambda_scale #?????????? ?????????? ?? ?????????????? sparse'????????????
        
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
        
        #???????????? ???????????????????? A = L + S, L --- ????????????????????????, S --- ??????????????????????
        #??????????: L = U Sigma V^T --- SVD ?????? L. ?????????????????? ?????????????????????? ?????????? ?????????? ???????? (????????????????)
        #?????????????????? S ??????????????, L ???????????????????????? ?????? ????????????
        
        rpca = r_pca.R_pca(A) 
        
        rpca.lmbda = rpca.lmbda * lambda_scale #?????????? ?????????? ?? ?????????????? sparse'????????????
        
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
        #???????????? ??????????, ?????????? ????????????, ?????? ????????????????, ?????? ?????? csr
        
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
        
    #prod_modes == modes[-1] ?? ??????????
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