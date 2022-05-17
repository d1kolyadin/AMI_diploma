import numpy as np
import scipy.sparse
import pywt
    
def values(func, left, right, n):
    return func(np.linspace(left, right, n))

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

def get_vector_modes(modes):
    return [m ** 2 for m in modes]

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

def restore_matrices(
    vectors,
    d,
    vector_modes
):
    assert len(vectors) == d

    matrices = []
    for v in vectors:
        matrices.append(
            vector_to_matrix(
                v.toarray().flatten(order='F'),
                d,
                vector_modes
            )[0]
        )
    return matrices

def psnr(m1, m2, max_i=255.0):
    assert m1.shape == m2.shape
    
    #mse = np.linalg.norm(m1 - m2) ** 2 / np.prod(m1.shape)
    mse = np.mean(np.square(m1 - m2))
    return 20 * np.log10(max_i) - 10 * np.log10(mse)

def threshold_2d_coeffs(coeffs, thr):
    arr, slc = pywt.coeffs_to_array(coeffs)
    arr = pywt.threshold(arr, thr, mode='hard')
    return pywt.array_to_coeffs(arr, slc, output_format='wavedec2')