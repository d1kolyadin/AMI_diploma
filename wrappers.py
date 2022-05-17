import utils
import wtt
import r_pca

class WTTwrapper:
    def __init__(self, alg_type):
        #1 --- классический
        #2 --- wtt_rpca_v1
        #3 --- wtt_rpca_v2
        
        assert 1 <= alg_type <= 3
        self.alg_type = alg_type
        
        self.d = None
        self.modes = []
        self.filters = []
        self.ranks = []
        self.sparse_parts = []
    
    def adjust(self,
               input_data,
               d,
               modes,
               ranks=None,
               eps=None,
               lambda_scale=1.0,
               is_2D=False
        ):        
        if is_2D is True:
            data, vector_modes = utils.matrix_to_vector(input_data, d, modes)
        else:
            data, vector_modes = input_data, modes
        
        if self.alg_type == 1:
            filters, ranks = wtt.wtt_filter(
                data,
                d,
                vector_modes,
                ranks,
                eps
            )
            sparse_parts = []
        elif self.alg_type == 2:
            filters, ranks = wtt.wtt_rpca_v1(
                data,
                d,
                vector_modes,
                ranks,
                eps,
                lambda_scale,
                False
            )
            sparse_parts = []
        elif self.alg_type == 3:
            filters, sparse_parts, ranks = wtt.wtt_rpca_v2(
                data,
                d,
                vector_modes,
                ranks,
                eps,
                lambda_scale,
                False
            )
            
        self.d = d
        self.modes = modes
        self.filters = filters
        self.ranks = ranks
        self.sparse_parts = sparse_parts
        
    def apply(self, input_data, is_2D=False):
        if is_2D is True:
            data, vector_modes = utils.matrix_to_vector(input_data, self.d, self.modes)
        else:
            data, vector_modes = input_data, self.modes
            
        if 1 <= self.alg_type <= 2:
            wtt_result = wtt.wtt_apply(
                data,
                self.d,
                self.filters,
                vector_modes,
                self.ranks
            )
        elif self.alg_type == 3:
            wtt_result = wtt.wtt_apply_rpca_v2(
                data,
                self.d,
                self.filters,
                self.sparse_parts,
                vector_modes,
                self.ranks
            )

        return wtt_result
    
    def apply_inverse(self, input_data, is_2D=False):
        vector_modes = self.modes if is_2D is False else utils.get_vector_modes(self.modes)
        if 1 <= self.alg_type <= 2:
            iwtt_result = wtt.iwtt_apply(
                input_data,
                self.d,
                self.filters,
                vector_modes,
                self.ranks
            )
        elif self.alg_type == 3:
            iwtt_result = wtt.iwtt_apply_rpca_v2(
                input_data,
                self.d,
                self.filters,
                self.sparse_parts,
                vector_modes,
                self.ranks
            )
        if is_2D is True:
            iwtt_result, _ = utils.vector_to_matrix(iwtt_result, self.d, vector_modes)
        return iwtt_result
    
class RPCA_preprocessing_wrapper:
    def __init__(self, alg_type):
        #1 --- wtt_rpca_preprocessing_v1 (обычный RPCA)
        #2 --- wtt_rpca_preprocessing_v2 (тензоризованный RPCA)
        
        assert 1 <= alg_type <= 2
        self.alg_type = alg_type
        
        self.d = None
        self.modes = []
        self.ranks = []
        self.sparse_parts = []
        
    def adjust(
            self,
            input_vector,
            d,
            modes,
            upper_ranks=None,
            lambda_scale=1.,
            max_iter=1000
    ):
        
        if self.alg_type == 1:
            low_rank, sparse_parts, ranks = r_pca.wtt_rpca_preprocessing_v1(
                input_vector,
                d,
                modes,
                upper_ranks,
                lambda_scale,
                max_iter,
                False
            )
        elif self.alg_type == 2:
            low_rank, sparse_parts, ranks = r_pca.wtt_rpca_preprocessing_v2(
                input_vector,
                d,
                modes,
                upper_ranks,
                lambda_scale,
                max_iter,
            )
        
        self.d = d
        self.modes = modes
        self.ranks = ranks
        self.sparse_parts = sparse_parts
        
        return low_rank, sparse_parts, ranks
    
    def subtract_sparse_parts(self, input_vector):
        return utils.subtract_sparse_parts(input_vector, self.d, self.modes, self.sparse_parts)
    
    def add_sparse_parts(self, input_vector):
        return utils.add_sparse_parts(input_vector, self.d, self.modes, self.sparse_parts)
    