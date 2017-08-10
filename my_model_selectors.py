import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        best_score = float('inf')
        best_model = None
        
        logN = np.log(self.X.shape[0])
        n_features = self.X.shape[1]
        
        for n_components in range(self.min_n_components, self.max_n_components + 1):
        
            try:
                model = self.base_model(n_components)
                logL = model.score(self.X, self.lengths)        
                
                # n_params = n_components * (n_components - 1) + 2 * n_features * n_components
                n_params = (model.startprob_.size - 1) + (model.transmat_.size - 1) + model.means_.size + model.covars_.diagonal().size
                
                alpha = 1
                
                bic = -2 * logL + n_params * logN * alpha
                
                if bic < best_score:
                    best_score = bic
                    best_model = model
                    
            except Exception as e:
                continue            

        if best_model is None:
            best_model = self.base_model(self.n_constant)

        return best_model
        
        


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    models = {}
    values = {}
    
    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        
        best_score = float('-inf')
        best_model = None
        
        for n in range(self.min_n_components, self.max_n_components+1):
            try:
                model = self.base_model(n)
                logL = model.score(self.X, self.lengths)

                score_list = []
                for word in self.words:
                    if(word != self.this_word):
                        w, length = self.hwords[word]
                        score_list.append(model.score(w, length))

                penalty = np.mean(score_list)

                dic = logL - penalty

                if(dic > best_score):
                    best_model = model
                    best_dic = dic
            except:
                pass
        
        return best_model
    

class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''
    
    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        best_score = float('-inf')
        best_model = None
        
        n_splits = 3
        
        for n_components in range(self.min_n_components, self.max_n_components + 1):
            scores = []
    
            model, log = None, None
            
            try:
                # Using the line below will have a problem that, if self.sequences is smaller than 3 (e.g. 1),
                # It will return an error that n_splits cannot be smaller than 2
                n_splits = min(3, len(self.sequences))

                split_method = KFold(random_state=self.random_state, n_splits=n_splits)

                for cv_train_idx, cv_test_idx in split_method.split(self.sequences):

                    X_train, lengths_train = combine_sequences(cv_train_idx, self.sequences)
                    X_test,  lengths_test  = combine_sequences(cv_test_idx, self.sequences)

                    try:
                        model = GaussianHMM(n_components=n_components, covariance_type="diag", n_iter=1000,
                                            random_state=inst.random_state, verbose=False).fit(X_train, lengths_train)

                        log = model.score(X_test, lengths_test)

                        scores.append(log)
                    except Exception as e:
                        pass

                if len(scores) > 0:
                    avg = np.average(scores)  
                else:
                    avg = float("-inf")

                if avg > best_score:
                    best_score = avg
                    best_model = model
            except:
                pass
        
        if best_model is None:
            best_model = self.base_model(self.n_constant)

        return best_model