
import time
from .utils import clustetring_acc_hungarian
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score


class BaseCluster:

    def __init__(self, name=None):
        self.name = name if name is not None else self.__class__.__name__
        self.labels = None
        self.time = time.time()
    
    def fit_predict(self, X):
        raise NotImplementedError("Subclasses should implement this method.")

        # necessary code for instance functionality
        # self.time = time.time() - self.time
        # return self.labels

    def evaluation(self, Y_true):

        assert self.labels is not None, "Please run fit or fit_predict first."
        acc = float(clustetring_acc_hungarian(Y_true, self.labels))
        nmi = float(normalized_mutual_info_score(Y_true, self.labels))
        ari = float(adjusted_rand_score(Y_true, self.labels))

        return acc, nmi, ari