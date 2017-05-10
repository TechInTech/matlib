import numpy as np
from base import BaseEstimator
from scipy.stats import itemfreq

class Apriori(BaseEstimator):
    y_required=False

    def __init__(self, min_confidence=0.3, min_support=0.3):
        self.min_confidence = min_confidence
        self.min_support = min_support
    
    def evaluate(self, X):
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        flatX = []
        for x in X:
            for xi in x:
                flatX.append(xi)
        flatX = np.array(flatX)
        counts = flatX.size
        items = np.unique(flatX)
        itemfreqs = itemfreq(flatX)
        
        freqitems = itemfreqs[itemfreqs[:,1] >= counts * self.min_support][:, 0]
        freqs = np.array(freqitems, dtype=np.object)

        itemnum = 1
        while(itemnum <= len(freqitems)):
            candidates = self._get_candidates(freqs, freqitems, itemnum)
            itemnum += 1
            if len(candidates) == 0:
                break
            for candidate in candidates:
                count = 0
                for x in X:
                    idx = 0
                    for xi in x:
                        if xi == candidate[idx]:
                            idx +=1
                    
                    if idx == itemnum:
                        count += 1
                if count >= counts * self.min_support:
                    freqs.append(candidate)
        print freqs
        
    def _get_candidates(self, freqs, freqitems, itemnum):
        candidates = []
        for itemset in freqs:
            if itemnum == 1:
                if isinstance(itemset, int):
                    for item in freqitems:
                        if item != itemset:
                            c = np.array([itemset, item])
                            c.sort()
                            had = False
                            for ci in candidates:
                                if np.array_equal(ci, c):
                                    had=True
                                    break
                            if not had:
                                candidates.append(c)
            else:
                if not isinstance(itemset, int) and len(itemset) == itemnum:
                    for item in freqitems:
                        c = np.concatenate(itemset, item)
                        c.sort()
                        candidates.append(c)
        
        return candidates