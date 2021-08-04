import numpy as np
from sklearn.base import TransformerMixin

class Normalizer(TransformerMixin):
  def __init__(self):
    pass

  def fit(self, X):
    self.mean_,self.std_=np.mean(X,axis=0),np.std(X,axis=0)

  def transform(self, X):
    return (X-self.mean_)/self.std_

  def fit_transform(self, X):
    self.fit(X)
    return self.transform(X)