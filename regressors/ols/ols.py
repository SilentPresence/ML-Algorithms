import numpy as np
from sklearn.base import BaseEstimator,RegressorMixin

class Ols(BaseEstimator,RegressorMixin):
  def __init__(self):
    self.w = None
    
  @staticmethod
  def pad(X):
    return np.hstack((np.ones((X.shape[0],1)),X))
  
  def fit(self, X, y):
    self._fit(X,y)

  def _fit(self, X, y):
    self.w=np.dot(np.linalg.pinv(self.pad(X)),y)

  def predict(self, X):
    return self._predict(X)

  def _predict(self, X):
    return np.dot(self.pad(X),self.w)

  def score(self, X, y):
    return np.square(self.predict(X)-y).mean()

  def score_r2(self, X, y):
    return 1-np.square(self.predict(X)-y).sum()/np.square(y.mean()-y).sum()