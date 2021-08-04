import numpy as np
from ..ols.ols import Ols

class RidgeLs(Ols):
  def __init__(self, ridge_lambda, *wargs, **kwargs):
    super(RidgeLs,self).__init__(*wargs, **kwargs)
    self.ridge_lambda = ridge_lambda
    
  def _fit(self, X, y):
    padded_X=self.pad(X)
    regularization_term=self.ridge_lambda*np.eye(padded_X.shape[1])
    #do not penalize the w0, the bias
    regularization_term[0,0]=0
    inverse=np.linalg.inv(np.dot(padded_X.T,padded_X)+regularization_term)
    self.w=np.dot(inverse.dot(padded_X.T),y)
