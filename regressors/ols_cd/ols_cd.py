import numpy as np
from ...preprocessing.normalizer import Normalizer
from ..ols_gd.ols_gd import OlsGd

class OlsCd(OlsGd):
  """Coordinate descent"""
  
  def __init__(self, 
               num_iteration=10, 
               epsilon=1e-3,
               learning_rate=1, 
               normalize=True,
               early_stop=True,
               random_weights=True,
               verbose=True):
    super(OlsCd, self).__init__()
    self.num_iteration = num_iteration
    self.learning_rate = learning_rate
    self.verbose = verbose
    self.epsilon = epsilon
    self.early_stop = early_stop
    self.normalize = normalize
    self.normalizer = Normalizer()  

  def _step(self, X, y):
    self.steps_+=1
    for j in range(self.w.shape[0]):
      #Coordinate gradient descent step for the single dimension 
      y_pred=np.dot(X,self.w)
      self.w[j]-=(self.learning_rate/X.shape[0])*np.dot(X[:,j].T,(y_pred-y))