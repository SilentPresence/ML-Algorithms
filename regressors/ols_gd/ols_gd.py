import numpy as np
from ...preprocessing.normalizer import Normalizer
from ..ols.ols import Ols

class OlsGd(Ols):
  
  def __init__(self,
               learning_rate=.05, 
               num_iteration=1000, 
               normalize=True,
               early_stop=True,
               random_weights=True,
               epsilon=1e-3,
               verbose=True):
    
    super(OlsGd, self).__init__()
    self.learning_rate = learning_rate
    self.num_iteration = num_iteration
    self.early_stop = early_stop
    self.normalize = normalize
    self.normalizer = Normalizer()    
    self.verbose = verbose
    self.epsilon = epsilon
    self.random_weights = random_weights

  
  def _cost(self, X, y):
    #divide by 2 so that it will cancel out with the 2 at the numerator after derivation
    return 0.5*np.square(np.dot(X,self.w)-y).mean()

  def _fit(self, X, y, reset=True, track_loss=True):
    training_X=X
    if self.normalize:
      training_X=self.normalizer.fit_transform(training_X)
    training_X=self.pad(training_X)
    _,m=training_X.shape
    if reset:
      if self.random_weights:
        self.w=np.random.randn(m)
      else:
        self.w=np.zeros(m)
      self.loss_=[]
      self.weight_history_=[]
      self.steps_=0
    for i in range(self.num_iteration):
      prev_loss=self._cost(training_X,y)
      self._step(training_X,y)
      self.weight_history_.append(self.w.copy())
      current_loss=self._cost(training_X,y)
      if self.verbose and i%10==0:
        print(f'i={i} loss={current_loss},delta-loss={prev_loss-current_loss}')
      if np.isinf(current_loss):
        break
      if track_loss:
        self.loss_.append(current_loss)
      if i>0 and self.early_stop and prev_loss-current_loss<self.epsilon:
        break
 
  def _predict(self, X):
    pred_X=X
    if self.normalize:
      pred_X=self.normalizer.transform(pred_X)
    return self.pad(pred_X)@self.w
      
  def _step(self, X, y):
    self.steps_+=1
    y_pred=np.dot(X,self.w)
    gradient=np.dot(X.T,(y_pred-y))/X.shape[0]
    self.w-=(self.learning_rate)*gradient