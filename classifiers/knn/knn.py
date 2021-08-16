from sklearn.base import BaseEstimator,ClassifierMixin
import numpy as np
from scipy import stats
from scipy.spatial.distance import cdist
import pandas as pd

class kNNClassifier(BaseEstimator, ClassifierMixin):
  def __init__(self, n_neighbors,distance='euclidean'):
    self.n_neighbors = n_neighbors
    self.distance=distance

  def fit(self, X, y):
    if isinstance(X, list):
      X=np.array(X)
    elif isinstance(X, pd.DataFrame):
      X=X.values
    if isinstance(y, list):
      y=np.array(y)
    elif isinstance(y, pd.DataFrame):
      y=y.values
    elif isinstance(y, pd.Series):
      y=y.values

    self._X=X
    self._y=y
    return self

  def predict(self, X,method='brute'):
    if isinstance(X, list):
      X=np.array(X)
    elif isinstance(X, pd.DataFrame):
      X=X.values

    if method=='brute':
      return self.__predict__(X)
    #TODO: add kdetree
    return self.__predict__(X)

  def __predict__(self,X):
    distance_matrix=cdist(self._X,X,metric=self.distance)

    prediction=np.empty(X.shape[0])
    for i in range(X.shape[0]):
      distance_vector=distance_matrix[:,i]
      indices = np.argpartition(distance_vector, self.n_neighbors)[:self.n_neighbors]
      prediction[i]=stats.mode(self._y[indices])[0]
    return prediction