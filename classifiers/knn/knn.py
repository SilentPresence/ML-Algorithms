from sklearn.base import BaseEstimator,ClassifierMixin
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from scipy import stats

class kNNClassifier(BaseEstimator, ClassifierMixin):
  def __init__(self, n_neighbors):
    self.n_neighbors = n_neighbors

  def fit(self, X, y):
    self._X=X
    self._y=y
    return self

  def predict(self, X,method='brute'):
    if method=='brute':
      return self.__predict__(X)
    #TODO: add kdetree
    return self.__predict__(X)

  def __predict__(self,X):
    distance_matrix=euclidean_distances(self._X,X)

    prediction=np.empty(X.shape[0])
    for i in range(X.shape[0]):
      distance_vector=distance_matrix[:,i]
      indices = np.argpartition(distance_vector, self.n_neighbors)[:self.n_neighbors]
      prediction[i]=stats.mode(self._y[indices])[0]
    return prediction