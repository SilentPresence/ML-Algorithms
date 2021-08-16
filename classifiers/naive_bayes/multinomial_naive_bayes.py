from sklearn.base import BaseEstimator,ClassifierMixin
import numpy as np
import pandas as pd

class MultinomialNaiveBayes(BaseEstimator, ClassifierMixin):
  """Implements the naive Bayes algorithm for multinomially distributed data, and is one of the two classic naive Bayes variants used in text classification

    Attributes:
        laplace_smoother: A number that is used to smooth the maximum likelihood for features not present in the learning samples
    """
  def __init__(self,laplace_smoother=1.0):
    self.laplace_smoother=laplace_smoother

  def fit(self, X, y):
    if isinstance(X, pd.DataFrame):
      X=X.values
    if isinstance(y, pd.DataFrame):
      y=y.values
    #Count frequency of each class and which classes are present
    self.classes_,self.classes_count_=np.unique(y,return_counts=True)
    self.priors_log_=np.log( self.classes_count_ / self.classes_count_.sum() )   
    self.likelihood=self.calc_likelihood(X,y,self.classes_.shape[0],self.laplace_smoother)

  def calc_likelihood(self,X,y,class_num,laplace_smoother):
    likelihood=np.zeros((class_num,X.shape[1]))
    voc_size=X.shape[1]
    for class_i in range(class_num):
      data_in_class_i = X[np.where(y==class_i)]
      feature_counts_for_class_i = data_in_class_i.sum(axis=0)
      feature_sum_class_i = data_in_class_i.sum()
      likelihood[class_i]=np.log((laplace_smoother+feature_counts_for_class_i)/(feature_sum_class_i+laplace_smoother*voc_size))
    return likelihood

  def predict_proba(self, X):
    return np.exp(self.predict_log_proba(X))

  def predict_log_proba(self, X):
    if self.priors_log_ is None or self.likelihood is None:
      raise ValueError('The model was not fitted')
    if isinstance(X, pd.DataFrame):
      X=X.values
    return self.priors_log_+X@self.likelihood.T

  def predict(self, X):
    return self.classes_[np.argmax(self.predict_log_proba(X),axis=1)]