from sklearn.base import BaseEstimator,ClassifierMixin
import numpy as np

class MultinomialNaiveBayes(BaseEstimator, ClassifierMixin):
  """Implements the naive Bayes algorithm for multinomially distributed data, and is one of the two classic naive Bayes variants used in text classification

    Attributes:
        laplace_smoother: A number that is used to smooth the maximum likelihood for features not present in the learning samples
    """
  def __init__(self,laplace_smoother=1.0):
    self.laplace_smoother=laplace_smoother

  def fit(self, X, y):
    #Count frequency of each class and which classes are present
    self.classes_,self.classes_count_=np.unique(y,return_counts=True)
    #Calculate the log prior
    self.priors_log_=np.log( self.classes_count_ / self.classes_count_.sum() )
    self.likelihood=np.zeros((self.classes_.shape[0],X.shape[1]))
    voc_size=X.shape[1]
    for class_i in range(self.classes_.shape[0]):
      data_in_class_i = X[np.where(y==class_i)]
      feature_counts_for_class_i = data_in_class_i.sum(axis=0)
      feature_sum_class_i = data_in_class_i.sum()
      self.likelihood[class_i]=np.log((self.laplace_smoother+feature_counts_for_class_i)/(feature_sum_class_i+self.laplace_smoother*voc_size))

  def predict_proba(self, X):
    return np.exp(self.predict_log_proba)

  def predict_log_proba(self, X):
    if self.priors_log_ is None or self.likelihood is None:
      raise ValueError('The model was not fitted')
    return self.priors_log_+X@self.likelihood.T

  def predict(self, X):
    return self.classes_[np.argmax(self.predict_log_proba(X),axis=1)]