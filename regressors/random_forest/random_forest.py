from sklearn.metrics import r2_score
import numpy as np
from ..decision_tree.decision_tree import DecisionTree
class TreeEnsemble():
  def __init__(self, X, y, n_trees, sample_sz, min_leaf,max_features=None):
    if X.shape[0]<sample_sz:
      raise ValueError('Sample size cannot be greater than data size')
    if min_leaf<1 :
      raise ValueError('Min leaf must be greater than 0')
    if min_leaf>(X.shape[0]/2):
      raise ValueError('Min leaf must allow for a partition on the samples')
    if n_trees<1:
      raise ValueError('Number of tree must be greater than 0')
    self.X=X
    self.y=y
    self.n_tree=n_trees
    self.sample_sz=sample_sz
    self.min_leaf=min_leaf
    self.max_features=max_features
    if max_features is None or max_features>X.shape[1] or max_features<=0:
      self.max_features=X.shape[1]

  def fit(self):
    self.trees=[]
    #sum of errors for each oob sample
    self.oob=np.zeros_like(self.y)
    #number of times each oob was not trained on
    self.oob_count=np.zeros_like(self.y)

    index_range=np.arange(0,self.X.shape[0])
    for i in range(self.n_tree):
      #get a bootstrap sample
      sample_index=np.random.choice(index_range,size=self.sample_sz,replace=True)
      #get all samples which are not in the training set
      oob_index_range=np.setdiff1d(index_range,sample_index)

      X_sample=self.X[sample_index]
      y_sample=self.y[sample_index]
      dt=DecisionTree(min_leaf=self.min_leaf,max_features=self.max_features,random_features=True)
      dt.fit(X_sample,y_sample)
      oob_prediction=dt.predict(self.X[oob_index_range])
      #save oob prediction for later use
      self.oob[oob_index_range]=self.oob[oob_index_range]+oob_prediction
      self.oob_count[oob_index_range]=self.oob_count[oob_index_range]+1
      self.trees.append(dt)

  def predict(self, X):
    y_pred=np.zeros(X.shape[0])
    for i in range(self.n_tree):
      y_pred=y_pred+self.trees[i].predict(X)
    return y_pred/self.n_tree
  
  def oob_mse(self):
    oob_mean=self.oob/self.oob_count
    return np.square(self.y-oob_mean).mean()

  #from documentation of sklearn https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html#sklearn.ensemble.RandomForestRegressor.score
  def oob_determination(self):
    oob_per_row=self.oob/self.oob_count
    #not sure about this, but maybe it will be better to just exclude nan instead of filling them
    oob_per_row[np.isnan(oob_per_row)]=np.nanmean(oob_per_row)
    return r2_score(self.y,oob_per_row,multioutput='uniform_average')