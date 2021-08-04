from sklearn.base import BaseEstimator,RegressorMixin
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import numpy as np
from sklearn.metrics import mean_squared_error

class l2boost(BaseEstimator,RegressorMixin):
    """
        A model that uses stumps (decision tree with max_depth=1) to minimize the L2 square loss iteration by iteration (boosting)
    """
    def __init__(self,max_depth=1,n_estimators=100,learning_rate=0.1,tol=1e-3,validation_fraction=0.1,n_iter_no_change=5):
        if n_iter_no_change is not None:
            if n_iter_no_change<=0:
                raise ValueError(f'n_iter_no_change must be greater than 0,was provided:{n_iter_no_change}')
            
        self.max_depth=max_depth
        self.n_estimators=n_estimators
        self.learning_rate=learning_rate
        self.validation_fraction=validation_fraction
        self.tol=tol
        self.n_iter_no_change=n_iter_no_change


    def fit(self,X,y):
        early_stop_enabled=self.validation_fraction is not None and self.n_iter_no_change is not None
        if early_stop_enabled:
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=self.validation_fraction, random_state=0)
        else:
            X_train=X
            y_train=y

        self.y_mean=y_train.mean()
        self.fs=[]
        self.fs.append(np.repeat(self.y_mean,y_train.shape[0]))
        self.resids=[]
        self.trees=[]
        self.val_mse=[]
        self.train_mse=[]
        self.val_mse_improv=[]

        for i in range(self.n_estimators):
            self.train_mse.append(self.score(y_train,self.fs[i]))

            if early_stop_enabled:
                self.val_mse.append(self.score(y_val,self.predict(X_val)))
                if i>0:
                    self.val_mse_improv.append(self.val_mse[i-1]-self.val_mse[i])
                if i>=self.n_iter_no_change:
                    n_iter_improv=np.array(self.val_mse_improv[i-self.n_iter_no_change:i])
                    if (n_iter_improv>0).all() and (n_iter_improv<self.tol).all():
                        print(f'Early stop.{n_iter_improv}')
                        break

            resid=y_train-self.fs[i]
            self.resids.append(resid)
            tree=DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X_train,resid)
            self.trees.append(tree)
            self.fs.append(self.fs[i]+self.learning_rate*tree.predict(X_train))

        self.final_estimators_=len(self.trees)
            

    def score(self,y_true,y_pred):
        return mean_squared_error(y_true,y_pred)

    def predict(self,X):
        y_pred=np.repeat(self.y_mean, X.shape[0])
        for i in range(len(self.trees)):
            y_pred+=self.learning_rate*self.trees[i].predict(X)
        return y_pred