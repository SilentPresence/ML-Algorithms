from ..ols_gd.ols_gd import OlsGd
import numpy as np

class RidgeOlsGd(OlsGd):
    def __init__(self, ridge_lambda, *wargs, **kwargs):
        super(RidgeOlsGd,self).__init__(*wargs, **kwargs)
        self.ridge_lambda = ridge_lambda
    
    def _cost(self, X, y):
        return 0.5*np.square(np.dot(X,self.w)-y).mean() + 0.5*self.ridge_lambda*self.w[1:].dot(self.w[1:].T)/X.shape[0]

    def _step(self, X, y):
        self.steps_+=1
        y_pred=np.dot(X,self.w)
        gradient=np.dot(X.T,(y_pred-y))/X.shape[0]
        self.w-=(self.learning_rate)*gradient
        #do not penalize the bias
        self.w[1:]-=self.learning_rate*self.ridge_lambda * self.w[1:]/X.shape[0]
