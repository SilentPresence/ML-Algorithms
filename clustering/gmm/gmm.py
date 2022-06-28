from sklearn.base import BaseEstimator,ClusterMixin
import numpy as np
from scipy.spatial import distance
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans

class Gmm(BaseEstimator, ClusterMixin):
    def __init__(self,
     components=2,
     convergence_delta=1e-4,
     max_iter=300,
     mean_init='random_data',
     covariance_init='identity',
     iteration_callback=None):
        self.components=components
        self.convergence_delta=convergence_delta
        self.max_iter=max(max_iter,1)
        self.mean_init=mean_init
        self.iteration_callback=iteration_callback
        self.covariance_init=covariance_init

    def __init_parameters__(self,X):
        n_rows,n_cols=X.shape
        if self.mean_init=='random_data':
            self.mean_=X[np.random.choice(np.arange(X.shape[0]),size=self.components,replace=False)]
        elif self.mean_init=='kmeans++':
            kmeans=KMeans(n_clusters=self.components)
            kmeans.fit(X)
            self.mean_=kmeans.cluster_centers_
        else:
            self.mean_=X[np.random.choice(np.arange(X.shape[0]),size=self.components,replace=False)]

        if self.covariance_init=='identity':
            self.covariance_= np.full((self.components,n_cols,n_cols),np.eye(n_cols))
        elif self.covariance_init=='data':
            self.covariance_= np.full((self.components,n_cols,n_cols),np.cov(X,rowvar=False))
        else:
            self.covariance_= np.full((self.components,n_cols,n_cols),np.eye(n_cols))
        self.weights_=np.full(self.components,1/self.components)
        self.responsibillity_=np.zeros((n_rows,self.components))
        self.converged_=False

    def __calculate_log_liklihood__(self,X):
        liklihood=np.empty((X.shape[0],self.components))
        for j in range(self.components):
                liklihood[:,j]=self.weights_[j]*multivariate_normal(self.mean_[j],self.covariance_[j]).pdf(X)+1e-16
        return np.log(liklihood.sum(-1)).sum()

    def fit(self, X, y=None):
        if X.ndim!=2:
            raise ValueError('')
        if X.shape[0]==0:
            raise ValueError('')
        self.__init_parameters__(X)
        prev_ll=-np.Infinity
        current_ll=-np.Infinity
        if self.iteration_callback:
            self.iteration_callback({
                'iteration':-1,
                'mean':self.mean_,
                'covariance':np.copy(self.covariance_),
                'responsibillity':self.responsibillity_,
                'current_ll':current_ll
                })
        for i in range(self.max_iter):
            # E-step
            for j in range(self.components):
                self.responsibillity_[:,j]=self.weights_[j]*multivariate_normal(self.mean_[j],self.covariance_[j]).pdf(X)+1e-32
            self.responsibillity_=self.responsibillity_/self.responsibillity_.sum(-1,keepdims=True)
            # M-step
            n_k=self.responsibillity_.sum(0,keepdims=True).T
            self.mean_=(self.responsibillity_.T@X)/n_k
            for j in range(self.components):
                mean_delta=(X-self.mean_[j]).T
                self.covariance_[j]=(self.responsibillity_[:,j]*((mean_delta))@(mean_delta.T))/n_k[j]
                # covariance regularization
                self.covariance_[j][np.diag_indices(X.shape[1], X.shape[1])]+=1e-6
            self.weights_=n_k.reshape(-1)/X.shape[0]
            # Calculate loss
            current_ll=self.__calculate_log_liklihood__(X)
            if self.iteration_callback:
                self.iteration_callback({
                    'iteration':i,
                    'mean':self.mean_,
                    'covariance':self.covariance_,
                    'responsibillity':self.responsibillity_,
                    'current_ll':current_ll
                    })
            if i>0 and abs(prev_ll-current_ll)<=self.convergence_delta:
                self.converged_=True
                break
            prev_ll=current_ll
        if i<self.max_iter and not self.converged_:
            print(f'GMM has not converged after {i} iterations')
        self.iterations_=i

    def predict(self, X, y=None): 
        return self.predict_proba(X).argmax(axis=1)

    def predict_proba(self, X, y=None): 
        if X.ndim!=2:
            raise ValueError('')
        if X.shape[0]==0:
            raise ValueError('')
        if X.shape[1]!=self.covariance_.shape[1]:
            raise ValueError('')
        pred=np.empty((X.shape[0],self.components))
        for j in range(self.components):
            pred[:,j]=self.weights_[j]*multivariate_normal(self.mean_[j],self.covariance_[j]).pdf(X)
        pred=pred/pred.sum(-1,keepdims=True)
        return pred

    def fit_predict(self, X, y=None):
        self.fit(X)
        return self.predict(X)