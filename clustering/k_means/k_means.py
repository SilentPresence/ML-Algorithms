from sklearn.base import BaseEstimator,ClusterMixin
import numpy as np
from scipy.spatial import distance

class k_means(BaseEstimator, ClusterMixin):
    def __init__(self, k=8,convergence_delta=1e-4,max_iter=300,centroid_init='furthest',distance='euclidean'):
        self.k=k
        self.convergence_delta=convergence_delta
        self.max_iter=max(max_iter,1)
        self.centroid_init=centroid_init
        self.distance=distance

    def __init_centroids__(self,X):
        if self.centroid_init=='random':
            self.centroids_=X[np.random.choice(np.arange(X.shape[0]),size=self.k,replace=False)]
        if self.centroid_init=='furthest':
            unique_points=np.unique(X,axis=0)
            current_point_idx=np.argmin(unique_points,axis=0)[0]
            points=[]
            points.append(unique_points[current_point_idx])
            used_points_mask=np.zeros(unique_points.shape[0]).astype(bool)
            used_points_mask[current_point_idx]=True
            for k in range(self.k-1):
                current_point=points[k]
                centroid_distance=distance.cdist(unique_points,current_point.reshape(1,-1), self.distance)
                centroid_distance[used_points_mask]=-1
                new_point_idx=np.argmax(centroid_distance)
                used_points_mask[new_point_idx]=True
                points.append(unique_points[new_point_idx])
            self.centroids_=np.array(points)
        else:
            self.centroids_=X[np.random.choice(np.arange(X.shape[0]),size=self.k,replace=False)]

    def fit(self, X, y=None):
        self.__init_centroids__(X)
        self.clusters_=np.zeros_like(X.shape[0])
        # print(self.centroids_)
        for i in range(self.max_iter):
            #calculate distance from data points to the centroids
            centroid_distance=distance.cdist(X,self.centroids_, self.distance)
            #find for each data points the closest cluster(index)
            new_clusters=np.argmin(centroid_distance,axis=1)
            self.clusters_=new_clusters
            centroid_delta=np.empty_like(self.centroids_)
            #update centroids
            for k in range(self.k):
                new_centroid=X[self.clusters_==k,:].mean(axis=0)
                centroid_delta[k]=np.abs(new_centroid-self.centroids_[k])
                self.centroids_[k]=new_centroid
            
            if (centroid_delta<self.convergence_delta).all():
                break
        self.iterations_=i
        self.wcss_=np.min(distance.cdist(X,self.centroids_, 'sqeuclidean'),axis=1).sum()
        self.bcss_=0
        cluster_distances=distance.cdist(self.centroids_,X.mean(axis=0).reshape(1,-1), 'sqeuclidean')
        for i in range(self.centroids_.shape[0]):
            cluster=X[self.clusters_==i]
            self.bcss_=self.bcss_+cluster.shape[0]*cluster_distances[i]
    
    def predict(self, X, y=None):
        return np.argmin(distance.cdist(X,self.centroids_, self.distance),axis=1)

    def fit_predict(self, X, y=None):
        self.fit(X)
        return self.predict(X)