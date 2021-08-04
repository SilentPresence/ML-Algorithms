import numpy as np

class DecisionTreeNode:
  def __init__(self,right,left,feature_index,threshold,value,size,variance,depth):
    self.right=right
    self.left=left
    self.feature_index=feature_index
    self.threshold=threshold
    self.value=value
    self.size=size
    self.variance=variance
    self.depth=depth

  def is_leaf(self):
    return self.feature_index is None

  def __tab_depth__(self):
      return '\t'*self.depth

  def __repr__(self):
     return f"feature_index={self.feature_index},threshold={self.threshold},variance={self.variance},size={self.size},value={self.value},\n{self.__tab_depth__()}left={self.left},\n{self.__tab_depth__()}right={self.right}"

def calculate_variance(y):
  return np.var(y)
  
class DecisionTree():
    def __init__(self,min_leaf,split_criterion='variance',max_features=None,max_depth=None,seed=None,random_features=False):
        if min_leaf<=0:
            raise ValueError(f'min leaf must be positive, got {min_leaf}')
        self.min_leaf=min_leaf
        self.max_features=max_features
        self.split_criterion=split_criterion
        self.max_features=max_features
        self.max_depth=max_depth
        self.random_features=random_features
        np.random.seed(seed)
            

    def __grow_leaf__(self,y,target_variance,current_depth):
        return DecisionTreeNode(
            right=None,
            left=None,
            feature_index=None,
            threshold=None,
            value=y.mean(),
            size=y.shape[0],
            variance=target_variance,
            depth=current_depth
            )

    def __should_stop__(self,X,y,variance,current_depth):
        return (X.shape[0]<=self.min_leaf
         or X.shape[0]/2<self.min_leaf #if cannot split the datapoints by half while satisfiying the leaf condition, there is no possible partition
         or variance==0 #if the node is pure there is no need in a split
         or (self.max_depth is not None and current_depth==self.max_depth)) #if max depth is specified and reached max depth, stop

    def __find_partitionable_features__(self,X,feature_idx):
        """
        Find which features are partitionable, if the minimal leaf constraint is 1, just check that there is more than one value
        Otherwise check the number of distinct values and try to see if it possible to partition the values given the leaf constraint
        """
        partitionable_features=[]
        if self.min_leaf>1:
            for j in feature_idx:
                _,counts=np.unique(X[:,j],return_counts=True)
                #there is only one value, cannot partition
                if counts.shape[0]==1:
                    continue
                max_value_count=np.max(counts)
                other_values_count=X.shape[0]-max_value_count
                #check if it is possible to partition while satisfying the leaf constraint
                if other_values_count<self.min_leaf:
                    continue
                left_count=0
                right_count=counts.sum()
                for i in range(counts.shape[0]):
                    left_count=left_count+counts[i]
                    right_count=right_count-counts[i]
                    #at least one possible partition is available
                    if left_count>=self.min_leaf and right_count>=self.min_leaf:
                        partitionable_features.append(j)
                        break
        else:
            for j in feature_idx:
                """
                Checking min and max is wasteful.
                It is possible to find it in one iteration, but numpy does not have this functional and doing it in python without optimization is slower
                """
                x_min,x_max=np.min(X[:,j]),np.max(X[:,j])
                if x_max==x_min:
                    continue                
                partitionable_features.append(j)
        return partitionable_features

    def __find_best_split__(self,X,y):
        best_j=None
        best_threshold=None
        splitting_quality=np.Infinity

        y_min,y_max=np.min(y),np.max(y)
        #the target contains the same values, it is impossible to find a partition
        if y_max==y_min:
            return (best_j,best_threshold)

        partitionable_feature_idx=self.__find_partitionable_features__(X,self.partitionable_feature_idx)
        non_constant_count=len(partitionable_feature_idx)
        if non_constant_count==0:
            return (best_j,best_threshold)

        data_count=y.shape[0]
        y_sum=y.sum()
        y_sum_sqr=(y**2).sum()
        if self.random_features:
            feature_index_sample=np.random.choice(np.array(partitionable_feature_idx),size=np.min([self.max_features,non_constant_count]),replace=False)
        else:
            feature_index_sample=np.arange(X.shape[1])
        
        """
            O(m*(nlogn+n))=O(mnlogn+mn), m-number of features, n-number of data points
            O(mnlogn)-iterate m times over all features and for each feature we sort the feature
            O(mn)-iterate m times over n values of the feature
        """
        for j in feature_index_sample:
            #sort feature columns to be able and move the threshold boundary each time from left to right, O(nlogn)
            sorted_indices = np.argsort(X[:,j])
            y_sorted=y[sorted_indices]
            x_sorted=X[sorted_indices,j]

            current_left_sum=np.array([0])
            current_right_sum=np.array([y_sum])

            current_left_sum_square=np.array([0])
            current_right_sum_square=np.array([y_sum_sqr])

            #O(n)
            for i in range(data_count-1):
                current_value = y_sorted[i]
                next_value=y_sorted[i+1]
                left_count=i+1
                right_count=data_count-i-1
                
                current_right_sum=current_right_sum-current_value
                current_left_sum=current_left_sum+current_value

                current_right_sum_square=current_right_sum_square-np.square(current_value)
                current_left_sum_square=current_left_sum_square+np.square(current_value)
                #calculate the split variance once for each possible threshold value
                if (x_sorted[i] == x_sorted[i+1] 
                    or right_count<self.min_leaf 
                    or left_count<self.min_leaf):
                    continue

                left_mean_squared=np.square(current_left_sum/left_count)
                right_mean_squared=np.square(current_right_sum/right_count)

                left_ratio=left_count/data_count
                right_ratio=right_count/data_count
                
                current_quality=left_ratio*(current_left_sum_square/left_count-left_mean_squared)+right_ratio*(current_right_sum_square/right_count-right_mean_squared)
                    
                if current_quality<splitting_quality:
                    best_j=j
                    best_threshold=x_sorted[i]
                    splitting_quality=current_quality
        return (best_j,best_threshold)

    def __grow_tree__(self,X,y,current_depth):
        target_variance=calculate_variance(y)
        if self.__should_stop__(X,y,target_variance,current_depth):
            return self.__grow_leaf__(y,target_variance,current_depth)

        best_j,best_threshold=self.__find_best_split__(X,y)
        #Could not find a good split, make the current node a leaf
        if best_threshold is None or best_j is None:
            return self.__grow_leaf__(y,target_variance,current_depth)

        right_indices=np.where(X[:,best_j]>best_threshold)[0]
        left_indices=np.where(X[:,best_j]<=best_threshold)[0]

        #something went wrong
        if left_indices.shape[0]<self.min_leaf or right_indices.shape[0]<self.min_leaf:
            raise RuntimeError(f'invalid indices. best_j={best_j},best_threhold={best_threshold},')

        return DecisionTreeNode(
            right=self.__grow_tree__(X[right_indices],y[right_indices],current_depth+1),
            left=self.__grow_tree__(X[left_indices],y[left_indices],current_depth+1),
            feature_index=best_j,
            threshold=best_threshold,
            value=y.mean(),
            size=y.shape[0],
            variance=target_variance,
            depth=current_depth)

    def fit(self,X,y):
        if self.max_features is None:
            self.max_features=X.shape[1]
        self.partitionable_feature_idx=self.__find_partitionable_features__(X,np.arange(X.shape[1]))
        self.root=self.__grow_tree__(X,y,1)

    def predict(self, X):
        y_pred=np.empty(X.shape[0])
        for i in range(X.shape[0]):
            current_node=self.root
            while(not current_node.is_leaf()):
                j=current_node.feature_index
                threshold=current_node.threshold
                if X[i,j]<=threshold:
                    current_node=current_node.left
                else:
                    current_node=current_node.right
            y_pred[i]=current_node.value
        return y_pred