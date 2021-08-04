import numpy as np

#TODO:Refactor code to extend from sklearn base classes

def calculate_entropy(unique_arr_count):
    pArr=unique_arr_count/unique_arr_count.sum()
    return (pArr,(-np.sum(pArr*np.log2(pArr))).sum())
def get_target_majority(y):
    vals,counts = np.unique(y, return_counts=True)
    index = np.argmax(counts)
    return vals[index]
class ClassificationDecisionTreeNode:
  def __init__(self,children,feature_index,value,size,entropy,depth):
    self.children=children
    self.feature_index=feature_index
    self.value=value
    self.size=size
    self.entropy=entropy
    self.depth=depth

  def is_leaf(self):
    return self.feature_index is None

  def __tab_depth__(self):
      return '\t'*self.depth

  def __repr__(self):
    return f'<ClassificationDecisionTreeNode:feature_index={self.feature_index},entropy={self.entropy},size={self.size},depth={self.depth}\n{self.__tab_depth__()}children={self.children},value={self.value}>\n'

class ClassificationDecisionTree():
    def __init__(self,max_depth=None):
        self.max_depth=max_depth

    def __grow_leaf__(self,y,current_depth):
        _,y_unique_counts=np.unique(y,return_counts=True)
        _,entropy=calculate_entropy(y_unique_counts)
        return ClassificationDecisionTreeNode(
            children=None,
            feature_index=None,
            value=get_target_majority(y),
            size=y.shape[0],
            entropy=entropy,
            depth=current_depth
            )
    
    def __should_stop__(self,X,y,current_depth):
        y_unique=np.unique(y)
        return y_unique.shape[0]==1 or (self.max_depth is not None and current_depth==self.max_depth)

    def __find_best_split__(self,X,y):
        best_j=None
        splitting_quality=-np.Infinity
        feature_gains={}
        _,target_counts=np.unique(y,return_counts=True)
        _,target_entropy=calculate_entropy(target_counts)
        
        for j in range(X.shape[1]):
            unique_values,unique_counts=np.unique(X[:,j],return_counts=True)
            pX=unique_counts/unique_counts.sum()
            entropy_element_wise=np.empty_like(unique_values)
            
            for i in range(unique_values.shape[0]):
                indices=np.where(X[:,j]==unique_values[i])[0]
                _,y_counts=np.unique(y[indices],return_counts=True)
                _,y_entropy=calculate_entropy(y_counts)
                entropy_element_wise[i]=y_entropy

            feature_gain=target_entropy-(pX*entropy_element_wise).sum()
            feature_gains[j]=feature_gain
            if feature_gain>splitting_quality:
                best_j=j
                splitting_quality=feature_gain
        return (best_j,feature_gains)

    def __grow_tree__(self,X,y,current_depth):
        if self.__should_stop__(X,y,current_depth):
            return self.__grow_leaf__(y,current_depth)

        best_j,feature_gains=self.__find_best_split__(X,y)
        if best_j is None:
            return self.__grow_leaf__(y,current_depth)
        feature_values=np.unique(X[:,best_j])
        children={}
        for i in range(feature_values.shape[0]):
            indices=np.where(X[:,best_j]==feature_values[i])[0]
            children[feature_values[i]]=ClassificationDecisionTreeNode(
                    children=self.__grow_tree__(X[indices],y[indices],current_depth+1),
                    feature_index=best_j,
                    entropy=None,
                    value=None,
                    size=y.shape[0],
                    depth=current_depth)
        return children

    def fit(self,X,y):
        self.root=self.__grow_tree__(X,y,1)

    def predict(self, X):
        y_pred=np.empty(X.shape[0], dtype=object)
        for i in range(X.shape[0]):
            current_node_dict=self.root
            found_leaf=False
            while(not found_leaf):
                feature=current_node_dict[0].feature_index
                for key in current_node_dict:
                    if X[i,feature]==key:
                        current_node_dict=current_node_dict[key].children
                        break
                if current_node_dict.children is None:
                    found_leaf=True
            
            y_pred[i]=current_node_dict.value
        return y_pred