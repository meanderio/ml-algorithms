import numpy as np

class Node:
    def __init__(
            self,
            feature=None, 
            threshold=None, 
            left=None, 
            right=None, 
            gain=None, 
            value=None
        ):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.gain = gain
        self.value = value
        pass


class DecisionTree:
    def __init__(self, min_samples=2, max_depth=3):
        self.root = Node()
        self.min_samples = min_samples
        self.max_depth = max_depth
    
    def entropy(self, y):
        entropy = 0.0
        labels  = np.unique(y)
        for label in labels:
            y_i      = len(y[y == label]) / len(y)
            entropy += -y_i * np.log2(y_i)
        return entropy
    
    def information_gain(self, parent, left, right):
        information_gain = 0

        information_gain += self.entropy(parent)
        lw = (len(left) / len(parent)) * self.entropy(left)
        rw = (len(right) / len(parent)) * self.entropy(right)
        information_gain -= lw + rw
        return information_gain

    def split_data(self, dataset, feature, threshold):
        ld = []
        rd = []

        for sample in dataset:
            if sample[feature] < threshold:
                ld.append(sample)
            else:
                rd.append(sample)
        return np.array(ld), np.array(rd)
    
    def find_best_split(self, dataset, num_samples, num_features):
        best_split = {
            'gain':-1, 
            'feature': None, 
            'threshold': None,
            'left_data': None,
            'right_data': None
        }
        for f_idx in range(num_features):
            f_vals = dataset[:,f_idx]
            thresholds = np.unique(f_vals)
            for threshold in thresholds:
                ld, rd = self.split_data(dataset, f_idx, threshold)

                
                if len(ld) and len(rd):
                    cur_ig = self.information_gain(dataset[:, -1], ld[:, -1], rd[:, -1])
                    if cur_ig > best_split["gain"]: 
                        best_split["gain"]       = cur_ig
                        best_split["feature"]    = f_idx
                        best_split["threshold"]  = threshold
                        best_split["left_data"]  = ld
                        best_split["right_data"] = rd
                
        return best_split
    
    def calculate_leaf_value(self, y):
        y = list(y)
        return max(y, key=y.count)

    def build_tree(self, dataset, current_depth=0):
        X, y = dataset[:, :-1], dataset[:, -1]
        n_samples, n_features = X.shape

        if n_samples >= self.min_samples and current_depth <= self.max_depth:
            best_split = self.find_best_split(dataset, n_samples, n_features)
            if best_split["gain"] > 0:
                
                left_node  = self.build_tree(best_split["left_data"], current_depth+1)
                right_node = self.build_tree(best_split["right_data"], current_depth+1)
               
                return Node(
                    gain      = best_split["gain"],
                    threshold = best_split["threshold"],
                    feature   = best_split["feature"],
                    left      = left_node,
                    right     = right_node
                )
        leaf_value = self.calculate_leaf_value(y)
        return Node(value=leaf_value)

    def fit(self, X, y):
        dataset   = np.concatenate((X, y.reshape(y.shape[0], 1)), axis=1)
        self.root = self.build_tree(dataset)
        pass
    
    def predict(self, X):
        y_pred = []
        for x in X:
            prediction = self.make_prediction(x, self.root)
            y_pred.append(prediction)
        return np.array(y_pred)
    
    def make_prediction(self, x, node):

        if node.value != None: return node.value

        if x[node.feature] < node.threshold:
            return self.make_prediction(x, node.left)
        else:
            return self.make_prediction(x, node.right)

