import random
import numpy as np
from collections import Counter

class DecisionNode():
    """Class to represent a single node in
    a decision tree."""

    def __init__(self, left, right, decision_function,class_label=None):
        """Create a node with a left child, right child,
        decision function and optional class label
        for leaf nodes."""
        self.left = left
        self.right = right
        self.decision_function = decision_function
        self.class_label = class_label

    def decide(self, feature):
        """Return on a label if node is leaf,
        or pass the decision down to the node's
        left/right child (depending on decision
        function)."""
        if(self.decision_function):
            val = self.decision_function(feature)
            # print "feature: ", feature
            # print "go left: ", val
        if self.class_label is not None:
            # print "feature: ", feature
            # print "prediction: ", self.class_label
            return self.class_label
        elif self.decision_function(feature):
            return self.left.decide(feature)
        else:
            return self.right.decide(feature)

def build_decision_tree():
    """Create decision tree
    capable of handling the provided 
    data."""

    #build full tree from root
    decision_tree_root= DecisionNode(None, None, lambda ex: ex[0] == 1)
    decision_tree_root.left = DecisionNode(None, None, None, 1)

    a4 = DecisionNode(None, None, lambda ex: ex[3] == 1)
    a4.right =  DecisionNode(None, None, None, 0)
    a4.left =  DecisionNode(None, None, None, 1)

    a3a4 = DecisionNode(None, None, None,None)
    a3a4.decision_function = test_lambda #lambda ex: ex[2]==ex[3]
    a3a4.right =  DecisionNode(None, None, None, 0)
    a3a4.left =  DecisionNode(None, None, None, 1)

    a2 = DecisionNode(None, None, lambda ex: ex[1] == 0)
    a2.right = a3a4
    a2.left = a4

    decision_tree_root.right = a2

    return decision_tree_root
def test_lambda(ex):
    a = ex[2]
    b = ex[3]
    return a == b
def confusion_matrix(classifier_output, true_labels):
    #output should be [[true_positive, false_negative], [false_positive, true_negative]]
    result = [[0.0, 0.0], [0.0, 0.0]]
    for i in range(0,len(classifier_output)):
        if classifier_output[i] == 1: #predict 1
            if true_labels[i] == 1: # actual 1
                #true positive
                result[0][0]+=1.0
            else: # actual 0
                #false positive
                # print i
                result[1][0]+=1.0
        else: #predict 0
            if true_labels[i] == 0: # actual 0
                #true negative
                result[1][1]+=1.0
            else: # actual 1
                #false negative
                # print i
                result[0][1]+=1.0
    return result

def precision(classifier_output, true_labels):
    #precision is measured as: true_positive/ (true_positive + false_positive)
    cm = confusion_matrix(classifier_output, true_labels)
    divisor = (cm[0][0]+cm[1][0])
    return float(cm[0][0]/ divisor) if divisor else 0
    
def recall(classifier_output, true_labels):
    #recall is measured as: true_positive/ (true_positive + false_negative)
    cm = confusion_matrix(classifier_output, true_labels)
    divisor = (cm[0][0]+cm[0][1])
    return float(cm[0][0]/ divisor) if divisor else 0
    
def accuracy(classifier_output, true_labels):
    #accuracy is measured as:  correct_classifications / total_number_examples
    cm = confusion_matrix(classifier_output, true_labels)
    divisor = (cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1])
    return float((cm[0][0] + cm[1][1])/ divisor) if divisor else 0

def entropy(class_vector):
    """Compute the entropy for a list
    of classes (given as either 0 or 1)."""
    list = class_vector.tolist()
    if len(list)>0:
        positive = float(list.count(1))
        negative = float(list.count(0))
        p = float(positive / (positive + negative))
        if p==0 or p==1:
            return 0
        else:
            logs = np.log2([p,1-p])
            result = -p*logs[0] - (1-p)*logs[1]
            return result
    return 0

def information_gain(previous_classes, current_classes ):
    """Compute the information gain between the
    previous and current classes (each 
    a list of 0 and 1 values)."""
    remainder = 0
    num_examples = float(len(previous_classes))
    prev_entropy = entropy(previous_classes)

    for classification in current_classes:
        class_examples = float(len(classification))
        class_entropy =  entropy(np.array(classification))
        remainder+= (class_examples/num_examples)*class_entropy

    return prev_entropy - remainder

def most_frequent_class(classes):
    count1 = np.count_nonzero(classes == 1)
    count0 = classes.size - count1
    if count0>count1:
        return 0
    else:
        return 1

class DecisionTree():
    """Class for automatic tree-building
    and classification."""

    def __init__(self, depth_limit=10):
        """Create a decision tree with an empty root
        and the specified depth limit."""
        self.root = None
        self.depth_limit = depth_limit

    def fit(self, features, classes):
        """Build the tree from root using __build_tree__()."""
        self.root = self.__build_tree__(np.array(features), np.array(classes))

    def __build_tree__(self, features, classes, depth=0):
        """Implement the above algorithm to build
        the decision tree using the given features and
        classes to build the decision functions."""

        classes = classes.astype(int)
        types = set(classes.flatten())
        num_types = len(types)
        # Check for base cases:
        if num_types==1:# a) If all elements of a list are of the same class
            return DecisionNode(None,None,None,types.pop())# return a leaf node with the appropriate class label.

        elif depth == self.depth_limit: # b) If a specified depth limit is reached,
            label = most_frequent_class(classes)
            return DecisionNode(None,None,None,label) # return a leaf labeled with the most frequent class.

        else:

            # store the best index, gain and split value of alpha_best
            alpha_best_ig = float("-inf")
            alpha_best = -1
            alpha_best_split = float("-inf")
            # For each attribute alpha
            for index,alpha in enumerate(features.transpose()):
                # print alpha.shape
                # print alpha.size

                # base case all in alpha have the same value, choose majority
                if len(set(alpha))==1:
                    return DecisionNode(None,None,None,most_frequent_class(classes))

                min_ = min(alpha)
                max_ = max(alpha)
                # compute step size between min and max
                step_size = float(abs(max_ - min_)/100)

                 # store the best split value and its alpha
                best_split = min_
                best_split_alpha = 0.

                # for each split
                for split in np.arange(min_+step_size,max_,step_size):
                    split_classes = [[],[]]
                    # get indexes for all values below the threshold
                    less = np.where(alpha < split)

                    # split classes using current split value
                    split_classes[0] = classes[np.where(alpha < split)]
                    split_classes[1] = classes[np.where(alpha >= split)]

                    # evaluate the normalized information gain gained by splitting on attribute $\alpha$
                    split_alpha = information_gain(classes,split_classes)

                    # if this is the best_split save it for node creation
                    if split_alpha > best_split_alpha:
                        best_split = split
                        best_split_alpha = split_alpha
                # after looping through all splits, retain the best one and its alpha
                if best_split_alpha > alpha_best_ig:
                    alpha_best = index
                    alpha_best_ig = best_split_alpha
                    alpha_best_split = best_split
            new_nodes = []

            # negative classes are to the left of split threshold
            neg_indxs = np.where(features[:,alpha_best] < alpha_best_split)
            neg_features = features[neg_indxs]
            neg_classes = classes[neg_indxs]
            new_nodes.append(self.__build_tree__(neg_features,neg_classes,depth+1))

             # positive classes are to the right of split threshold
            pos_indxs = np.where(features[:,alpha_best] >= alpha_best_split)
            pos_features = features[pos_indxs]
            pos_classes = classes[pos_indxs]
            new_nodes.append(self.__build_tree__(pos_features,pos_classes,depth+1))

            # print(pos_features.shape)
            # print(neg_features.shape)

            return DecisionNode(new_nodes[1],new_nodes[0],lambda x: x[alpha_best] >= alpha_best_split)

    def classify(self, features):
        """Use the fitted tree to 
        classify a list of examples. 
        Return a list of class labels."""
        class_labels = [self.root.decide(feature) for feature in features]
        return class_labels

def load_csv(data_file_path, class_index=-1):
    handle = open(data_file_path, 'r')
    contents = handle.read()
    handle.close()
    rows = contents.split('\n')
    out = np.array([[float(i) for i in r.split(',')] for r in rows if r ])
    classes= map(int,  out[:, class_index])
    features = out[:, :class_index]
    return features, classes

def generate_k_folds(dataset, k):
    #this method should return a list of folds,
    # where each fold is a tuple like (training_set, test_set)
    # where each set is a tuple like (examples, classes)

    full_features = dataset[0]
    num_examples = len(full_features)
    # full_classes = np.transpose(np.matrix(dataset[1]))

    full_classes = np.asarray(dataset[1]).reshape(num_examples,1)

    sample_size = int(num_examples/k)

    full_examples = np.concatenate((full_features,full_classes), axis=1)
    np.random.shuffle(full_examples)

    k_folds = []

    for i in range(0,k):
        start_idx = i*sample_size
        end_idx = start_idx+sample_size
        if end_idx>num_examples:
            end_idx=num_examples-1

        index_arr = range(start_idx,end_idx)

        sample = full_examples[index_arr,:]
        scale = int(sample.shape[0] * 0.10)
        train = sample[scale:,:]
        test = sample[:scale,:]
        train_e = train[:,0:-1]
        train_c = train[:,-1]
        test_e = test[:,0:-1]
        test_c = test[:,-1]
        k_folds.append([(train_e,train_c),(test_e,test_c)])


    return k_folds

class RandomForest():
    """Class for random forest
    classification."""

    def __init__(self, num_trees, depth_limit, example_subsample_rate, attr_subsample_rate):
        """Create a random forest with a fixed 
        number of trees, depth limit, example
        sub-sample rate and attribute sub-sample
        rate."""
        self.trees = []
        self.num_trees = num_trees
        self.depth_limit = depth_limit
        self.example_subsample_rate = example_subsample_rate
        self.attr_subsample_rate = attr_subsample_rate
        self.attrs = []

    def fit(self, features, classes):
        """Build a random forest of 
        decision trees."""

        num_examples = len(features)
        full_classes = np.asarray(classes)
        for i in range(self.num_trees):
            example_idxs = np.random.choice(features.shape[0], int(self.example_subsample_rate * features.shape[0]))

            _f = features[example_idxs]
            sample_c = full_classes[example_idxs]
            num_attrs = int(self.attr_subsample_rate * features.shape[1])
            idx = np.random.randint(features.shape[1], size=num_attrs)
            sample_f = _f[:,idx]

            tree = DecisionTree(self.depth_limit)
            tree.fit(sample_f,sample_c)
            self.trees.append(tree)
            self.attrs.append(idx)

    def classify(self, features):
        """Classify a list of features based
        on the trained random forest."""

        classifier_output = []
        for index, tree in enumerate(self.trees):
            feat_by_attr = features[:,self.attrs[index]]
            classifications = tree.classify(feat_by_attr)
            classifier_output.append(classifications)

        class_matrix =  np.column_stack(classifier_output)
        return np.apply_along_axis(most_frequent_class, 1, class_matrix)

class ChallengeClassifier():
    """Class for random forest
    classification."""

    def __init__(self,num_trees=7, depth_limit=3, example_subsample_rate=.4, attr_subsample_rate=.5):
        """Create a random forest with a fixed
        number of trees, depth limit, example
        sub-sample rate and attribute sub-sample
        rate."""
        self.trees = []
        self.num_trees = num_trees
        self.depth_limit = depth_limit
        self.example_subsample_rate = example_subsample_rate
        self.attr_subsample_rate = attr_subsample_rate
        self.attrs = []
        
    def fit(self, features, classes):
        """Build a random forest of
        decision trees."""

        num_examples = len(features)
        full_classes = np.asarray(classes)
        for i in range(self.num_trees):
            example_idxs = np.random.choice(features.shape[0], int(self.example_subsample_rate * features.shape[0]))

            _f = features[example_idxs]
            sample_c = full_classes[example_idxs]
            num_attrs = int(self.attr_subsample_rate * features.shape[1])
            idx = np.random.randint(features.shape[1], size=num_attrs)
            sample_f = _f[:,idx]

            tree = DecisionTree(self.depth_limit)
            tree.fit(sample_f,sample_c)
            self.trees.append(tree)
            self.attrs.append(idx)
        
    def classify(self, features):
        classifier_output = []
        for index, tree in enumerate(self.trees):
            feat_by_attr = features[:,self.attrs[index]]
            classifications = tree.classify(feat_by_attr)
            classifier_output.append(classifications)

        class_matrix =  np.column_stack(classifier_output)
        return np.apply_along_axis(most_frequent_class, 1, class_matrix)