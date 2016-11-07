import decision_trees as dt
import numpy as np
import pickle

def test_tree():
  return dt.build_decision_tree()

def validate_tree(tree,examples, classes):
  tree_root = dt.build_decision_tree()

  classifier_output = [tree_root.decide(example) for example in examples]
  p1_accuracy = dt.accuracy(classifier_output, classes)
  p1_precision = dt.precision(classifier_output, classes)
  p1_recall = dt.recall(classifier_output, classes)
  p1_confusion_matrix = dt.confusion_matrix(classifier_output, classes)
  # dt.information_gain(examples,)
  print "p1 accuracy = ", p1_accuracy
  print "p1 precision = ", p1_precision
  print "p1 recall = ", p1_recall
  print "p1 confusion matrix = ", p1_confusion_matrix

def test_information_gain():
   """ Assumes information_gain() accepts (classes, [list of subclasses])
       Feel free to edit / enhance this note with more tests """
   restaurants = [0]*6 + [1]*6
   split_patrons =   [[0,0], [1,1,1,1], [1,1,0,0,0,0]]
   split_food_type = [[0,1],[0,1],[0,0,1,1],[0,0,1,1]]
   # If you're using numpy indexing add the following before calling information_gain()
   # split_patrons =   [np.array(i) for i in split_patrons]   #convert to np array
   # split_food_type = [np.array(i) for i in split_food_type]

   gain_patrons = dt.information_gain(restaurants, split_patrons)
   gain_type = dt.information_gain(restaurants, split_food_type)
   assert round(gain_patrons,3) == 0.541, "Information Gain on patrons should be 0.541"
   assert gain_type == 0.0, "Information gain on type should be 0.0"
   print "Information Gain calculations correct..."

def test_decision_tree(f,c):
    tree = dt.DecisionTree()
    tree.fit(f,c)
    validate_tree(tree,f,c)
def test_k_folds():
    dataset = dt.load_csv('part2_data.csv')
    ten_folds = dt.generate_k_folds(dataset, 10)

    accuracies = []
    precisions = []
    recalls = []
    confusion = []

    for fold in ten_folds:
        train, test = fold
        train_features, train_classes = train
        test_features, test_classes = test
        tree = dt.DecisionTree()
        tree.fit(train_features, train_classes)
        output = tree.classify(test_features)

        accuracies.append( dt.accuracy(output, test_classes))
        precisions.append( dt.precision(output, test_classes))
        recalls.append( dt.recall(output, test_classes))
        confusion.append( dt.confusion_matrix(output, test_classes))
    print np.mean(accuracies)

def test_forests():
    dataset = dt.load_csv('part2_data.csv')
    forest = dt.RandomForest(5,5,.5,.5)
    forest.fit(dataset[0],dataset[1])

    ten_folds = dt.generate_k_folds(dataset, 10)

    accuracies = []
    precisions = []
    recalls = []
    confusion = []

    for fold in ten_folds:
        train, test = fold
        train_features, train_classes = train
        test_features, test_classes = test
        output = forest.classify(test_features)
        accuracies.append( dt.accuracy(output, test_classes))
        precisions.append( dt.precision(output, test_classes))
        recalls.append( dt.recall(output, test_classes))
        confusion.append( dt.confusion_matrix(output, test_classes))
    print np.mean(accuracies)

def test_challenge():
    dataset = pickle.load(open('challenge_data.pickle', 'rb'))
    ch = dt.ChallengeClassifier()
    ch.fit(np.asarray(dataset[0]),np.asarray(dataset[1]))
    ten_folds = dt.generate_k_folds(dataset, 10)

    accuracies = []
    precisions = []
    recalls = []
    confusion = []

    for fold in ten_folds:
        train, test = fold
        train_features, train_classes = train
        test_features, test_classes = test
        output = ch.classify(test_features)
        accuracies.append( dt.accuracy(output, test_classes))
        precisions.append( dt.precision(output, test_classes))
        recalls.append( dt.recall(output, test_classes))
        confusion.append( dt.confusion_matrix(output, test_classes))

    print np.mean(accuracies)

def main():
  examples = [[1.,0.,0.,0.],
            [1.,0.,1.,1.],
            [0.,1.,0.,0.],
            [0.,1.,1.,0.],
            [1.,1.,0.,1.],
            [0.,1.,0.,1.],
            [0.,0,1.,1.],
            [0.,0.,1.,0.]]
  classes = [1,1,1,0,1,0,1,0]
  # examples [[0,1,0,0],[0,1,0,1]]
  # classes = [1,1]


  # validate_tree(test_tree(), examples, classes)
  # test_information_gain()
  test_decision_tree(examples,classes)
  # test_k_folds()
  # test_forests()
  # test_challenge()

if __name__ == '__main__':
  main()