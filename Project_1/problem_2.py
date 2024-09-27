import numpy as np                                      
import pandas as pd
from sklearn.tree import export_graphviz
from sklearn.metrics import accuracy_score 
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier                                      
from sklearn.model_selection import train_test_split

# defining hyperparameters
FEATURES_END = 13
FEATURES_START = 0
TREE_DEPTH = 5

def print_results(test_sam, test_miss, test_acc, combined_sam, combined_miss, combined_acc):
    print('\nNumber in test: ', test_sam)
    print('Misclassified samples: %d' % test_miss)
    print('Accuracy: %.2f' % test_acc)
    print('Number in combined: ', combined_sam)
    print('Misclassified combined samples: %d' % combined_miss)
    print('Combined Accuracy: %.2f' % combined_acc)

def decision_tree(x_trn, x_tst, y_trn, y_tst, depth, cols):
    # create the classifier
    tree = DecisionTreeClassifier(criterion='entropy', max_depth=depth, random_state=0)
    tree.fit(x_trn, y_trn)                      # do the training

    y_pred = tree.predict(x_tst)                # now try with test data
    test_acc = accuracy_score(y_tst, y_pred)

    # combine the train and test data
    X_combined = np.vstack((x_trn, x_tst))
    y_combined = np.hstack((y_trn, y_tst))

    # we did the stack so we can see how the combination of test and train data did
    y_combined_pred = tree.predict(X_combined)
    combined_samples = (y_combined != y_combined_pred).sum()
    combined_acc = accuracy_score(y_combined, y_combined_pred)

    # print analysis result
    print_results(len(y_tst), (y_tst != y_pred).sum(
    ), test_acc, len(y_combined), combined_samples, combined_acc)
    export_graphviz(tree, out_file='tree.dot', feature_names=cols)

def main():
    # load the data set
    df = pd.read_csv('heart1.csv')
    print(f'csv file was successfully read')

    # convert data frame to numpy array
    numpy_df = df.to_numpy()

    # split the data to training & testing datasets
    X = numpy_df[:,:FEATURES_END]               # separate all the features
    y = numpy_df[:, FEATURES_END].ravel()       # extract the classifications

    # split the problem into train and test: 70% training and 30% test
    # random_state allows the split to be reproduced
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0)

    # mean and standard deviation may be overridden with options
    sc = StandardScaler()                       # create the standard scalar
    sc.fit(X_train)                             # compute the required transformation
    X_train_std = sc.transform(X_train)         # apply to the training data
    X_test_std = sc.transform(X_test)           # and SAME transformation of test data
    print('decision tree')
    decision_tree(X_train, X_test, y_train, y_test, TREE_DEPTH, 
    df.columns.values[FEATURES_START:FEATURES_END])

main()