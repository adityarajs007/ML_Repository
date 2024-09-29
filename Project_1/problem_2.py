############# Author Info ###################################
# Adityaraj Suresh
# EEE 591
# Project1- Problem 2
# asures56@asu.edu
#############################################################

import numpy as np                                      
import pandas as pd
from sklearn.svm import SVC  
from sklearn.tree import export_graphviz
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression    
from sklearn.tree import DecisionTreeClassifier                                      
from sklearn.model_selection import train_test_split

# defining hyperparameters

TREE_DEPTH = 5               # Maximum depth of the decision tree
KNN_NEIGHBORS = 5            # Number of neighbors for KNN
NUM_FEATURES = 13            # Number of features in the dataset
FIRST_FEATURE_INDEX = 0      # Starting index for features         
PERCEPTRON_MAX_EPOCHS = 7    # Maximum number of iterations for the Perceptron
LOGISTIC_REG_C = 0.25        # Regularization parameter for Logistic Regression
SVM_REGULARIZATION = 0.25    # Regularization parameter for SVM
RANDOM_FOREST_TREES = 5      # Number of trees in the Random Forest      

############################################
# Code for print method to keep uniform printing format for all methods
##########################################
def print_results(test_sam, test_miss, test_acc, combined_sam, combined_miss, combined_acc):
    print('Number in test: ', test_sam)
    print('Misclassified samples: %d' % test_miss)
    print('Accuracy: %.2f' % test_acc)
    print('Number in combined: ', combined_sam)
    print('Misclassified combined samples: %d' % combined_miss)
    print('Combined Accuracy: %.2f' % combined_acc)

###############################################################
#Code for Perceptron
##############################################################@
def perceptron(x_train_std, x_test_std, y_train, y_test, iterations):
    # create the classifier
    prcptrn = Perceptron(max_iter=iterations, tol=1e-3, eta0=0.001,
                     fit_intercept=True, random_state=0, verbose=True)
    prcptrn.fit(x_train_std, y_train)                   # training the model

    y_pred = prcptrn.predict(x_test_std)             # try to predict with the test data
    test_acc = accuracy_score(y_test, y_pred)

    # combine the train and test data
    X_combined_std = np.vstack((x_train_std, x_test_std))
    y_combined = np.hstack((y_train, y_test))

    # we did the stack so we can see how the combination of test and train data did
    y_combined_pred = prcptrn.predict(X_combined_std)
    combined_samples = (y_combined != y_combined_pred).sum()
    combined_acc = accuracy_score(y_combined, y_combined_pred)

    # print analysis result
    print_results(len(y_test), (y_test != y_pred).sum(), test_acc, 
                  len(y_combined), combined_samples, combined_acc)

###############################################################
#Code for Logistic Regression
##############################################################@
def logistic_regression(x_train_std, x_test_std, y_train, y_test, c_val):
    # create the classifier
    lr = LogisticRegression(C=c_val, solver='liblinear',
                            multi_class='ovr', random_state=0)
    lr.fit(x_train_std, y_train)                    # training the model

    y_pred = lr.predict(x_test_std)              # try to predict with the test data
    test_acc = accuracy_score(y_test, y_pred)

    # combine the train and test data
    X_combined_std = np.vstack((x_train_std, x_test_std))
    y_combined = np.hstack((y_train, y_test))

    # we did the stack so we can see how the combination of test and train data did
    y_combined_pred = lr.predict(X_combined_std)
    combined_samples = (y_combined != y_combined_pred).sum()
    combined_acc = accuracy_score(y_combined, y_combined_pred)

    # print analysis result
    print_results(len(y_test), (y_test != y_pred).sum(
    ), test_acc, len(y_combined), combined_samples, combined_acc)

###############################################################
#Code for Support Vector Machine
##############################################################@
def support_vector_machine(x_train_std, x_test_std, y_train, y_test, c_val):
    # create the classifier
    svm = SVC(kernel='linear', C=c_val, random_state=0)
    svm.fit(x_train_std, y_train)                   # training the model

    y_pred = svm.predict(x_test_std)             # try to predict with the test data
    test_acc = accuracy_score(y_test, y_pred)

    # combining the train and test data
    X_combined_std = np.vstack((x_train_std, x_test_std))
    y_combined = np.hstack((y_train, y_test))

    # we did the stack so we can see how the combination of test and train data did
    y_combined_pred = svm.predict(X_combined_std)
    combined_samples = (y_combined != y_combined_pred).sum()
    combined_acc = accuracy_score(y_combined, y_combined_pred)

    # print analysis result
    print_results(len(y_test), (y_test != y_pred).sum(
    ), test_acc, len(y_combined), combined_samples, combined_acc)

###############################################################
#Code for Decision Tree
##############################################################@
def decision_tree(x_train, x_test, y_train, y_test, depth, cols):
    # create the classifier
    tree = DecisionTreeClassifier(criterion='entropy', max_depth=depth, random_state=0)
    tree.fit(x_train, y_train)                      # training the model

    y_pred = tree.predict(x_test)                # try to predict with the test data
    test_acc = accuracy_score(y_test, y_pred)

    # combine the train and test data
    X_combined = np.vstack((x_train, x_test))
    y_combined = np.hstack((y_train, y_test))

    # we did the stack so we can see how the combination of test and train data did
    y_combined_pred = tree.predict(X_combined)
    combined_samples = (y_combined != y_combined_pred).sum()
    combined_acc = accuracy_score(y_combined, y_combined_pred)

    # print analysis result
    print_results(len(y_test), (y_test != y_pred).sum(), test_acc, 
                  len(y_combined), combined_samples, combined_acc)
    export_graphviz(tree, out_file='tree.dot', feature_names=cols)

###############################################################
#Code for Random Forest
##############################################################@
def random_forest(x_trn, x_tst, y_train, y_test, trees):
    # create the classifier
    forest = RandomForestClassifier(
        criterion='entropy', n_estimators=trees, random_state=1, n_jobs=4)
    forest.fit(x_trn, y_train)                    # training the model

    y_pred = forest.predict(x_tst)              # try to predict with the test data
    test_acc = accuracy_score(y_test, y_pred)

    # combine the train and test data
    X_combined = np.vstack((x_trn, x_tst))
    y_combined = np.hstack((y_train, y_test))

    # we did the stack so we can see how the combination of test and train data did
    y_combined_pred = forest.predict(X_combined)
    combined_samples = (y_combined != y_combined_pred).sum()
    combined_acc = accuracy_score(y_combined, y_combined_pred)

    # print analysis result
    print_results(len(y_test), (y_test != y_pred).sum(
    ), test_acc, len(y_combined), combined_samples, combined_acc)

###############################################################
#Code for K-nearest neighbors
##############################################################@
def k_nearest(x_train_std, x_test_std, y_train, y_test, neighs):
    knn = KNeighborsClassifier(n_neighbors=neighs, p=2, metric='minkowski') # create the classifier
    knn.fit(x_train_std, y_train)                   # training the model

    y_pred = knn.predict(x_test_std)             # try to predict with the test data
    test_acc = accuracy_score(y_test, y_pred)

    # combine the train and test data
    X_combined_std = np.vstack((x_train_std, x_test_std))
    y_combined = np.hstack((y_train, y_test))

    # we did the stack so we can see how the combination of test and train data did
    y_combined_pred = knn.predict(X_combined_std)
    combined_samples = (y_combined != y_combined_pred).sum()
    combined_acc = accuracy_score(y_combined, y_combined_pred)

    # print analysis result
    print_results(len(y_test), (y_test != y_pred).sum(
    ), test_acc, len(y_combined), combined_samples, combined_acc)

def main():
    # load the data set
    df = pd.read_csv('heart1.csv')
    print(f'csv file was successfully read')

    # convert data frame to numpy array
    numpy_df = df.to_numpy()

    # split the data to training & testing datasets
    X = numpy_df[:,:NUM_FEATURES]               # separate all the features
    y = numpy_df[:, NUM_FEATURES].ravel()       # extract the classifications

    # split the problem into test & train dataset
    X_train, X_test, y_train, y_test = train_test_split( # 70% training and 30% test
        X, y, test_size=0.3, random_state=0) # random_state allows the split to be reproduced

    # mean and standard deviation may be overridden with options
    sc = StandardScaler()                       
    sc.fit(X_train)                            
    X_train_std = sc.transform(X_train)         
    X_test_std = sc.transform(X_test)     #apply standardisation to the test and train data
    
    
    print('\n\n############ perceptron ############')
    perceptron(X_train_std, X_test_std, y_train, y_test, PERCEPTRON_MAX_EPOCHS)

    
    print('\n\n############ logistic regression ############')
    logistic_regression(X_train_std, X_test_std, y_train, y_test, LOGISTIC_REG_C)
 
    print('\n\n############ support vector machine ############')
    support_vector_machine(X_train_std, X_test_std, y_train, y_test, SVM_REGULARIZATION)

    
    print('\n\n########## decision tree ############')
    decision_tree(X_train, X_test, y_train, y_test, TREE_DEPTH, 
    df.columns.values[FIRST_FEATURE_INDEX:NUM_FEATURES])
 
    print('\n\n########## random forest ############')
    random_forest(X_train, X_test, y_train, y_test, RANDOM_FOREST_TREES)

    print('\n\n########## k-nearest neighbors ##########')
    k_nearest(X_train, X_test, y_train, y_test, KNN_NEIGHBORS)

main()