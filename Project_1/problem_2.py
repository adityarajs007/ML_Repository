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
FEATURES_END = 13
FEATURES_START = 0
TREE_DEPTH = 5
PRCPTRN_MAX_ITERATIONS = 7
LR_C_VAL = .25
SVM_C_VAL = .25
RF_TREES = 5 
KNN_NEIGHBORS = 5 

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
def perceptron(x_trn_std, x_tst_std, y_trn, y_tst, iterations):
    # create the classifier
    prcptrn = Perceptron(max_iter=iterations, tol=1e-3, eta0=0.001,
                     fit_intercept=True, random_state=0, verbose=True)
    prcptrn.fit(x_trn_std, y_trn)                   # training the model

    y_pred = prcptrn.predict(x_tst_std)             # try to predict with the test data
    test_acc = accuracy_score(y_tst, y_pred)

    # combine the train and test data
    X_combined_std = np.vstack((x_trn_std, x_tst_std))
    y_combined = np.hstack((y_trn, y_tst))

    # we did the stack so we can see how the combination of test and train data did
    y_combined_pred = prcptrn.predict(X_combined_std)
    combined_samples = (y_combined != y_combined_pred).sum()
    combined_acc = accuracy_score(y_combined, y_combined_pred)

    # print analysis result
    print_results(len(y_tst), (y_tst != y_pred).sum(), test_acc, 
                  len(y_combined), combined_samples, combined_acc)

###############################################################
#Code for Logistic Regression
##############################################################@
def logistic_regression(x_trn_std, x_tst_std, y_trn, y_tst, c_val):
    # create the classifier
    lr = LogisticRegression(C=c_val, solver='liblinear',
                            multi_class='ovr', random_state=0)
    lr.fit(x_trn_std, y_trn)                    # training the model

    y_pred = lr.predict(x_tst_std)              # try to predict with the test data
    test_acc = accuracy_score(y_tst, y_pred)

    # combine the train and test data
    X_combined_std = np.vstack((x_trn_std, x_tst_std))
    y_combined = np.hstack((y_trn, y_tst))

    # we did the stack so we can see how the combination of test and train data did
    y_combined_pred = lr.predict(X_combined_std)
    combined_samples = (y_combined != y_combined_pred).sum()
    combined_acc = accuracy_score(y_combined, y_combined_pred)

    # print analysis result
    print_results(len(y_tst), (y_tst != y_pred).sum(
    ), test_acc, len(y_combined), combined_samples, combined_acc)

###############################################################
#Code for Support Vector Machine
##############################################################@
def support_vector_machine(x_trn_std, x_tst_std, y_trn, y_tst, c_val):
    # create the classifier
    svm = SVC(kernel='linear', C=c_val, random_state=0)
    svm.fit(x_trn_std, y_trn)                   # training the model

    y_pred = svm.predict(x_tst_std)             # try to predict with the test data
    test_acc = accuracy_score(y_tst, y_pred)

    # combine the train and test data
    X_combined_std = np.vstack((x_trn_std, x_tst_std))
    y_combined = np.hstack((y_trn, y_tst))

    # we did the stack so we can see how the combination of test and train data did
    y_combined_pred = svm.predict(X_combined_std)
    combined_samples = (y_combined != y_combined_pred).sum()
    combined_acc = accuracy_score(y_combined, y_combined_pred)

    # print analysis result
    print_results(len(y_tst), (y_tst != y_pred).sum(
    ), test_acc, len(y_combined), combined_samples, combined_acc)

###############################################################
#Code for Decision Tree
##############################################################@
def decision_tree(x_trn, x_tst, y_trn, y_tst, depth, cols):
    # create the classifier
    tree = DecisionTreeClassifier(criterion='entropy', max_depth=depth, random_state=0)
    tree.fit(x_trn, y_trn)                      # training the model

    y_pred = tree.predict(x_tst)                # try to predict with the test data
    test_acc = accuracy_score(y_tst, y_pred)

    # combine the train and test data
    X_combined = np.vstack((x_trn, x_tst))
    y_combined = np.hstack((y_trn, y_tst))

    # we did the stack so we can see how the combination of test and train data did
    y_combined_pred = tree.predict(X_combined)
    combined_samples = (y_combined != y_combined_pred).sum()
    combined_acc = accuracy_score(y_combined, y_combined_pred)

    # print analysis result
    print_results(len(y_tst), (y_tst != y_pred).sum(), test_acc, 
                  len(y_combined), combined_samples, combined_acc)
    export_graphviz(tree, out_file='tree.dot', feature_names=cols)

###############################################################
#Code for Random Forest
##############################################################@
def random_forest(x_trn, x_tst, y_trn, y_tst, trees):
    # create the classifier
    forest = RandomForestClassifier(
        criterion='entropy', n_estimators=trees, random_state=1, n_jobs=4)
    forest.fit(x_trn, y_trn)                    # training the model

    y_pred = forest.predict(x_tst)              # try to predict with the test data
    test_acc = accuracy_score(y_tst, y_pred)

    # combine the train and test data
    X_combined = np.vstack((x_trn, x_tst))
    y_combined = np.hstack((y_trn, y_tst))

    # we did the stack so we can see how the combination of test and train data did
    y_combined_pred = forest.predict(X_combined)
    combined_samples = (y_combined != y_combined_pred).sum()
    combined_acc = accuracy_score(y_combined, y_combined_pred)

    # print analysis result
    print_results(len(y_tst), (y_tst != y_pred).sum(
    ), test_acc, len(y_combined), combined_samples, combined_acc)

###############################################################
#Code for K-nearest neighbors
##############################################################@
def k_nearest(x_trn_std, x_tst_std, y_trn, y_tst, neighs):
    knn = KNeighborsClassifier(n_neighbors=neighs, p=2, metric='minkowski') # create the classifier
    knn.fit(x_trn_std, y_trn)                   # training the model

    y_pred = knn.predict(x_tst_std)             # try to predict with the test data
    test_acc = accuracy_score(y_tst, y_pred)

    # combine the train and test data
    X_combined_std = np.vstack((x_trn_std, x_tst_std))
    y_combined = np.hstack((y_trn, y_tst))

    # we did the stack so we can see how the combination of test and train data did
    y_combined_pred = knn.predict(X_combined_std)
    combined_samples = (y_combined != y_combined_pred).sum()
    combined_acc = accuracy_score(y_combined, y_combined_pred)

    # print analysis result
    print_results(len(y_tst), (y_tst != y_pred).sum(
    ), test_acc, len(y_combined), combined_samples, combined_acc)

def main():
    # load the data set
    df = pd.read_csv('heart1.csv')
    print(f'csv file was successfully read')

    # convert data frame to numpy array
    numpy_df = df.to_numpy()

    # split the data to training & testing datasets
    X = numpy_df[:,:FEATURES_END]               # separate all the features
    y = numpy_df[:, FEATURES_END].ravel()       # extract the classifications

    # split the problem into test & train dataset- 70%-train & 30%-test
    X_train, X_test, y_train, y_test = train_test_split( # 70% training and 30% test
        X, y, test_size=0.3, random_state=0) # random_state allows the split to be reproduced

    # mean and standard deviation may be overridden with options
    sc = StandardScaler()                       
    sc.fit(X_train)                            
    X_train_std = sc.transform(X_train)         
    X_test_std = sc.transform(X_test)     #apply standardisation to the test and train data
    
    
    print('\n\n ############ perceptron ############')
    perceptron(X_train_std, X_test_std, y_train, y_test, PRCPTRN_MAX_ITERATIONS)

    
    print('\n\n ############ logistic regression ############')
    logistic_regression(X_train_std, X_test_std, y_train, y_test, LR_C_VAL)
 
    print('\n\n ############ support vector machine ############')
    support_vector_machine(X_train_std, X_test_std, y_train, y_test, SVM_C_VAL)

    
    print('\n\n ########## decision tree ############')
    decision_tree(X_train, X_test, y_train, y_test, TREE_DEPTH, 
    df.columns.values[FEATURES_START:FEATURES_END])
 
    print('\n\n ########## random forest ############')
    random_forest(X_train, X_test, y_train, y_test, RF_TREES)

    print('\n\n ########## k-nearest neighbors ##########')
    k_nearest(X_train, X_test, y_train, y_test, KNN_NEIGHBORS)

main()