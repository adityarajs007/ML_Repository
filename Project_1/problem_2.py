import numpy as np                                      
import pandas as pd
from sklearn.preprocessing import StandardScaler                                      
from sklearn.model_selection import train_test_split

FEATURES_END = 13
FEATURES_START = 0


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
main()