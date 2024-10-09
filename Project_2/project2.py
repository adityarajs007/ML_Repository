import numpy as np                                      # For array manipulation
import pandas as pd                                     # For handling dataframes
import matplotlib.pyplot as plt                         # For plotting graphs
from warnings import filterwarnings                     # To suppress unwanted warnings
from sklearn.decomposition import PCA                   # For Principal Component Analysis
from sklearn.metrics import accuracy_score, confusion_matrix  # For evaluation metrics
from sklearn.neural_network import MLPClassifier        # Multilayer Perceptron Classifier
from sklearn.preprocessing import StandardScaler        # For feature scaling
from sklearn.model_selection import train_test_split    # For splitting the dataset

# File containing the sonar data
FILE_NAME = 'sonar_all_data_2.csv'

# Parameters
TOTAL_COLUMNS = 62           # Total number of columns in the dataset
NUM_FEATURES = TOTAL_COLUMNS - 2  # Number of feature columns (excluding label)

# pca_analysis:: Perform PCA and train MLPClassifier, then evaluate performance
def pca_analysis(X_train_scaled, X_test_scaled, y_train, y_test):
    accuracies = []      # To store accuracy for each component count
    confusion_matrices = []  # To store confusion matrices for each component count
    
    print(f"{'Components'} \t {'Accuracy'}")
    print('-' * 22)

    # Test various numbers of principal components, from 1 to MAX_COMPONENTS
    for n in range(1, NUM_FEATURES + 1): # One-based indexing for looping over components
        # Applying PCA for dimensionality reduction
        pca = PCA(n_components=n)
        X_train_pca = pca.fit_transform(X_train_scaled)  # Fit and transform training data
        X_test_pca = pca.transform(X_test_scaled)        # Transform test data

        # Train a neural network on the PCA-transformed data
        mlp = MLPClassifier(hidden_layer_sizes=(200, 100), activation='relu', max_iter=200, 
                            solver='adam', alpha=0.0001, tol=0.001, random_state=1)
        mlp.fit(X_train_pca, y_train)  # Training the MLP

        # Predict on the test set and calculate accuracy
        y_pred = mlp.predict(X_test_pca)
        accuracy = accuracy_score(y_test, y_pred)

        # Store accuracy and confusion matrix
        accuracies.append(accuracy)
        confusion_matrices.append(confusion_matrix(y_test, y_pred))

        # Print results for this iteration
        print(f'{n} \t {round(accuracy, 2)}')

    return accuracies, confusion_matrices

# main:: Main function to load data, perform PCA analysis and evaluate results
def main():
    # Load the data from CSV
    df = pd.read_csv(FILE_NAME, header=None)
    print(f'Data loaded from  csv file \n')

    # Separate features and labels
    X = df.iloc[:, :NUM_FEATURES].values  # Features are in columns 0 to 59
    y = df.iloc[:, NUM_FEATURES].values   # Labels are in the last column

    # Split data into training and testing sets (70% train, 30% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    # Standardize the feature data (zero mean, unit variance)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Perform PCA and evaluate using MLP
    accuracies, confusion_matrices = pca_analysis(X_train_scaled, X_test_scaled, y_train, y_test)

    # Find the best accuracy and the corresponding number of components
    best_accuracy = np.max(accuracies)
    best_n_components = accuracies.index(best_accuracy) + 1  # Adjust for 0-based index

    print(f'\nBest accuracy: {round(best_accuracy, 2)} with {best_n_components} components')

    # Plot the accuracy against the number of PCA components
    components = range(1, NUM_FEATURES + 1) # One-based indexing for looping over components
    
    # to plot components vs accuracy 
    plt.plot(components, accuracies, marker='o', linestyle='-', color='g')
    plt.title("PCA Components vs. Test Accuracy")
    plt.xlabel("Number of PCA Components")
    plt.ylabel("Test Accuracy")
    plt.grid(True)
    plt.show()

    # Print the confusion matrix for the optimal number of components
    print(f'\nConfusion matrix for {best_n_components} components:')
    print(confusion_matrices[best_n_components - 1])

main()
