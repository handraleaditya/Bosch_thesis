import numpy as np
from sklearn.model_selection import train_test_split

# importing classifier sto be used
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score, StratifiedKFold # for cross validation

# calculate the metrics for the ML models  
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# Imports for building a neural network
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras.layers import Dropout
from keras.optimizers import Adam


"""
    A class for evaluating machine learning models (SVM, k-NN, Random Forest, and Neural Network) on a feature space.
    It also supports cross-validation for model evaluation.

    Attributes:
        model_name (str): The name of the machine learning model.
        features (numpy.ndarray): The feature space data used for evaluation.
        labels (numpy.ndarray): The true class labels for the feature space.
        low_dim_features (numpy.ndarray): Low-dimensional feature space for visualization.
        X_train (numpy.ndarray): Training features.
        X_test (numpy.ndarray): Testing features.
        y_train (numpy.ndarray): Training labels.
        y_test (numpy.ndarray): Testing labels.

    Methods:
        __init__(self, model, data_loader, feature_visualizer, test_size=0.30, verbose=False):
            Initializes a ModelEvaluator instance with the provided model, data loader, and feature visualizer.

        evaluate_random_forest(self, n_estimators=100):
            Evaluates the Random Forest classifier on the feature space and prints metrics.

        print_metrics(self, title, predictions):
            Prints metrics including accuracy, precision, recall, and F1 score for a given model.

        evaluate(self):
            Evaluates multiple machine learning models (SVM, k-NN, Random Forest, and Neural Network) on the feature space.

        evaluate_with_cross_validation(self, n_splits=10):
            Evaluates models with cross-validation and prints the accuracy for each model.

        evaluate_low_dim_with_cross_validation(self, n_splits=10):
            Evaluates models with cross-validation on low-dimensional feature space and prints the accuracy for each model.

    Usage:
    # Example Usage
    data_loader = DataLoader(...)  # Replace with your data loading logic
    feature_visualizer = FeatureSpaceVisualizer(...)
    model = CustomModel(...)
    evaluator = ModelEvaluator(model, data_loader, feature_visualizer, test_size=0.30, verbose=True)
    evaluator.evaluate()
    evaluator.evaluate_with_cross_validation(n_splits=10)
    evaluator.evaluate_low_dim_with_cross_validation(n_splits=10)
    """
class ModelEvaluator:
    
    def __init__(self, model, data_loader, feature_visualizer,  test_size=0.30, verbose=False):

        self.verbose = verbose # for print brevity
        self.model_name = model.model_name
        self.features = model.features
        self.labels= data_loader.labels
        self.low_dim_features = feature_visualizer.low_dim_feature_space 
        # divide the dataset into test train split with 
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.features, self.labels, test_size=test_size, random_state=42
        )

    # build and evaluate SVM 
    def evaluate_svm(self):
        svm_classifier = SVC(kernel='linear', C=1.0)
        svm_classifier.fit(self.X_train, self.y_train)
        svm_predictions = svm_classifier.predict(self.X_test)
        if self.verbose:
            self.print_metrics("SVM Classifier Metrics:", svm_predictions)
        else:
            self.print_short_metrics("SVM Classifier", svm_predictions)
        

    # Build and evaluate K nearest neighbors
    def evaluate_knn(self):
        n_neighbors = int(np.sqrt(len(self.X_train)))
        knn_classifier = KNeighborsClassifier(n_neighbors=n_neighbors)
        knn_classifier.fit(self.X_train, self.y_train)
        knn_predictions = knn_classifier.predict(self.X_test)
        if self.verbose:
            self.print_metrics("k-NN Classifier Metrics:", knn_predictions)
        else: 
            self.print_short_metrics("k-NN Classifier", knn_predictions)

    #  Build and train the neural network
    def build_nn(self):

        optimizer = Adam(learning_rate=0.001)
        num_classes = len(np.unique(self.labels))
        print('num claasees :',num_classes)
        # Define a neural network model for multi-class classification
        model = Sequential()
        model.add(Dense(512, input_dim=self.X_train.shape[1], activation='relu'))
        # model.add(Dense(512, activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(num_classes, activation='softmax'))  # Use 'softmax' for multi-class
        
        # Compile the model
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        
        return model
        
    #  Evaluate the neural network
    def evaluate_nn(self):
        y_train_categorical = to_categorical(self.y_train)
        y_test_categorical = to_categorical(self.y_test)
        model = self.build_nn()
    
        # Train the model
        model.fit(self.X_train, y_train_categorical, epochs=30, batch_size=32, verbose=1)
        
        # Make predictions on the test data
        nn_pred_onehot = model.predict(self.X_test)
        nn_pred = np.argmax(nn_pred_onehot, axis=1)  # Convert one-hot to class labels

        if self.verbose:
            self.print_metrics("Neural Network Classifier Metrics:", nn_pred)
        else: 
            self.print_short_metrics("Neural Network Classifier", nn_pred)

        return model

         
    
    def evaluate_random_forest(self, n_estimators=100):
        rf_classifier = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        rf_classifier.fit(self.X_train, self.y_train)
        rf_predictions = rf_classifier.predict(self.X_test)
        if self.verbose:
            self.print_metrics("Random Forest Classifier Metrics:", rf_predictions)
        else:
            self.print_short_metrics("Random Forest Classifier", rf_predictions)
        

    def print_metrics(self, title, predictions):
        accuracy = accuracy_score(self.y_test, predictions)
        precision = precision_score(self.y_test, predictions, average='weighted')
        recall = recall_score(self.y_test, predictions, average='weighted')
        f1 = f1_score(self.y_test, predictions, average='weighted')
        confusion = confusion_matrix(self.y_test, predictions)

        print(title)
        print(f"Accuracy: {accuracy*100:.2f}%")
        print(f"Precision: {precision*100:.2f}%")
        print(f"Recall: {recall*100:.2f}%")
        print(f"F1 Score: {f1*100:.2f}%")
        print("Confusion Matrix:\n", confusion,'\n')

    def print_short_metrics(self, title, predictions):
        f1 = f1_score(self.y_test, predictions, average='weighted')
        print(f"{title} F1 Score: {f1*100:.2f}%")
        

    def evaluate(self):
        print(f'-----------------------{self.model_name}----------------------------')

        # evaluate the NN model
        self.evaluate_nn()
        # Evaluate SVM Classifier
        self.evaluate_svm()
        
        # Evaluate k-NN Classifier
        self.evaluate_knn()
        
        # Evaluate Random Forest Classifier
        self.evaluate_random_forest()


    # Evaluate with cross validation, 10 folds for all the ML models
    def evaluate_with_cross_validation(self, n_splits=10):
        ml_performances ={}
        # Initialize cross-validation
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        # Evaluate SVM Classifier with cross-validation
        svm_classifier = SVC(kernel='linear', C=1.0)
        svm_scores = cross_val_score(svm_classifier, self.features, self.labels, cv=cv, scoring='accuracy')
        print("SVM Classifier Cross-Validation Accuracy: {:.2f}%".format(np.mean(svm_scores) * 100))
        
        # Evaluate k-NN Classifier with cross-validation
        knn_classifier = KNeighborsClassifier(n_neighbors=int(np.sqrt(len(self.X_train))))
        knn_scores = cross_val_score(knn_classifier, self.features, self.labels, cv=cv, scoring='accuracy')
        print("k-NN Classifier Cross-Validation Accuracy: {:.2f}%".format(np.mean(knn_scores) * 100))

        # Evaluate Random Forest Classifier with cross-validation
        rf_classifier = RandomForestClassifier(n_estimators=10, random_state=42)
        rf_scores = cross_val_score(rf_classifier, self.features, self.labels, cv=cv, scoring='accuracy')
        print("Random Forest Classifier Cross-Validation Accuracy: {:.2f}%".format(np.mean(rf_scores) * 100))

        # # Evaluate NN Classifier with cross-validation
        nn_scores = []
        
        # For each of the split, train and evaluate models
        for train_index, val_index in cv.split(self.features, self.labels):
            X = self.features
            y = self.labels
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]
        
            # Define and compile your neural network model
            model = self.build_nn()
        
            # Convert labels to one-hot encoding
            y_train_categorical = to_categorical(y_train)
            y_val_categorical = to_categorical(y_val)
        
            # Train the model
            model.fit(X_train, y_train_categorical, epochs=5, batch_size=32, verbose=0)
        
            # Make predictions on the validation set
            y_val_pred = model.predict(X_val)
            y_val_pred_classes = np.argmax(y_val_pred, axis=1)
        
            # Calculate accuracy and store it in the list
            accuracy = accuracy_score(y_val, y_val_pred_classes)
            nn_scores.append(accuracy)

        
        
        print("Neural network Classifier Cross-Validation Accuracy: {:.2f}%".format(np.mean(nn_scores) * 100))

        #  Store and return results
        ml_performances['rf'] = np.mean(rf_scores) * 100
        ml_performances['knn'] = np.mean(knn_scores) * 100
        ml_performances['svm'] = np.mean(svm_scores) * 100
        ml_performances['nn'] = np.mean(nn_scores) * 100
        return ml_performances
        
    #  For evaluating on low dimnesional feature space instead of a high dimnesional features space
    def evaluate_low_dim_with_cross_validation(self, n_splits=10):
        # Initialize cross-validation
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        # Evaluate SVM Classifier with cross-validation
        svm_classifier = SVC(kernel='linear', C=1.0)
        svm_scores = cross_val_score(svm_classifier, self.low_dim_features, self.labels, cv=cv, scoring='accuracy')
        print("SVM Classifier Cross-Validation Accuracy: {:.2f}%".format(np.mean(svm_scores) * 100))

        # Evaluate k-NN Classifier with cross-validation
        knn_classifier = KNeighborsClassifier(n_neighbors=int(np.sqrt(len(self.X_train))))
        knn_scores = cross_val_score(knn_classifier, self.low_dim_features, self.labels, cv=cv, scoring='accuracy')
        print("k-NN Classifier Cross-Validation Accuracy: {:.2f}%".format(np.mean(knn_scores) * 100))

        # Evaluate Random Forest Classifier with cross-validation
        rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_scores = cross_val_score(rf_classifier, self.low_dim_features, self.labels, cv=cv, scoring='accuracy')
        print("Random Forest Classifier Cross-Validation Accuracy: {:.2f}%".format(np.mean(rf_scores) * 100))




