
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from s_dbw import S_Dbw
from sklearn.neighbors import LocalOutlierFactor
import hdbscan
import numpy as np


"""
    A class for calculating and reporting various metrics for a given feature space, including silhouette score,
    Davies Bouldin Index, S_Dbw, Calinski Harabasz Index, and the number of outliers.

    Attributes:
        feature_space (numpy.ndarray): The feature space data for which metrics are calculated.
        data_loader (DataLoader): A data loader object providing true class labels.
        decimal_places (int): The number of max decimal places for metric values to be printd

    Methods:
        __init__(self, feature_space, data_loader, decimal_places=3):
            Initializes a Metrics instance with the provided feature space and data loader

        calculate(self):
            Calculates various

    Usage:
    # Example Usage
    data_loader = DataLoader(...)  # Replace with your data loading logic
    feature_space = ...  # Replace with your feature space
    metrics = Metrics(feature_space, data_loader, decimal_places=3)
    metrics.calculate()
    print(metrics.metrics)
    """

class Metrics:

    def __init__(self, feature_space, data_loader, decimal_places = 3):
        self.feature_space = feature_space
        self.labels = data_loader.labels
        self.decimal_places = decimal_places
        self.metrics = None
        self.calculate()
        
    

    # Calculate the metrics on the high dimensional featue space
    def calculate(self):
        features = self.feature_space
        labels = self.labels
    
        # # of outliers
        LOF = LocalOutlierFactor(n_neighbors = int(np.sqrt(len(labels))))
        
        outliers = LOF.fit_predict(features)
        num_outliers = len(np.where(outliers==-1)[0])
        
        
        # Intrinsic metrics
        silhouette = silhouette_score(features, labels)
        davies_bouldin_index = davies_bouldin_score(features, labels)
        calinski_harabasz_index = calinski_harabasz_score(features, labels)
        s_dbw = S_Dbw(features, labels, centers_id=None, method='Tong', alg_noise='bind', centr='mean', nearest_centr=True, metric='euclidean')


        # store the metrics in a dictionary for future access
        metrics = {'silhouette' : silhouette, 'DBI':davies_bouldin_index, 'CH':calinski_harabasz_index, 'sdbw':s_dbw, 'outliers':num_outliers}
        
        print('# of outliers : {:.{dp}f}'.format(metrics['outliers'], dp=self.decimal_places))
        print('Silhouette score : {:.{dp}f}'.format(metrics['silhouette'], dp=self.decimal_places))
        print('Davies Bouldin Index *: {:.{dp}f}'.format(metrics['DBI'], dp=self.decimal_places))
        print('Calinski Harabasz Index: {:.{dp}f}'.format(metrics['CH'], dp=self.decimal_places))
        print('S_Dbw *: {:.{dp}f}'.format(metrics['sdbw'], dp=self.decimal_places))

        self.metrics = metrics
        return metrics
