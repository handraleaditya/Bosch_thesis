import numpy as np
import pathlib, os
import matplotlib.pyplot as plt

# for dimensionality reduction
from sklearn.manifold import TSNE
from sklearn.manifold import MDS
import umap
#clustering techniques
from sklearn.cluster import SpectralClustering, KMeans
import hdbscan

# Extrinsic metrics 
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, fowlkes_mallows_score,homogeneity_completeness_v_measure

# intrinsic metrics
from s_dbw import S_Dbw
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score



# Boken imports used for plotting with images on hover
import bokeh
from bokeh.plotting import figure, show, output_notebook, output_file
from bokeh.models import HoverTool, ColumnDataSource
from bokeh.models import HoverTool, ColumnDataSource, ImageURL
from bokeh.transform import linear_cmap
from bokeh.palettes import Category10
from bokeh.models import  LinearColorMapper
from bokeh.plotting import figure, show
from bokeh.models import HoverTool, ColumnDataSource, LegendItem, Legend
from bokeh.palettes import Category10




"""
    A class for visualizing features extracted from a pre-trained model using different dimensionality reduction techniques
    and clustering algorithms.

    Attributes:
        model (CustomModel): An instance of the CustomModel class containing feature data.
        reduction_method (str): The dimensionality reduction method ('tsne', 'mds', or 'umap').

    Methods:
        __init__(self, model, reduction_method='tsne'):
            Initializes a FeatureSpaceVisualizer instance with the provided CustomModel and reduction method.

        initialize_visualizer(self):
            Initializes the low-dimensional feature space using the specified reduction method.

        visualize(self):
            Visualizes the low-dimensional TSNE feature space with class labels.

        spectral_clustering(self):
            Applies spectral clustering to the feature space and visualizes the clusters.

        supervised_metrics(self, true_labels, cluster_labels, title):
            Calculates and prints various supervised metrics for clustering.

        kmeans_clustering(self):
            Applies K-Means clustering to the feature space and visualizes the clusters.

        hdbscan_clustering(self):
            Applies HDBSCAN clustering to the feature space and visualizes the clusters.

        visualize_bokeh(self, save=False):
            Visualizes the feature space using Bokeh with tooltips and legends.

    Usage:
    # Example Usage
    custom_model = CustomModel(model_name='resnet50', data_loader=data_loader, weights='imagenet', include_top=False)
    visualizer = FeatureSpaceVisualizer(custom_model, reduction_method='tsne')
    visualizer.visualize()
    visualizer.spectral_clustering()
    kmeans_metrics = visualizer.kmeans_clustering()
    bokeh_plot = visualizer.visualize_bokeh()
    ```
    """


class FeatureSpaceVisualizer:

    def __init__(self, model, reduction_method = 'tsne'):

        # store the original raw feature space (non reduced)
        self.original_feature_space = model.features
        # store the images data 
        self.data = model.data
        # store the name of the model used, useful for putting titles in plots
        self.model_name = model.model_name
        self.class_names = model.class_names
        self.reduction_method = reduction_method
        # variable to store reduced feature space
        self.low_dim_feature_space = None
        self.extrinsic_metrics = {}
        
        self.initialize_visualizer()
        self.data_loader = model.data_loader
        
    # Intitialize the respective dimensionality reduction technique and fit the original feature space
    def initialize_visualizer(self):
        if self.reduction_method == 'tsne':
            tsne = TSNE(n_components = 2, random_state = 42)
            self.low_dim_feature_space = tsne.fit_transform(self.original_feature_space)
            print(f'TSNE reduced {self.model_name} feature space shape :', self.low_dim_feature_space.shape)
            
        if self.reduction_method == 'mds':
            mds = MDS(n_components=2)
            self.low_dim_feature_space = mds.fit_transform(self.original_feature_space)

        if self.reduction_method == 'umap':
            self.low_dim_feature_space = umap.UMAP().fit_transform(self.original_feature_space)
            print(f'UMAP reduced {self.model_name} feature space shape :', self.low_dim_feature_space.shape)

    
    # Visualize the reduced tsne feature space
    def visualize(self):
       
        labels = self.data_loader.labels
        labels = np.array(labels)
        # Create a scatter plot
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(self.low_dim_feature_space[:, 0], self.low_dim_feature_space[:, 1], c=labels, alpha=0.3, cmap='viridis')       
        class_names = self.class_names
        # Create a dictionary to map class labels to unique colors
        unique_labels = np.unique(labels)
        colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))
        class_color_mapping = {label: color for label, color in zip(unique_labels, colors)}       
        # Add legend with custom class names and colors
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=class_names[label], markerfacecolor=color, markersize=10) for label, color in class_color_mapping.items()]
        plt.legend(handles=legend_elements, loc='upper right')   
        plt.title(f't-SNE Visualization of {self.model_name} features')
        plt.xlabel(f'{self.model_name} t-SNE Dimension 1')
        plt.ylabel(f'{self.model_name} t-SNE Dimension 2')
        plt.savefig(f'{self.model_name}.png', dpi=300)
        plt.show()
    
    # perform spectral clustering 
    def spectral_clustering(self):
        n_clusters = len(self.class_names)
        spectral_clustering = SpectralClustering(n_clusters=n_clusters, random_state=42)
        spectral_cluster_labels = spectral_clustering.fit_predict(self.low_dim_feature_space)
        plt.figure(figsize=(8, 6))
        plt.scatter(self.low_dim_feature_space[:, 0], self.low_dim_feature_space[:, 1], c=spectral_cluster_labels, cmap='viridis', s=50)
        plt.title('Spectral Clustering')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.show()

    # Calculate extrinsic metrics on the clusterings performed by KMeans and HDBSCAN
    # Also calculates intrinsic metrics on post clustering results
    # Finally store them in an array and return it 
    def supervised_metrics(self,true_labels, cluster_labels, title):
        ari = adjusted_rand_score(true_labels, cluster_labels)
        nmi = normalized_mutual_info_score(true_labels, cluster_labels)
        fm = fowlkes_mallows_score(true_labels, cluster_labels)
        vm = homogeneity_completeness_v_measure(true_labels, cluster_labels)

        # Intrinsic metrics post clustering
        silhouette = silhouette_score(self.original_feature_space, cluster_labels)
        davies_bouldin_index = davies_bouldin_score(self.original_feature_space, cluster_labels)
        calinski_harabasz_index = calinski_harabasz_score(self.original_feature_space, cluster_labels)
        s_dbw = S_Dbw(self.original_feature_space, cluster_labels, centers_id=None, method='Tong', alg_noise='bind', centr='mean', nearest_centr=True, metric='euclidean')

        
        print(f"{title} Adjusted Rand Index (ARI -1-1): {ari:.2f}")
        print(f"{title} Normalized Mutual Information (NMI 0-1): {nmi:.2f}")
        print(f"{title} Fowlkes-Mallows Score (0-1): {fm:.2f}")
        print(f"{title} Vmeasure (0-1): {vm[2]:.2f}")

        
        print(f"{title} post clustering Silhouette: {silhouette:.2f}")
        print(f"{title} post clustering calinski_harabasz_index: {calinski_harabasz_index:.2f}")
        print(f"{title} post clustering davies_bouldin_index: {davies_bouldin_index:.2f}")
        print(f"{title} post clustering s_dbw: {s_dbw:.2f}")


        extrinsic_metrics = {}
        extrinsic_metrics[f'{title}ari'] = ari
        extrinsic_metrics[f'{title}nmi'] = nmi
        extrinsic_metrics[f'{title}fm'] = fm
        extrinsic_metrics[f'{title}vm'] = vm[2]

        extrinsic_metrics[f'{title}silhouette'] = silhouette
        extrinsic_metrics[f'{title}calinski_harabasz_index'] = calinski_harabasz_index
        extrinsic_metrics[f'{title}s_dbw'] = s_dbw
        extrinsic_metrics[f'{title}davies_bouldin_index'] = davies_bouldin_index

        return extrinsic_metrics
        

    # Perform KMeans clustering on the high dimensional feature space
    def kmeans_clustering(self):
        n_clusters = len(self.class_names)
        kmeans_clustering = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans_clustering.fit_predict(self.original_feature_space)
        kmeans_metrics = self.supervised_metrics(self.data_loader.labels, kmeans_clustering.labels_, 'KMeans')
        plt.figure(figsize=(8, 6))
        plt.scatter(self.low_dim_feature_space[:, 0], self.low_dim_feature_space[:, 1],alpha=0.3, c=kmeans_clustering.labels_, cmap='viridis', s=50)
        plt.title('KMeans Clustering')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.show()
        return kmeans_metrics



    # Perform HDBSCAN clustering on high dimensional feature space
    def hdbscan_clustering(self):

        HDB =  hdbscan.HDBSCAN(min_cluster_size=5)
        HDB.fit(self.original_feature_space)
        extrinsic_metrics = self.supervised_metrics(self.data_loader.labels, HDB.labels_, 'HDBSCAN')
        plt.figure(figsize=(8, 6))

        # Scatter plot each data point with a color corresponding to its cluster
        plt.scatter(self.low_dim_feature_space[:, 0], self.low_dim_feature_space[:, 1], c=HDB.labels_, cmap='viridis', s=50)
        
        plt.title('HDBSCAN Clustering')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.show()

        return extrinsic_metrics
         

    # Function used to create and view a bokeh plot
    # In the bokeh plot, we add tooltips that display images, 
    # so that we can hover over a scatterpoint and see the image associated it for investigation  
    def visualize_bokeh(self, save=False):
    
        title = f"TSNE visualization for {self.model_name}" 
        data = self.low_dim_feature_space
        data_loader = self.data_loader
        
        #Extract class labels:
        class_labels = data_loader.labels
        class_names = data_loader.class_labels()
    
        # Create image paths for bokeh
        image_paths = [str(pathlib.Path.cwd() / image_path) for image_path in data_loader.data.file_paths]
        image_paths_with_file_scheme = ['file:///' + path for path in image_paths]

        #Store file names
        file_names = [str(os.path.split(path)[1]) for path in image_paths]

    
        # Define a color palette based on the number of unique class labels
        unique_labels = list(set(class_labels))
        num_classes = len(unique_labels)
        if num_classes<3:
            colors = ['#1f77b4', '#ff7f0e']
        else:
            colors = Category10[num_classes]  # we can choose a different palette if needed
    
        # Map class labels to colors
        color_mapping = {label: colors[i] for i, label in enumerate(unique_labels)}
        point_colors = [color_mapping[label] for label in class_labels]
    
        # Map numerical labels to class names
        class_names_mapping = {i: class_name for i, class_name in enumerate(class_names)}
        label_names = [class_names_mapping[label] for label in class_labels]
    
        # Create a Bokeh ColumnDataSource with image data
        source = ColumnDataSource(data=dict(
            x=data[:, 0],
            y=data[:, 1],
            imgs=image_paths_with_file_scheme,  # Store image filenames for tooltips
            labels=label_names,
            fnames = file_names,
            colors=point_colors,  # Store point colors
        ))
    
        # Create a new Bokeh figure for the scatter plot
        p = figure(title=title, toolbar_location='right', tools="pan,box_zoom,reset,wheel_zoom")

         # <div>
         #    <span style="font-size: 10px; font-weight: bold;">File:@imgs</span>
         # </div>
        
        # Define the tooltip template and how it should look on hover
        tooltip_template = """
            <div>
                <div>
                    <span style="font-size: 14px; font-weight: bold;">Label: </span>
                    <span style="font-size: 14px;">@labels</span>
                </div>
               
                 <div>
            <span style="font-size: 10px; font-weight: bold;">File:@imgs</span>
         </div>
                <div>
                    <img src="@imgs" alt="" width="200" height="200">
                </div>
            </div>
            
        """

    
        # Add tooltips using the template
        hover = HoverTool(tooltips=tooltip_template)
    
        # Add the hover tool to the plot
        p.add_tools(hover)
    
        # Create a legend and legend items
        legend_items = []
        for class_label, class_color in color_mapping.items():
            class_indices = [i for i, label in enumerate(class_labels) if label == class_label]
            class_source = ColumnDataSource(data=dict(
                x=[data[i, 0] for i in class_indices],
                y=[data[i, 1] for i in class_indices],
                imgs=[image_paths_with_file_scheme[i] for i in class_indices],
                labels=[label_names[i] for i in class_indices],
                colors=[class_color] * len(class_indices)
            ))
            scatter = p.scatter('x', 'y', source=class_source, size=8, color='colors', alpha=0.5, legend_label=class_names_mapping[class_label])
            legend_items.append(LegendItem(label=class_names_mapping[class_label], renderers=[scatter]))
    
        legend = Legend(items=legend_items)
        p.add_layout(legend)

      
        return p
