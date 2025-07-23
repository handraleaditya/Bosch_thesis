import pathlib # to handle paths to dataset
import matplotlib.pyplot as plt # for plotting
import numpy as np # for using helper functions 
import tensorflow as tf # machine learning library
import sys, os # for hiding prints


from tensorflow.keras.preprocessing.image import ImageDataGenerator # rescaling the images
from tensorflow.keras.utils import image_dataset_from_directory # loading the images



'''
Helper class used for loading a dataset from a filepath, store labels and has functions to add mislabels 
and data imbalances

Parameters

    dataset_path (str): The path to the dataset directory.
    image_size (tuple, optional): The target image size for resizing (default: (224, 224)).
    batch_size (int, optional): The batch size for data loading (default: 32).

Attributes

    dataset_path (Path): Path to the dataset.
    batch_size (int): Batch size for data loading.
    image_size (tuple): Target image size.
    datagen (ImageDataGenerator): Image data generator for preprocessing.
    data (tf.data.Dataset): The dataset loaded from the directory.
    shuffled_data (tf.data.Dataset): A shuffled version of the dataset.
    labels (numpy.ndarray): The labels corresponding to the dataset.
    class_names (list): List of class names.

'''
class DataLoader:

    def __init__(self, dataset_path, image_size = (224,224), batch_size=32):
        
        # Path to the dataset
        self.dataset_path = pathlib.Path(dataset_path)
        self.batch_size = batch_size
        self.image_size = image_size

        
        
        self.datagen = ImageDataGenerator(rescale = 1./255,) # rescale pixel values

        # load the images
        self.data = image_dataset_from_directory(self.dataset_path,
                                                seed = 42,
                                                image_size = self.image_size,
                                                labels='inferred',
                                                shuffle=False,
                                                batch_size=self.batch_size,)
        
        # load data in a shuffled manner for preview 
        with HiddenPrints(): # internal prints are hidden
            self.shuffled_data = image_dataset_from_directory(self.dataset_path,
                                                seed = 42,
                                                image_size = self.image_size,
                                                labels='inferred',
                                                shuffle=True,
                                                batch_size=self.batch_size,)
            
        # store the labels and class_names
        self.labels = self.labels()
        self.class_names = self.data.class_names
        
    # helper function to that returns class names    
    def class_labels(self):
        return list(self.data.class_names)
    
    # helper function to that returns labels as np array    
    def labels(self):
        labels = []
        for images, label_batch in self.data:
            labels.extend(label_batch.numpy())
        labels = np.array(labels)
        return labels

    # if mislabels are intentionally created later, this function retores them
    def restore_original_labels(self):
        # Restore the original labels to their initial order
        original_order = np.argsort(self.labels)
        self.labels = self.labels[original_order]
        
    
    #  function used to create mislabelings in the dataset,  shuffles a percentage of labels as desired
    def mislabel_data(self, mislabel_percentage):
        # Create a copy of the original labels
        shuffled_labels = self.labels.copy()
    
        # Calculate the number of data points to mislabel
        num_mislabels = int(len(shuffled_labels) * mislabel_percentage)
    
        # Randomly select data points to mislabel
        mislabel_indices = np.random.choice(
            len(shuffled_labels), num_mislabels, replace=False
        )
    
        # Create new mislabeled labels (by shuffling a copy of the selected labels)
        selected_labels_copy = shuffled_labels[mislabel_indices].copy()
        np.random.shuffle(selected_labels_copy)
        shuffled_labels[mislabel_indices] = selected_labels_copy
    
        # Return the shuffled labels
        self.labels= shuffled_labels


        
    # function used to create class imbalance intentionally, takes the percentage of imbalance to create 
    # and the index of the class to be used, ex: class index 0 referes to NOK images in PEG 
    def create_class_imbalance(self, imbalance_percentage= 0.9, class_index = 0):

        # Convert dataset to numpy arraays
        dataset = self.data.unbatch()    
        features = []
        labels = []
        class_names = self.class_names
        for feature, label in dataset:
            features.append(feature)
            labels.append(label)
        features = np.array(features)
        labels = np.array(labels)
    
        class_mask = (labels == class_index)
    
        # Calculate the number of samples to keep in the specified class
        num_samples_to_keep = int(imbalance_percentage * np.sum(class_mask))
        print('num samples to keep', num_samples_to_keep)
        # Randomly shuffle the indices of the specified class
        indices_to_keep = np.where(class_mask)[0]
        np.random.shuffle(indices_to_keep)
    
        # Select the specified percentage of samples to keep
        indices_to_keep = indices_to_keep[:num_samples_to_keep]

        # Filter out samples from other classes
        other_class_mask = (labels != class_index)
        other_features = features[other_class_mask]
        other_labels = labels[other_class_mask]

        # Concatenate selected samples with samples from other classes
        new_features = np.concatenate((features[indices_to_keep], other_features))
        new_labels = np.concatenate((labels[indices_to_keep], other_labels))

        print('len of new labels ', len(new_labels), np.unique(new_labels))
    
        # Create a new dataset from the selected samples
        new_dataset = tf.data.Dataset.from_tensor_slices((new_features, new_labels))
        new_dataset = new_dataset.batch(32)
        self.data = new_dataset
        self.labels = new_labels
        self.class_names = self.class_names

        # clear memory because sometimes OOM error for huge datasets(ex : 40k+ highres images)
        del features, labels, new_features, new_labels, other_features
    

    
    # Function to visualize some sample images loaded into the dataset
    def visualize(self, num_samples = 5):
        plt.figure(figsize = (10,8))
        for images,labels in self.shuffled_data.take(1):
            for i in range(num_samples):
                plt.subplot(1, num_samples, i + 1)
                plt.imshow(images[i]/255.0)
                plt.title(f'{self.class_labels()[labels[i]]}')

        plt.tight_layout()
        plt.show()
        
    # visualize some sample images in a grid form
    def visualize_grid(self, num_samples=5):
        num_rows = (num_samples + 4) // 5  # Calculate the number of rows needed
    
        plt.figure(figsize=(10, 8))
    
        for images, labels in self.shuffled_data.take(1):
            for row in range(num_rows):
                for i in range(5):
                    index = row * 5 + i
                    if index >= len(images):
                        break  # No more images to display in this row
                    plt.subplot(num_rows, 5, row * 5 + i + 1)
                    plt.imshow(images[index] / 255.0)
                    plt.title(f'{self.class_labels()[labels[index]]}')
    
        plt.tight_layout()
        plt.show()


    # function used to visualize the class distribution in the dataset in a bar chart form
    def visualize_class_distribution(self):
            plt.figure(figsize=(10, 6))
    
            # Extract labels and class names
            labels = self.labels
            class_names = self.class_names
    
            # Count occurrences of each class label
            class_counts = {class_name: np.sum(labels == idx) for idx, class_name in enumerate(class_names)}
    
            # Generate a unique color for each class
            colors = plt.cm.get_cmap('tab20', len(class_names))
    
            # Create a bar graph with different colors for each class
            plt.bar(class_counts.keys(), class_counts.values(), color=[colors(i) for i in range(len(class_names))])
            plt.xlabel('Class Label')
            plt.ylabel('Count')
            plt.title('Class Distribution')
            
            # Add a legend for class names
            handles = [plt.Rectangle((0,0),1,1, color=colors(i), ec="k", label=class_name) for i, class_name in enumerate(class_names)]
            plt.legend(handles=handles, title='Class Names', loc='upper right')
    
            plt.show()


# Another helper class used to hide internal prints of any function
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout