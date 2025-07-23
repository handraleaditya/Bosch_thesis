<!---

	Copyright (c) 2009, 2018 Robert Bosch GmbH and its subsidiaries.
	This program and the accompanying materials are made available under
	the terms of the Bosch Internal Open Source License v4
	which accompanies this distribution, and is available at
	http://bios.intranet.bosch.com/bioslv4.txt

-->

# Evaluating high dimensional feature spaces: Effects and Metrics <!-- omit in toc -->



[![License: BIOSL v4](http://bios.intranet.bosch.com/bioslv4-badge.svg)](#license)
This repo contains the code for the experiments ran during the masterthesis focusing on the effects happening in the a high dimensional feature spaces extracted by an image feature extractor and using metrics to evaluate them.


## Table of Contents  <!-- omit in toc -->

- [Getting Started ](#getting-started-)
- [Content description ](#content-description-)
	- [DatasetWiz](#datasetwiz)
	- [Notebooks](#notebooks)
	- [References](#references)
	- [Miscellanious](#miscellanious)
- [Datasets info](#datasets-info)
- [Results TL;DR](#results-tldr)
- [About ](#about-)
	- [Contributors ](#contributors-)
	- [License ](#license-)

## Getting Started <a name="getting-started"></a>

To run this repo and install all the libraries used, create a conda environment using the requirements.txt file

``$ conda create --name NameOfYourEnv --file requirements.txt``

## Content description <a name="content_description"></a>

### DatasetWiz
A helper library named DatasetWiz was created with reusable functions for performing tasks such as loading a dataset, creating mislabels, class imbalances, feature extraction, dimensionality reduction, calculating metrics, feature space evaluation and data visualization.

``loader.py`` : Contains the DataLoader class which is used to load dataset images as a keras/tensorflow dataset. Has functionalities such as store labels, rescale RGB values, create class imbalances and mislabelings. It also has helper functions to visualize samples of the dataset and view class distributions.

``model.py`` : Has functions to build image feature extraction models like ResNet50, VGG19 and EfficientNetB0. Models are pretrained on ImageNet and specific layers are used to extract features. Note: models will give different shape feature vectors ex ResNet has 2048 dimensions for one image, while EfficientNet has 1280 dimensions.

``visualizer.py`` : This class is used to perform dimensionality reduction on the high dimensional features extracted from the image feature extractor and plot them in a 2D plot for better visualization. Includes methods like TSNE, UMAP and MDS. It also has functions to perform clustering using KMeans and HDBSCAN. Additionally it also has a function to calculate supervised or extrinsic metrics on the results of the clustering.  The class has functionality to plot interactive 2D plots using the ``bokeh`` library to see the images by hovering over a data point in the scatterplots, enabling a better understanding of the feature extractor and the effects.


``metrics.py`` : Contains the functions for applying chosen metrics to any feature space, high or low dimensional. It uses Silhouette score, Calinski Harabasz score, S_Dbw, Davies Bouldin index and Local Outlier Factor. All of which are intrinsic measures. The extrinsic measure implementations are contained in ``visualizer.py``.

``evaluation.py`` : This module contains the code for creating and evaluating ML models to test the correlation of metrics with the actual feature space. It has models such as KNN, Random Forest, SVM and Neural Networks. It also has option to perform KFold cross validation to minimize the randomness of evaluations.

### Notebooks
The repo also contains various notebooks that show how to use the library on the datasets for performing analysis. NOTE: Please consider the results and figures from the PPT as the final ones instead of the outputs in the notebooks as some parameters have been altered for the sake of brevity and it is impossible to record ALL of the conducted experiments. Although this does NOT mean that results in the PPT cannot be recreated. 


### References
This folder contains all the references exported as a .bib file from the Mendeley reference manager, which were used during the literature review. There are over 120+ paper and resources with around 50 of them being relevant and 25-30 of which are very relevant to this topic.

### Miscellanious
This folder contains unorganized files such as images, excel file of old results, old notebooks etc which we felt the need to store for future reference, just in case, rather than be deleted.



## Datasets info

- PEG, Mattendichtung and ASQMM : Bosch property
- Bottle, Screw, Metal Nut, Cable, Capsule : Taken from MVTec AD available [here](https://www.mvtec.com/company/research/datasets/mvtec-ad)
- Cats Vs. Dogs : Filtered version taken from Google ML Crash course/Kaggle available [here](https://colab.research.google.com/github/google/eng-edu/blob/master/ml/pc/exercises/image_classification_part1.ipynb#scrollTo=UY6KJV6z6l7_)
- MNIST : MNIST as JPG took the trainingSample from [here](https://www.kaggle.com/datasets/scolianni/mnistasjpg?select=trainingSet )


## Results TL;DR

According to the experiments conducted,
- Silhouette score, Davies Bouldin and Calinski Harabasz are intrinsic metrics that are correlated with ML model accuracies, and react intuitively and accordingly to effects such as mislabels, outliers, blur and class imbalance and hence can be used to evaluate high dimensional feature spaces, while S_Dbw is not recommended
- For extrinsic metrics ARI, NMI and V Measure should be used over Fowlkes Mallows
- These metrics are heuristic devices and hence should be used in a comparative fashion. One value of a metric in one dataset does not mean we get the same ML model accuracy in another dataset with same value of the metric
- Metrics are difficult to interpret, especially for non domain experts
- A single metric does not capture one single effect nor is most sensitive to a single effect, rather they capture overall 'goodness' of the clustering.

## About <a name="about"></a>

### Contributors <a name="contributors"></a>

Aditya Handrale  
[FIXED-TERM Handrale Aditya Vasant (CR/APT4)](fixed-term.adityavasant.handrale@de.bosch.com)


### License <a name="license"></a>

[![License: BIOSL v4](http://bios.intranet.bosch.com/bioslv4-badge.svg)](#license)

> Copyright (c) 2009, 2018 Robert Bosch GmbH and its subsidiaries.
> This program and the accompanying materials are made available under
> the terms of the Bosch Internal Open Source License v4
> which accompanies this distribution, and is available at
> http://bios.intranet.bosch.com/bioslv4.txt

<!---

	Copyright (c) 2009, 2018 Robert Bosch GmbH and its subsidiaries.
	This program and the accompanying materials are made available under
	the terms of the Bosch Internal Open Source License v4
	which accompanies this distribution, and is available at
	http://bios.intranet.bosch.com/bioslv4.txt

-->
