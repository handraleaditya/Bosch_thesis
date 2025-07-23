
# Resnet50 and its preprocessing functions
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess_input
from tensorflow.keras.applications import ResNet50

# VGG 19 and its preprocessing functions
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input as vgg_preprocess_input

# Efficientnet and its preprocessing functions
from tensorflow.keras.applications.efficientnet import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess_input

# Used to build custom neural network to test feature space quality
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model 


"""
    A custom image classification model for extracting features using various pre-trained architectures.

    Attributes:
        model_name (str): The name of the pre-trained model architecture ('resnet50', 'vgg19', 'efficientnet', etc.).
        data_loader (DataLoader): A data loader object providing input data for the model.
        weights (str): Optional, the weight initialization to use ('imagenet' or custom weights).
        include_top (bool): Optional, whether to include the fully connected layers at the top of the model.

    Methods:
        __init__(self, model_name, data_loader, weights="imagenet", include_top=False):
            Initializes a CustomModel instance with the specified parameters.

        initialize_model(self):
            Initializes the specified pre-trained model with custom parameters.

        extract_features(self):
            Extracts features from the initialized model using the provided data_loader.

    Usage:
    # Example Usage
    data_loader = DataLoader(...)  
    custom_model = CustomModel(model_name='resnet50', data_loader=data_loader, weights='imagenet', include_top=False)
    custom_model.extract_features()
    print(custom_model.features)
    """

class CustomModel:

    # Constructor for CustomModel, initializes model according to parameters passed
    def __init__(self, model_name, data_loader, weights ="imagenet", include_top = False):
        self.model_name = model_name
        self.weights = weights
        self.include_top = include_top
        self.data = data_loader.data
        self.class_names = data_loader.class_names
        self.model = self.initialize_model()
        self.features = None
        self.data_loader = data_loader
    
        
    
    # Initialize a model with custom parameters like model type, weights etc.
    def initialize_model(self):

        # ResNet50 model
        if self.model_name == 'resnet50':
            base_model = ResNet50(weights = self.weights, include_top = self.include_top)
            output = GlobalAveragePooling2D()(base_model.output)
            model = Model(inputs= base_model.input, outputs=output)
            self.data = self.data.map(lambda x,y : (resnet_preprocess_input(x),y))
            return model
        
        # VGG 16 model
        if self.model_name == 'vgg19':
            base_model = VGG19(weights=self.weights, include_top=self.include_top)
            intermediate_layer_name = 'block5_pool' # name of the intermediate layer we want to use
            output = GlobalAveragePooling2D()(base_model.output)
            # if we want to use some other layer instead
            # model = Model(inputs=base_model.input, outputs=base_model.get_layer(intermediate_layer_name).output)
            model = Model(inputs=base_model.input, outputs=output)
            # apply preprocessing steps specif to the model
            self.data = self.data.map(lambda x,y : (vgg_preprocess_input(x),y))
            return model

        if self.model_name == 'efficientnet':
            base_model = EfficientNetB0(weights= self.weights, include_top=self.include_top)
            output = GlobalAveragePooling2D()(base_model.output)
            model = Model(inputs = base_model.input, outputs = output)
            self.data = self.data.map(lambda x,y : (efficientnet_preprocess_input(x),y))
            return model
        
    # For extracting features from custom model, common for all models    
    def extract_features(self):
        features = self.model.predict(self.data)
        self.features = features
        print(f"{self.model_name} extracted feature space size :", self.features.shape) 
         
