
from modules.resnet.resnet import ResNetBuilder
import numpy as np
import warnings

from keras.layers import merge, Input
from keras.layers import Dense, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.layers import BatchNormalization
from keras.models import Model
from keras.preprocessing import image
import keras.backend as K
from keras.utils.layer_utils import convert_all_kernels_in_model
from keras.utils.data_utils import get_file
from modules.imagenet_utils import decode_predictions, preprocess_input

from modules.resnet_encoder import ResNet50


import time

t = time.time()

#model = ResNetBuilder.build_resnet_18((3, 224, 224), 1000)
model = ResNet50(include_top=True, weights='imagenet')

#model.compile(loss="categorical_crossentropy", optimizer="sgd")
#model.summary()

img_path = 'elephant.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
print('Input image shape:', x.shape)

preds = model.predict(x)
print('Predicted:', decode_predictions(preds))

#print 'Time: %f ' % (time.time() - t)
