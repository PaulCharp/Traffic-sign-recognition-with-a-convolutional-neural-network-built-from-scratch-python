# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 13:58:07 2018

@author: paulc
Here are implementing the different structures tested for our issue. 
-A VGG structure
-A structure proposed by Yann LeCun and Pierre Sermanet in : http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf
"""

from ConvNet import *

###############################################VGG _Structure############################################################

"""Layer 1 (Convolutional): The output shape should be 32x32x32.

Activation. Your choice of activation function.

Layer 2 (Convolutional): The output shape should be 32x32x32.

Activation. Your choice of activation function.

Layer 3 (Pooling) The output shape should be 16x16x32.

Layer 4 (Convolutional): The output shape should be 16x16x64.

Activation. Your choice of activation function.

Layer 5 (Convolutional): The output shape should be 16x16x64.

Activation. Your choice of activation function.

Layer 6 (Pooling) The output shape should be 8x8x64.

Layer 7 (Convolutional): The output shape should be 8x8x128.

Activation. Your choice of activation function.

Layer 8 (Convolutional): The output shape should be 8x8x128.

Activation. Your choice of activation function.

Layer 9 (Pooling) The output shape should be 4x4x128.

Flattening: Flatten the output shape of the final pooling layer such that it's 1D instead of 3D.

Layer 10 (Fully Connected): This should have 128 outputs.

Activation. Your choice of activation function.

Layer 11 (Fully Connected): This should have 128 outputs.

Activation. Your choice of activation function.

Layer 12 (Fully Connected): This should have 43 outputs.
from https://github.com/mohamedameen93/German-Traffic-Sign-Classification-Using-TensorFlow"""

"""PROBLEME AVEC LA TAILLE DES FILTRES DES MAXPOLLING"""
    
def createVGGNetNetwork():
    layers = [ConvLayer((1,1,32,32),(5,5),32, zero_padding = 2),
            ConvLayer((32,1,32,32),(5,5),1,zero_padding=2),
            MaxPollingLayer((32,1,32,32),filter_shape=(2,2)),
            ConvLayer((32,1,16,16),(5,5),2,zero_padding = 2),
            ConvLayer((64,1,16,16),(5,5),1,zero_padding=2), # JESPERE QUE CETTE LIGNE EST BONNE
            MaxPollingLayer((64,1,16,16),filter_shape=(2,2)),
            ConvLayer((64,1,8,8),(5,5),2,zero_padding=2),
            ConvLayer((128,1,8,8),(5,5),1,zero_padding=2),
            MaxPollingLayer((128,1,8,8),filter_shape = (2,2)),
            FCLayer(2048,128,'sigmoid'),
            FCLayer(128,128,'sigmoid'),
            FCLayer(128,43,'sigmoid')]
    cnn = ConvNet(layers)
    return cnn
################################################################################################################

####################################### Yann LeCun Conv Network########################################################
""" 
To create a ConvNet structure like the one proposed in the paper : http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf
We create a ConvNet with the parameter sign_recognition = True
I've already implemented this convnet inside the convnet object because its structure was too difficult to implement aside
Note that ConvLayer.guess_sign_recognition and ConvLayer.train_sign_recognition are specially writen for Yann LeCun 's ConvNet.
For the VGGNet ConvLayer.guess and ConvLayer.train are enougth.
"""

def createYannLeCunNetwork():
    return ConvNet(sign_recognition=True)
    