# -*- coding: utf-8 -*-
"""
Created on Sun Aug 26 11:48:09 2018

@author: paulc
"""

#Notations
# dEdI -> derivative of the cost function with respect of the inputs
# dEdH -> derivative of the cost function with respect of the ouputs
# dEdb -> derivative of the cost function with respect of the biais
# dEdw -> derivative of the cost function with respect of the weights
# dHdU -> derivative of the outputs with respect of (the outputs just before being passed in ReLU)
#Note that the terms "inputs" and "outputs" are relative to a layer. For example : 
#for a hidden layer output mean the result it process, not the result of the whole network

import numpy as np
import sys
from PIL import Image
import random
LR = 0.01 #This is arbitrary chose. I've no idea what to take so that it the network learn better.

def randomMatrix(rows,cols):
    m = []
    for i in range(rows):
        r = []
        for j in range(cols):
            r.append(random.gauss(0,1)) #On veut les valeurs entre -1 et 1
        m.append(r)
    m = np.matrix(m)
    return m

def pad_inputs(inputs,zero_padding): #inputs dimensions d*c*h*w
    d,c,h,w = np.shape(inputs)
    new_inputs = np.zeros((d,c,h+2*zero_padding, w+2*zero_padding))
    for depth in range(d):
        for channel in range(c):
            new_inputs[depth, channel] = np.pad(inputs[depth, channel], ((zero_padding,zero_padding),(zero_padding, zero_padding)), 'constant')
    return new_inputs

def sigmoid(m): # With numpy, exp can be applied to matrix, This is awesome !!!
    return 1 / (1 + np.exp(-m))

class FCLayer(): #Note that ReLU layer is included in FCLayer
    
    def __init__(self,nb_inputs,nb_nodes,a_fun, is_inputLayer = False, lr = LR) :
        self.nb_inputs = nb_inputs
        self.nb_nodes = nb_nodes
        self.a_fun = a_fun
        if is_inputLayer == False :
            self.w = randomMatrix(nb_nodes,nb_inputs) #np.matrix(np.random.normal(size = (nb_nodes,nb_inputs)))
            self.b = randomMatrix(nb_nodes, 1) #np.matrix(np.random.normal(size = (nb_nodes,1)))
        else :
            self.w = None
            self.b = None
        self.lr = lr
        self.out = np.zeros(nb_nodes)
        
    def a_function (self,m):  #Activation function
        if self.a_fun == 'sigmoid' :
            return sigmoid(m)
        if self.a_fun == 'id' :
            return m
        if self.a_fun == 'pixel_normalisation':
            return (m/255)
        sys.exit(str(m), 'isn\'t known as an activation function')
        
    def delta_b_w(self, dEdH, outputs, inputs):
        if self.a_fun == 'sigmoid':
            self.dEdb =  - np.multiply(dEdH, np.multiply(outputs, (1-outputs)))
            self.b -= self.lr * self.dEdb
            self.w -= self.lr * self.dEdb * (inputs.transpose())
        else :
            sys.exit('[-] Why calculating an error on an non defined a_function ?')
    
    def guess(self, inputs):
        if not isinstance(inputs, np.matrix):
            inputs = np.matrix(inputs.flatten()).transpose()
        self.out = np.array(sigmoid(self.w * inputs + self.b))
        return self.out
    
    def train(self, dEdH, outputs, inputs):
        if not isinstance(inputs, np.matrix):
            inputs = np.matrix(inputs.flatten()).transpose()
        self.delta_b_w(dEdH, outputs, inputs)
        
    def backPropagateGradient(self, dEdH, inputs = None) : #Return the gradient of the previous Layer : dEdI
#        if inputs != None : #If this layer is used in a convolutional layer architecture, we parse the inputs because it's easier to code this this way
        previousLayerShape = np.shape(inputs)
        if isinstance(previousLayerShape, int) : #If inputs is a vector, the previous layer is thus a FC
            return self.w.transpose() * self.dEdb
        elif len(previousLayerShape) > 1: #Which mean that the previous layer (k-1) is either a ConvLayer or a MaxPollingLayer
            return np.array((self.w.transpose() * self.dEdb)).reshape(previousLayerShape)
#    else :
#           return self.w.transpose() * self.dEdb  #return dEdI
        
class ConvLayer: #Note that ReLU layer is included in ConvLayer
    
    def __init__(self, input_shape, filter_shape, nb_filters, zero_padding = 0, stride = 1, lr = LR):
        #input_shape[0] -> Depth (1 si c'est une image)
        #input_shape[1] -> Channels (ex : number of pixel, rgb -> 3)
        #input_shape[2] -> Heigth
        #input_shape[3] -> Width
        
        self.input_shape = input_shape 
        self.in_depth, self.in_channel, self.in_heigth, self.in_width = self.input_shape
        self.filter_shape = filter_shape
        self.zero_padding = zero_padding
        self.nb_filters = nb_filters
        self.stride = stride
        self.lr = lr
        self.out_channel_out = 1 #After a convolution only one channel
        
        #Initialize weigths randomly
        self.w = np.random.rand(self.nb_filters, self.in_channel, self.filter_shape[0], self.filter_shape[1])
        #self.w[0] -> nb filter (depth)
        #self.w[1] -> channels of inputs
        #self.w[2] -> heigth of inputs
        #self.w[3] -> width of inputs
        self.b = np.random.rand(self.nb_filters, 1)
        
        #Formula from "Convolutional Neural network with Python" from Franck Milstein
        self.out_row_dim = int((self.in_heigth - self.filter_shape[0] + 2*self.zero_padding) / self.stride + 1) #Dimension of the row output
        self.out_cols_dim = int((self.in_width - self.filter_shape[1] + 2*self.zero_padding) / self.stride + 1) #Dimension of the cols output
        self.out = np.zeros((self.nb_filters *self.in_depth, 1, self.out_row_dim, self.out_cols_dim)) #We only have one channel after convolution, hence the "1"
        
        
    def process_convolution(self, inputs):
        #Prepare inputs with zero_padding
        self.inputsPadded = pad_inputs(inputs, self.zero_padding)
        outDepth = 0 #Depth of the output
        #Now we'll process for everyfilters
        #For each filter we calculate for all the depth inputs
        for filt in range(0, self.nb_filters):
            for d in range(len(self.inputsPadded)): #inputs depth
                cols = 0
                row = 0
                while row + self.filter_shape[0] <= len(self.inputsPadded[0,0]):
                    while cols + self.filter_shape[1] <= len(self.inputsPadded[0,0,0]):
                        #The weighted sum of the inputs parsed in the sigmoid
                        self.out[outDepth, 0, row, cols] = sigmoid(np.sum(np.multiply(self.inputsPadded[d, :, row:self.filter_shape[0] + row, cols:self.filter_shape[1] + cols], self.w[filt])) + self.b[filt])
                        cols = cols + self.stride #The filter is shifted with a path of stride
                    cols = 0
                    row = row + self.stride
                outDepth += 1

    def guess(self, inputs): #Easier to code the all process after
        self.process_convolution(inputs)
        return self.out
   
    def train(self, dEdH, outputs, inputs): #We won't use input here but self.inputPadded
        #dEdH - > #Gradient of the cost with respect of the output, size depth*channels*heigth*width
        dw = np.zeros(np.shape(self.w))
        db = np.zeros(np.shape(self.b))
        outDepth = 0
        d, c, h, w = np.shape(self.inputsPadded)
        self.dHdU = outputs * (1-outputs)
        #From https://becominghuman.ai/back-propagation-in-convolutional-neural-networks-intuition-and-code-714ef1c38199
        for filt in range(self.nb_filters):
            for depth in range(d):
                row = 0
                cols = 0
                while row + self.filter_shape[0] <= h:
                    while cols + self.filter_shape[1] <=  w:
                        dw[filt] += np.multiply(self.inputsPadded[depth, :, row:row+self.filter_shape[0], cols:cols+self.filter_shape[1]], np.multiply(self.dHdU[outDepth, :, row, cols], dEdH[outDepth, :, row, cols]))
                        db[filt] += np.multiply(self.dHdU[outDepth, :, row, cols], dEdH[outDepth, :, row, cols])
                        cols = cols + self.stride
                    row = row + self.stride
                    cols = 0
                outDepth += 1
        self.w -= self.lr * dw
        self.b -= self.lr * db
  
    def backPropagateGradient(self, dEdH, inputs = None): #dEdH gradient at the next layer, dE/dH (at k+1)
        #Gradient of the cost with respect of the inputs
        dEdI = np.zeros(self.input_shape)
        dEdIPadded = np.zeros(np.shape(self.inputsPadded))
        d,c,h,w = np.shape(dEdIPadded)
        outDepth = 0
        for filt in range(self.nb_filters):
            for depth in range(d):
                row = 0
                cols = 0 
                while row + self.filter_shape[0] <= h:
                    while cols + self.filter_shape[1] <= w :
                        dEdIPadded[depth, :, row:row+self.filter_shape[0], cols:cols+self.filter_shape[1]] += np.multiply(self.w[filt], np.multiply(self.dHdU[outDepth, :, row, cols], dEdH[outDepth, :, row, cols])) #+= car on additionne les erreurs de chaque filtre
                        cols += self.stride
                    row += self.stride
                    cols = 0
                outDepth += 1
        #Here we'll delete padding
        for depth in range(d):
            for cha in range(c):
                dEdI[depth,cha] = dEdIPadded[depth,cha, self.zero_padding:self.zero_padding+self.in_heigth, self.zero_padding: self.zero_padding+self.in_width]
        return dEdI

    
class MaxPollingLayer():
    
    def __init__(self, input_shape, filter_shape = (2,2), stride = None):
        if stride == None :
            stride = filter_shape
        #input_shape[0] -> depth
        #input_shape[1] -> Channel
        #input_shape[2] -> Heigth
        #input_shape[3] -> Width
        self.input_shape= input_shape
        self.in_depth, self.in_channel, self.in_heigth, self.in_width = self.input_shape
        self.filter_shape = filter_shape
        self.stride = stride 
        self.nb_filters = self.in_depth #Useful variable to make loops regardless of the type of Layer
        self.out_channel_out = self.in_channel #Useful variable to make loops regardless of the type
        #A COMPRENDRE !!
        #Formula from "Convolutional Neural network with Python" from Franck Milstein
        #self.out_row_dim = int((self.in_heigth - self.filter_shape[0] ) / self.stride[0] + 1) #Dimension of the row output
        #self.out_cols_dim = int((self.in_width - self.filter_shape[1] ) / self.stride[1] + 1) #Dimension of the cols output
        self.out_row_dim = int(self.in_heigth/self.filter_shape[0])
        self.out_cols_dim = int(self.in_width/self.filter_shape[1])
        self.out = np.zeros((self.in_depth, 1, self.out_row_dim, self.out_cols_dim)) #We only have one channel after convolution, hence the "1"
        self.max_index = np.zeros(input_shape) #Very inportant when propagating the error back
        
    def process_pooling(self, inputs):
        #for every map feature
        
        for mapp in range(0, self.in_depth):
            row = 0
            cols = 0
            while row + self.filter_shape[0] <= len(inputs[0,0]) : #Hauteur
                while cols + self.filter_shape[1] <= len(inputs[0,0,0]): #Largeur
                    maxx = np.amax(inputs[mapp, 0, row:self.filter_shape[0] + row, cols:self.filter_shape[1] + cols])
                    self.out[mapp, 0, int(row/self.filter_shape[0]), int(cols/self.filter_shape[1])] = maxx
                    we = np.where(inputs[mapp, 0, row:self.filter_shape[0] + row, cols:self.filter_shape[1] + cols] == maxx)
                    maxindex = (mapp, 0, we[0][0] + row, we[1][0] + cols) #not sure, we take the first one to avoid parralellism
                    self.max_index[maxindex] = 1
                    cols = cols + self.stride[1]
                row = row + self.stride[0]
                cols = 0
         
    
    def guess(self, inputs): 
        self.process_pooling(inputs)
        return self.out
    
    def train(self, dEdH, outputs, inputs): #More simple to iterate loops in ConvNet Layers
        pass
    
    def backPropagateGradient(self, dEdH, inputs = None): #Return gradient for the previous layer, inputs only if using FC out of the sign recognition issue
        #The back propagated gradient has to be for an convLayer
        #Formulas and demonstrations from https://medium.com/the-bioinformatics-press/only-numpy-understanding-back-propagation-for-max-pooling-layer-in-multi-layer-cnn-with-example-f7be891ee4b4
        dEdH = np.repeat(dEdH, self.filter_shape[0],axis = 2)
        dEdH = np.repeat(dEdH, self.filter_shape[1], axis = 3)
        return dEdH * self.max_index #Element wise multiplication
    

#Si CA MARCHE PAS REPRENDRE L'ANALYSE DU CODE A PARTIR DE LA
                
class ConvNet():
    
    def __init__(self, layers = [], sign_recognition = False):
        self.nb_layers = len(layers)
        self.layers = []
        #################################################################################################################################
        #        An example of how to config layers
        #        layers = ['FCLayer_first',nb_inputs, nb_nodes,a_fun, is_inputLayer = False, lr = 0.3],
        #                   ['FCLayer_notFirst', nb_nodes,a_fun, is_inputLayer = False, lr = 0.3]]
        #        layers = [
        #                ['ConvLayer', input_shape, filter_shape, nb_filters, zero_padding = 0, stride = 1],
        #                ['MaxPollingLayer', filter_shape = (2,2)],
        #                '[FCLayer_notfirst', nb_nodes,a_fun, is_inputLayer = False, lr = 0.3]]
        # Make sure there is no more layer than an FC the overall output after a FC
        #################################################################################################################################
        """
        This is a draft I'd like to keep. Please do not pay attention to this part of the code.
        
        
        if sign_recognition == False :
            for lay in layers:
                if lay[0] == 'ConvLayer' :
                    self.layers.append(ConvLayer(lay[1], lay[2], lay[3], lay[4], lay[5]))
                elif lay[0] == 'MaxPollingLayer' :
                    input_shape = (self.layers[-1].nb_filters, 1, self.layers[-1].out_row_dim, self.layers[-1].out_cols_dim) #Indeed, a maxpollinglayer is only after a convlayer
                    self.layers.append(MaxPollingLayer(input_shape, lay[1]))
                elif lay[0] == 'FCLayer_notFirst':
                    if isinstance(self.layers[-1], FCLayer): #If the previous layer is a FCLayer
                        nb_inputs = self.layers[-1].nb_nodes
                    elif isinstance(self.layers[-1], ConvLayer) or isinstance(self.layers[-1], MaxPollingLayer):
                        nb_inputs = self.layers[-1].out_cols_dim * self.layers[-1].out_row_dim * self.layers[-1].nb_filters * self.layers[-1].out_channel_out
                    else :
                        sys.exit("[-] FCLayer has to go after a FCLayer, a ConvLayer or a MaxPollingLayer only.")
                    self.layers.append(FCLayer(nb_inputs, lay[1], lay[2]))
                elif lay[0] == 'FCLayer_first':
                    self.layers.append(FCLayer(lay[1], lay[2], lay[3], lay[4], lay[5], is_inputLayer = True))
                else :
                    sys.exit("[-] Syntax error in layers" + str(lay[0]) + "isn't recognize as a layer type")"""
        if sign_recognition == False:
            self.layers = layers
        else :
            #from https://github.com/navoshta/traffic-signs
            self.layers = [ConvLayer((1,1,32,32), (5,5), 32, zero_padding = 2),
                           MaxPollingLayer((32, 1, 32, 32)),
                           MaxPollingLayer((32,1,16,16), filter_shape = (4,4)),
                           ConvLayer((32,1,16,16), (5,5), 2, zero_padding = 2),
                           MaxPollingLayer((64,1,16,16)),
                           MaxPollingLayer((64,1,8,8)),
                           ConvLayer((64,1,8,8), (5,5), 2, zero_padding = 2),
                           MaxPollingLayer((128,1,8,8)),
                           FCLayer(3584, 1024, 'sigmoid'),
                           FCLayer(1024, 43, 'sigmoid')]
            self.nb_layers = len(self.layers)
               
    
    def guess(self, inputs): # Inputs has to be an 4D array depth*channels*heigth*width (for an image, depth = 1)
            avancement = inputs
            for lay in self.layers:
                avancement = lay.guess(avancement)
            return avancement 
    
    def train(self,inputs, targets):
        avancement = []
        avancement.append(inputs)
        i = 1
        for lay in self.layers:
            avancement.append(lay.guess(avancement[i-1]))
            i += 1
        errors = targets - avancement[-1]
        for i in range(len(avancement)-1,0,-1) :#En decroissant, 0 exclu
            self.layers[i-1].train(errors, avancement[i], avancement[i-1])
            if i != 1 : #Which mean, it's not the first layer
                errors = self.layers[i-1].backPropagateGradient(errors, avancement[i-1])
        return targets - avancement[-1]
                
    def guess_sign_recognition(self, inputs, fromTrain = None):
        self.dictLay = {'firstConv' : self.layers[0],
                        'firstPolling' : self.layers[1],
                        'firstPollingToFC' : self.layers[2],
                        'secondConv' : self.layers[3],
                        'secondPolling' : self.layers[4],
                        'secondPollingToFC': self.layers[5],
                        'thirdConv' : self.layers[6],
                        'thirdPolling' : self.layers[7],
                        'firstFC' : self.layers[8],
                        'secondFC' : self.layers[9]}
        #Avancement 0:
        avancement = {}
        avancement["inputs"] = inputs
        #Avancement 1 :
        avancement['firstConv'] = self.dictLay['firstConv'].guess(inputs) #Process first convolution
        #Avancement 2:
        avancement['firstPolling'] = self.dictLay['firstPolling'].guess(avancement["firstConv"])#Process first convolution
        #Avancement 3:
        avancement['firstPollingToFC'] = self.dictLay['firstPollingToFC'].guess(avancement['firstPolling']) #Process firstPollingToFc
        #Avancement 4:
        avancement['secondConv'] = self.dictLay['secondConv'].guess(avancement['firstPolling'])#Process second convolution
        #Avancement 5:
        avancement['secondPolling'] = self.dictLay['secondPolling'].guess(avancement['secondConv']) #Process second max polling
        #Avancement 6:
        avancement['secondPollingToFC'] = self.dictLay['secondPollingToFC'].guess(avancement['secondPolling']) #Process second max polling to FC
        #Avancement 7:
        avancement['thirdConv'] = self.dictLay['thirdConv'].guess(avancement['secondPolling']) #Process third convolution
        #Avancement 8:
        avancement['thirdPolling'] = self.dictLay['thirdPolling'].guess(avancement['thirdConv']) #Process third max polling
        #Avancement 9:
        entryFC = np.concatenate((avancement['firstPollingToFC'].flatten(), avancement['secondPollingToFC'].flatten(), avancement['thirdPolling'].flatten()))
        entryFC = np.matrix(entryFC).transpose()
        avancement['firstFC'] = self.dictLay['firstFC'].guess(entryFC) #Process firs FC
        #Avancement 10:
        avancement['secondFC'] = self.dictLay['secondFC'].guess(avancement['firstFC'])#Process final FC
        if fromTrain == None :
            return avancement['secondFC']
        else :
            return avancement, entryFC
        
    def train_sign_recognition(self, inputs, targets):
        avancement, entryFC = self.guess_sign_recognition(inputs, fromTrain = True)
        dEdH = targets - avancement['secondFC']
        err = 0
        for i in dEdH :
            err += 1/2*(i)**2
        #Second FC training
        self.dictLay['secondFC'].train(dEdH, avancement['secondFC'], avancement['firstFC'])
        dEdH = self.dictLay['secondFC'].backPropagateGradient(dEdH)
        #First FC training
        self.dictLay['firstFC'].train(dEdH, avancement['firstFC'], entryFC)
        dEdH_FC = np.array(self.dictLay['firstFC'].backPropagateGradient(dEdH).flatten())[0] #transform the column vector error into a array
        #Third polling backpropagate error
        dEdH = dEdH_FC[1536:].reshape(np.shape(self.dictLay['thirdPolling'].out))
        dEdH = self.dictLay['thirdPolling'].backPropagateGradient(dEdH)
        #Third conv training
        self.dictLay['thirdConv'].train(dEdH, avancement['thirdConv'], avancement['secondPolling'])
        dEdH = self.dictLay['thirdConv'].backPropagateGradient(dEdH)
        #Second Polling and Second Polling to FC backPropagateGradient
        err = dEdH_FC[512:1536].reshape(np.shape(self.dictLay['secondPollingToFC'].out))
        dEdH += self.dictLay['secondPollingToFC'].backPropagateGradient(err) #Not sure but I'll try
        dEdH = self.dictLay['secondPolling'].backPropagateGradient(dEdH)
        #Second convolution training
        self.dictLay['secondConv'].train(dEdH, avancement['secondConv'], avancement['firstPolling'])
        dEdH = self.dictLay['secondConv'].backPropagateGradient(dEdH)
        #First Polling back propagate error
        err = dEdH_FC[:512].reshape(np.shape(self.dictLay['firstPollingToFC'].out))
        dEdH += self.dictLay['firstPollingToFC'].backPropagateGradient(err)
        dEdH = self.dictLay['firstPolling'].backPropagateGradient(dEdH) 
        #First Conv training
        self.dictLay['firstConv'].train(dEdH, avancement['firstConv'], inputs)
        return err
            
                