# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 11:42:31 2018

@author: paulc

Here is our main code to solve our issue. Here we train our network on our dataset and we test it. Youpi :) 
"""
import sys
import numpy as np
from ConvNet import *
from PIL import Image
import random
import os
import time
import pickle #To save our convNet
from Preprocess_dataset import preprocess_dataset
import matplotlib.pyplot as plt
from ConvNet_Structures import *

#################################Preprocessing###################################################################################
#We preprocess our dataset only if we haven't done it yet. Indeed we made sure to register the modification we made to our dataset
#so we only need to preprocess once and then we can run as many test as we may want to.

#preprocess_dataset()

###############################Hyperparameters###################################################################################
epoch = 17200

###############################Open the saving files#############################################################################
#Here we open the file we use to save our network. We have a .txt file to register our weigth matrices (just to have a look)
#and a pickle file to dump our network our to take a previous one we would have already trained. 

filePickle = 'sauvegarde__ConvNet.txt'
answer_file = 'sauvegarde_answer.txt'

################################# Defining our network ###########################################################################
    #We have different options. We can either create a VGG or a Yann LeCunn's Network. (Fonctions defined in ConvNet_Structures.py)
    #We can also download a network we already created and that we register in a file using Pickle 

cnn = createVGGNetNetwork() #The one for the VGG we are testing here
#cnn = createYannLeCunNetwork() #The one for the Yann LeCun's Network
#sauvegarde = open(filePickle,'rb')
#cnn = pickle.load(sauvegarde)

#################################Training the convnet#############################################################################
def train(cnn):
    print('[+] Start training')
    processedSetPath = ".\Preprocessed_dataset_GTSRB"
    #On crée un grand tableau avec le nom de toutes les images du set d'entrainement disponibles
    trainSet = {}
    for file in os.listdir(processedSetPath) :
        l = os.listdir(processedSetPath + '\\' + file)
        trainSet[file] = []
        for im in l:
            trainSet[file].append(im)
    sauvegarde_cnn = open(filePickle,'w')
    t = time.time()
    file = -1
    error = []
    epoche = []
    for i in range(0,epoch):
        file = (file+1)%43
        target = np.zeros(43)
        target[file] = 1
        target = np.matrix(target).transpose() #Target must be a column matrix
        im = random.choice(trainSet[str(file)])
        im = Image.open(processedSetPath + '\\' + str(file) + '\\' + im)
        im = np.array(im)
        im = np.array([[im]]) #By doing so, we add the grayscale channel dimension and the depth dimension
        im = im / 255 #Pixel uint8 normalization
        err = cnn.train(im, target)
        error.append(err)
        epoche.append(i)
        if int((i+1)/100) == (1+i)/100 :
            ti = (time.time() - t)/60 #temps d'execution en min
            print('    [+] Iteration number = ' + str(i + 1) + '   ' + str((i+1)/epoch*100) + '  % training completed')
            print('        [+] Execution time : ' + str(int(ti/60)) + 'h   ' + str(ti - int(ti/60)*60) + 'min')
            sauvegarde_cnn = open(filePickle,'wb') #write in binary mode
            pickle.dump(cnn, sauvegarde_cnn)
            sauvegarde_cnn.close()
            plt.plot(i,error)
    print('[+] End of training')

train(cnn)

###############################################Process a full test of the network#################################################
def test_network(cnn):
    print('[+] Start testing network')
    path = ".\Preprocessed_testingSet_GTSRB"
    testSet = []
    for file in os.listdir(path):
        l = os.listdir(path + '\\' + file)
        for im in l :
            testSet.append([im,file])
    testSet = np.array(testSet)
    testSize = len(testSet)
    print('    [+] TestingSet contains ' + str(testSize) + 'elements')
    t = time.time()
    rigth_answers = 0
    i = 0
    for [im,file] in testSet:
        pict = Image.open(path + '\\' + file + '\\' + im)
        pict = np.array(pict)
        pict = np.array([[pict]])#By doing so, we add the grayscale channel dimension and the depth dimension
        pict = pict/255 #Pixel between 0 and 1
        cnn_answer = cnn.guess(pict)
        cnn_answer = cnn_answer.argmax() + 1 #On prend la sortie la plus grande comme reponse
        #Probleme, on ne prends pas en compte la certitude de l'algo, softmax ?
        if int((i+1)/100) == (1+i)/100 :
            ti = (time.time() - t)/60 #temps d'execution en min
            print('    [+] Iteration number = ' + str(i + 1) + '   ' + str((i+1)/testSize*100) + '  % testing completed')
            print('        [+] Execution time : ' + str(int(ti/60)) + 'h   ' + str(ti - int(ti/60)*60) + 'min')
        if cnn_answer == int(file) :
            rigth_answers += 1
        i = i +1
    print('    [+] Proportion of rigth answers = ' + str(rigth_answers/testSize *100))
    f = open(answer_file, 'a')
    f.write('[+] Proportion of rigth answers = ' + str(rigth_answers/len(testSet) *100))
    print('[+] End')
    
##########################################Process a shorter test of the network####################################################
    #We can either process a full and very long test of the network or a shorter one (this one :)
def reduced_test(cnn):
    print('[+] Start testing reduced network')
    path = ".\Preprocessed_testingSet_GTSRB"
    testSet = []
    for file in os.listdir(path):
        l = os.listdir(path + '\\' + file)
        for im in l :
            testSet.append([im,file])
    testSet = np.array(testSet)
    testSize = len(testSet)
    print('    [+] TestingSet contains ' + str(testSize) + 'elements')
    t = time.time()
    rigth_answers = 0
    i = 0
    nbTest = 1000
    for i in range(nbTest) :
        [im,file] = random.choice(testSet)
        pict = Image.open(path + '\\' + file + '\\' + im)
        pict = np.array(pict)
        pict = np.array([[pict]])#By doing so, we add the grayscale channel dimension and the depth dimension
        pict = pict/255 #Pixel between 0 and 1
        cnn_answer = cnn.guess(pict)
        cnn_answer = cnn_answer.argmax() + 1 #On prend la sortie la plus grande comme reponse
        #Probleme, on ne prends pas en compte la certitude de l'algo, softmax ?
        if int((i+1)/100) == (1+i)/100 :
            ti = (time.time() - t)/60 #temps d'execution en min
            print('    [+] Iteration number = ' + str(i + 1) + '   ' + str((i+1)/nbTest*100) + '  % testing completed')
            print('        [+] Execution time : ' + str(int(ti/60)) + 'h   ' + str(ti - int(ti/60)*60) + 'min')
        print(cnn_answer, int(file))
        if cnn_answer == int(file) :
            rigth_answers += 1
        i = i +1
    print('    [+] Proportion of rigth answers = ' + str(rigth_answers/nbTest *100))
    f = open(answer_file, 'a')
    f.write('[+] Proportion of rigth answers = ' + str(rigth_answers/nbTest *100))
    print('[+] End')

reduced_test(cnn)
#test_network(cnn)