# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 23:19:36 2018

@author: paulc
"""

from PIL import Image 
import os
import numpy as np
import skimage
from skimage import io, transform
import random

imsize = (32,32)


def augmentation_1(pict, classe): #return a list Flip in the same class
    invariant_horizontal_rotation = np.array([11, 12, 13, 15, 17, 18, 22, 26, 30, 35])
    invariant_vertical_rotation = np.array([1, 5, 12, 15, 17])
    invariant_both = np.array([32, 40])
    listpict = [pict]
    if classe in invariant_horizontal_rotation :
        listpict.append(np.fliplr(pict))
    if classe in invariant_vertical_rotation :
        listpict.append(pict[::-1])
    if classe in invariant_both :
        listpict.append(pict[::-1])
    return listpict
        
def histoEqualization(pict): #Pict is an array
    #Here we'll proceed to a histogram equalization 
    L = 256 #Niveau de gris possibles
    hist = np.zeros(L)
    for i in range(len(pict)):
        for j in range(len(pict[0])):
            hist[int(pict[i,j])] +=1
    histcum = 0
    transformation = np.zeros(L)
    for i in range(len(hist)):
        histcum = histcum + hist[i]
        transformation[i] = (L-1)/(np.size(pict)) * histcum #From wiki
    for i in range(len(pict)):
        for j in range(len(pict[0])):
            pict[i,j] = transformation[int(pict[i,j])]
    return pict

def augmentation_2(classe): #Flip and change classe
    flip_antisymetrie = np.array([[19, 20], 
        [33, 34], 
        [36, 37], 
        [38, 39],
        [20, 19], 
        [34, 33], 
        [37, 36], 
        [39, 38]])
    if classe in flip_antisymetrie[: 1]:
        #On cherche la classe correspondant a la modification
        i = flip_antisymetrie[np.where(flip_antisymetrie[:,0])[0][0], 1]
        return i
    else : 
        return None
    
def augmentation_3(processedSetPath): #On va effectuer de faible rotation sur les images pour continuer a augmenter le set
    for folder in os.listdir(processedSetPath):
        for impng in os.listdir(processedSetPath + "\\" + folder):
            for i in range(1) : #On créer 1 nouvelles images a chaque fois, (choix arbitraire)
                pict = skimage.io.imread(processedSetPath + '\\' + folder + '\\' + impng)
                pict = skimage.transform.rotate(pict, random.random()*40 -20, mode = 'edge') #le mode egde permet en realite une deformation
                pict = skimage.img_as_ubyte(pict) #We want pixels value to lie between 0 and 255 which isn't the case when applying rotate
                io.imsave(processedSetPath + '\\' + folder + '\\' + impng[:-4] + '_rotation_' + str(i) + '.png', pict)
                
    
    
def preprocess_pictures(pict, classe):#return a list of picture that needs to be register
    #Turn into greyscale
    pict =  0.299 * pict[:, :, 0] + 0.587 * pict[:, :, 1] + 0.114 * pict[:, :, 2]
    #Hist equalization
    pict = histoEqualization(pict)
    #Now well increase out dataset
    listpict = augmentation_1(pict, classe)
    return listpict

def preprocess_dataset(): #We'll modify our pictures and register them. We want to register them because we'll have to run
                                #multiple test and we don't want to lose time to modify pictures each time.
    print('[+] Preprocessing_dataset')
    rawSetPath = os.getcwd() + '\\GTSRB\\Final_Training\\Images'
    #We create a new folder for our preprocessed_dataset
    os.makedirs(".\Preprocessed_dataset_GTSRB")
    processedSetPath = ".\Preprocessed_dataset_GTSRB"
    print('    [+] First step, grayscale conversion and histogram equalization, and flip from oneself class')
    for direct in os.listdir(rawSetPath):
        os.makedirs(processedSetPath + "\\" + str(int(direct))) #Pour chaque classe on crée un fichier
        for imppm in os.listdir(rawSetPath + "\\" + direct): #On prend chaque image dans chaque dossier
            if imppm[-3] + imppm[-2] + imppm[-1] == 'ppm' :
                i = 0
                pict = Image.open(rawSetPath +'\\'+ direct + '\\' +imppm)
                pict = pict.resize(imsize) #Toutes les images doivent avoir les mêmes dimensions
                pict = np.array(pict)
                for im in preprocess_pictures(pict, int(direct)):
                    Image.fromarray(im).convert('L').save(processedSetPath + "\\" + str(int(direct)) + "\\" + imppm[:-4] + '_' + str(i) + '.png')
                    i = i + 1
    #Ensuite, certains panneaux subissant une rotation change de sens, on le prend en compte
    print('    [+]Second step, flipping to another class')
    for folder in os.listdir(processedSetPath):
        res = augmentation_2(int(folder))
        if res != None : 
            #Pour chaque image dans, on l'inverse et on l'enregistre dans le dossier correspondant
            for impng in os.listdir(processedSetPath + "\\" + folder):
                #On fait la symetrie horizontale
                im = Image.open(processedSetPath + '\\' + folder + '\\' + impng)
                im = np.array(im)
                im = np.fliplr(im)
                Image.fromarray(im).convert('L').save(processedSetPath + "\\" + str(res) + "\\" + impng[:-4] + '_' + 'horizontaly_flipped' + '.png')
    print('    [+]Third step, apply small rotations')
    augmentation_3(processedSetPath)
    print('[+] preprocess_dataset end')
    
def preprocess_testingSet():
    os.makedirs(".\Preprocessed_testingSet_GTSRB")
    print('[+] Preprocessing_testingSet')
    rawSetPath = '.\GTSRB\\Testing_Set'
    processedSetPath = ".\Preprocessed_testingSet_GTSRB"
    print('    [+] Grayscale conversion and histogram equalization')
    for direct in os.listdir(rawSetPath):
        os.makedirs(processedSetPath + "\\" + str(int(direct))) #Pour chaque classe on crée un fichier
        for imppm in os.listdir(rawSetPath + "\\" + direct): #On prend chaque image dans chaque dossier
            if imppm[-3] + imppm[-2] + imppm[-1] == 'ppm' :
                pict = Image.open(rawSetPath +'\\'+ direct + '\\' +imppm)
                pict = pict.resize(imsize) #Toutes les images doivent avoir les mêmes dimensions
                pict = np.array(pict)
                pict =  0.299 * pict[:, :, 0] + 0.587 * pict[:, :, 1] + 0.114 * pict[:, :, 2]
                pict = histoEqualization(pict)
                Image.fromarray(pict).convert('L').save(processedSetPath + "\\" + str(int(direct)) + "\\" + imppm[:-4] + '_' + '.png')
    print('[+] End')
            
if __name__ == '__main__' :
    preprocess_dataset()
    preprocess_testingSet()
        
"""
Ideas :
    -change image encoding to float to get better precision
    -change image resolution
    -quatificate wether the algo is sure of its anwser or not.
"""