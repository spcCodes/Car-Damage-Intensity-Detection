# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 02:53:58 2018

@author: SUMAN
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 01:53:41 2018

@author: SUMAN
"""

def LoadData():
    
    from PIL import Image
    import numpy as np
    import os.path
    import scipy
    from scipy import ndimage
    
    length = 128 # pixels in length
    width = 128 # pixels in width
    
    '''MINOR'''
    directory_name = 'C:/Users/SUMAN/Desktop/artivatic/task_vision/data/training/minor'
    
    imgs = []
    for filename in os.listdir(directory_name):
        if filename.endswith(".JPEG") or filename.endswith(".jpeg"):
            img = Image.open(os.path.join(directory_name,filename)).convert('RGB')
            img1 = np.array(img)
            imgResize = scipy.misc.imresize(img1, size=(64, 64))
            imgs.append(imgResize)
    
    MinImg = np.array(imgs)
    
    np.save('C:/Users/SUMAN/Desktop/artivatic/task_vision/data/MinImg.npy' , MinImg)
    del imgs
    
    '''MODERATE'''
    directory_name = 'C:/Users/SUMAN/Desktop/artivatic/task_vision/data/training/moderate'
    
    imgs = []
    for filename in os.listdir(directory_name):
        if filename.endswith(".JPEG") or filename.endswith(".jpeg"):
            img = Image.open(os.path.join(directory_name,filename)).convert('RGB')
            img1 = np.array(img)
            imgResize = scipy.misc.imresize(img1, size=(64, 64))
            imgs.append(imgResize)
    
    ModImg = np.array(imgs)
    
    np.save('C:/Users/SUMAN/Desktop/artivatic/task_vision/data/ModImg.npy' , ModImg)
    del imgs
    
    '''severe'''
    directory_name = 'C:/Users/SUMAN/Desktop/artivatic/task_vision/data/training/severe'
    
    imgs = []
    for filename in os.listdir(directory_name):
        if filename.endswith(".JPEG") or filename.endswith(".jpeg"):
            img = Image.open(os.path.join(directory_name,filename)).convert('RGB')
            img1 = np.array(img)
            imgResize = scipy.misc.imresize(img1, size=(64, 64))
            imgs.append(imgResize)
    
    SevImg = np.array(imgs)
    
    np.save('C:/Users/SUMAN/Desktop/artivatic/task_vision/data/SevImg.npy' , SevImg)
    del imgs
    
    
    '''VALIDATION SETS'''
    '''MINOR'''
    
    directory_name = 'C:/Users/SUMAN/Desktop/artivatic/task_vision/data/test/01-minor'
    
    imgs = []
    for filename in os.listdir(directory_name):
        if filename.endswith(".JPEG") or filename.endswith(".jpeg"):
            img = Image.open(os.path.join(directory_name,filename)).convert('RGB')
            img1 = np.array(img)
            imgResize = scipy.misc.imresize(img1, size=(64, 64))
            imgs.append(imgResize)
    
    ValMinImg = np.array(imgs)
    
    np.save('C:/Users/SUMAN/Desktop/artivatic/task_vision/data/ValMinImg.npy' , ValMinImg)
    del imgs
    
    
    '''MODERATE'''
    
    directory_name = 'C:/Users/SUMAN/Desktop/artivatic/task_vision/data/test/02-moderate'
    
    imgs = []
    for filename in os.listdir(directory_name):
        if filename.endswith(".JPEG") or filename.endswith(".jpeg"):
            img = Image.open(os.path.join(directory_name,filename)).convert('RGB')
            img1 = np.array(img)
            imgResize = scipy.misc.imresize(img1, size=(64, 64))
            imgs.append(imgResize)
    
    ValModImg = np.array(imgs)
    
    np.save('C:/Users/SUMAN/Desktop/artivatic/task_vision/data/ValModImg.npy' , ValModImg)
    del imgs
    
    '''SEVERE'''
    
    directory_name = 'C:/Users/SUMAN/Desktop/artivatic/task_vision/data/test/03-severe'
    
    imgs = []
    for filename in os.listdir(directory_name):
        if filename.endswith(".JPEG") or filename.endswith(".jpeg"):
            img = Image.open(os.path.join(directory_name,filename)).convert('RGB')
            img1 = np.array(img)
            imgResize = scipy.misc.imresize(img1, size=(64, 64))
            imgs.append(imgResize)
    
    ValSevImg = np.array(imgs)
    
    np.save('C:/Users/SUMAN/Desktop/artivatic/task_vision/data/ValSevImg.npy' , ValSevImg)
    del imgs
    
    MinImg = np.load('C:/Users/SUMAN/Desktop/artivatic/task_vision/data/MinImg.npy')
    ModImg = np.load('C:/Users/SUMAN/Desktop/artivatic/task_vision/data/ModImg.npy')
    SevImg = np.load('C:/Users/SUMAN/Desktop/artivatic/task_vision/data/SevImg.npy')
    
    
    ValMinImg = np.load('C:/Users/SUMAN/Desktop/artivatic/task_vision/data/ValMinImg.npy')
    ValModImg = np.load('C:/Users/SUMAN/Desktop/artivatic/task_vision/data/ValModImg.npy')
    ValSevImg = np.load('C:/Users/SUMAN/Desktop/artivatic/task_vision/data/ValSevImg.npy')
    
    MinLabels = np.zeros(MinImg.shape[0])
    ModLabels = np.ones(ModImg.shape[0])
    SevLabels = 2*np.ones(SevImg.shape[0])
    
    ValMinLabels = np.zeros(ValMinImg.shape[0])
    ValModLabels = np.ones(ValModImg.shape[0])
    ValSevLabels = 2*np.ones(ValSevImg.shape[0])
    
    TrainData = np.concatenate((MinImg, ModImg , SevImg), axis = 0 )
    TestData = np.concatenate((ValMinImg, ValModImg , ValSevImg), axis = 0 )
    TrainLabels = np.concatenate((MinLabels,ModLabels , SevLabels), axis = 0 ).astype('int64')
    TestLabels = np.concatenate((ValMinLabels,ValModLabels , ValSevLabels), axis = 0 ).astype('int64')
    
    TrainLabels = TrainLabels.reshape((1 , TrainLabels.shape[0]))
    TestLabels = TestLabels.reshape((1 , TestLabels.shape[0]))
    
    return TrainData , TestData , TrainLabels , TestLabels


TrainData , TestData , TrainLabels , TestLabels = LoadData()