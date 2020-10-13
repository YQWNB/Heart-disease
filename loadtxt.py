'''
Descripttion: 
version: 
Author: @Yeqiwei
Date: 2020-08-24 09:27:13
LastEditors: @Yeqiwei
LastEditTime: 2020-10-13 08:40:03
'''
import torch
import os
import nibabel as nib
import numpy as np
import cv2
from nibabel.viewers import OrthoSlicer3D as osd
from nibabel import nifti1
import matplotlib.pyplot as plt
from scipy.misc import imread, imresize
def load_data():
    root = './training'
    files = open(".\\data.txt",'w')
    data =[]
    label = 0
    dir_path = os.listdir(root)
    for d in dir_path:
        dp = os.path.join(root,d)
        if os.path.isdir(dp):
            file_path = os.listdir(dp)
            for f in file_path:
                path_n = ''.join([f,'_4d.nii.gz'])
                fp = os.path.join(dp,f,path_n)
                if len(data)==0:
                    strs = fp+' '
                else:
                    strs ='\n'+ fp+' '
                img = nib.load(fp).dataobj
                strs+=str(np.shape(img)[3])+' '+str(label)
                data.append(strs)
                # print(np.shape(img))
        label = label + 1
    files.writelines(data)
    files.close()
    
load_data()

