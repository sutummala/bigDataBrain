
# Code created by Sudhakar on March 2020
# Re-arranging the data at HCP YA that matches apparoximately with BIDS guidelines

import os
import numpy as np
# import gzip
from distutils.dir_util import copy_tree
import shutil
# import json

# from zipfile import ZipFile

data_dir = "/usr/users/tummala/HCP-YAr" # Path to the original subjects 
data_dir1 = "/usr/users/tummala/HCP-YA-Re" # Path to the rearranged subjects 

if not os.path.exists(data_dir1):
    os.makedirs(data_dir1)

subjects = os.listdir(data_dir) # Subjects the data folder

for subject in subjects:
    print('################ Re-arrangement has started for HCP YA', subject, '################')
    datapath = data_dir+'/'+subject+'/unprocessed/3T'
    strucImages = os.listdir(datapath)
    nFiles = np.size(strucImages)
    
    # Finding number/type of images
    if nFiles == 0:
        print('Folder is empty for Subject', subject, 'moving on to next subject')
        continue
    
    datapathnew = data_dir1+'/'+subject # Creating new directory
    
    if not os.path.exists(datapathnew):
        os.makedirs(datapathnew)
    
    datapathAnat = datapathnew+'/''anat' # Creating new directory for storing anat files
    
    if not os.path.exists(datapathAnat):
        os.makedirs(datapathAnat)
        
    secondT1 = False
    secondT2 = False
    
    for images in strucImages:
        if images.endswith('MPR2'):
            secondT1 = True
        elif images.endswith('SPC2'):
            secondT2 = True
        
    for images in strucImages:
        if images.endswith('.csv'):
            shutil.copy(datapath+'/'+images, datapathnew) # Copying files
            print(images, 'file re-arranged successfully')
        elif images.endswith('MPR1'):
            files = os.listdir(datapath+'/'+images)
            for image in files:
                if any([image.endswith('AFI.nii.gz'), image.endswith('32CH.nii.gz'), image.endswith('BC.nii.gz'), image.endswith('Magnitude.nii.gz'), image.endswith('Phase.nii.gz')]):
                    shutil.copy(datapath+'/'+images+'/'+image, datapathAnat)
                    print(image, 'moved to anat folder')
                elif image.endswith('MPR1.nii.gz'):
                    shutil.copy(datapath+'/'+images+'/'+image, datapathAnat)
                    if secondT1:
                        image1 = image.replace('T1w_MPR1', 'hrT1.A')
                        os.rename(datapathAnat+'/'+image, datapathAnat+'/'+image1)
                    else:
                        image1 = image.replace('T1w_MPR1', 'hrT1')
                        os.rename(datapathAnat+'/'+image, datapathAnat+'/'+image1)
                        print(image1, 'moved to anat folder')
        elif images.endswith('MPR2'):
            files = os.listdir(datapath+'/'+images)
            for image in files:
                if image.endswith('MPR2.nii.gz'):
                    shutil.copy(datapath+'/'+images+'/'+image, datapathAnat)
                    image1 =image.replace('T1w_MPR2', 'hrT1.B')
                    os.rename(datapathAnat+'/'+image, datapathAnat+'/'+image1)
                    print(image1, 'moved to anat folder')
        elif images.endswith('SPC1'):
            files = os.listdir(datapath+'/'+images)
            for image in files:
                if image.endswith('SPC1.nii.gz'):
                    shutil.copy(datapath+'/'+images+'/'+image, datapathAnat)
                    if secondT2:
                        image1 = image.replace('T2w_SPC1', 'hrT2.A')
                        os.rename(datapathAnat+'/'+image, datapathAnat+'/'+image1)
                    else:
                        image1 = image.replace('T2w_SPC1', 'hrT2')
                        os.rename(datapathAnat+'/'+image, datapathAnat+'/'+image1)
                        print(image1, 'moved to anat folder')
        elif images.endswith('SPC2'):
            files = os.listdir(datapath+'/'+images)
            for image in files:
                if image.endswith('SPC2.nii.gz'):
                    shutil.copy(datapath+'/'+images+'/'+image, datapathAnat)
                    image1 = image.replace('T2w_SPC2', 'hrT2.B')
                    os.rename(datapathAnat+'/'+image, datapathAnat+'/'+image1)
                    print(image1, 'moved to anat folder')             
           
    print('\n')
    
print('Re-arrangement of HCP-YA data finished for all subjects')
