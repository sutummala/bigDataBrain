
# Code created by Sudhakar on June 2020
# Re-arranging the data of IXI (T1, T2 and PD) that matches with BIDS guidelines

import os
import shutil

def do_rearrange(*paths):
    '''rearrgnging the data spedicified in the data_path'''
    
    datapath_T1 = paths[0]+'/IXI-T1'
    datapath_T2 = paths[0]+'/IXI-T2'
    datapath_PD = paths[0]+'/IXI-PD'
    
    files_T1 = sorted(os.listdir(datapath_T1))
    files_T2 = sorted(os.listdir(datapath_T2))
    files_PD = sorted(os.listdir(datapath_PD))
    
    for file in files_T1:
        print(f'doing for {file}\n')
        sub_ID = file[:6]
        datapath_new = paths[1]+'/'+sub_ID+'/anat'
        if not os.path.exists(datapath_new):
            os.makedirs(datapath_new)
        shutil.copy(datapath_T1+'/'+file, datapath_new)
        file_new = file.replace('T1', 'hrT1')
        os.rename(datapath_new+'/'+file, datapath_new+'/'+file_new)
    
    print('all T1 files are rearranged\n')
    
    for file in files_T2:
        print(f'doing for {file}\n')
        sub_ID = file[:6]
        datapath_new = paths[1]+'/'+sub_ID+'/anat'
        if not os.path.exists(datapath_new):
            os.makedirs(datapath_new)
        shutil.copy(datapath_T2+'/'+file, datapath_new)
        file_new = file.replace('T2', 'hrT2')
        os.rename(datapath_new+'/'+file, datapath_new+'/'+file_new)
        
    print('all T2 files are rearranged\n')
    
    for file in files_PD:
        print(f'doing for {file}\n')
        sub_ID = file[:6]
        datapath_new = paths[1]+'/'+sub_ID+'/anat'
        if not os.path.exists(datapath_new):
            os.makedirs(datapath_new)
        shutil.copy(datapath_PD+'/'+file, datapath_new)
        file_new = file.replace('PD', 'hrPD')
        os.rename(datapath_new+'/'+file, datapath_new+'/'+file_new)
        
    print('all PD files are rearranged\n')
            
    
data_dir = "/usr/users/tummala/IXI-Original" # Path to the original subjects 
data_dir1 = "/usr/users/tummala/IXI-Re" # Path to the rearranged subjects 

if not os.path.exists(data_dir1):
    os.makedirs(data_dir1)

do_rearrange(data_dir, data_dir1) # calling the method that is doing the rearrangement

