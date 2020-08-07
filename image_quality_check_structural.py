# created by Sudhakar on July/August 2020
# quality checking before actual pre-processing starts 
# check for general image quality based on several image quality metrics.

import os
import sys
import json
import nibabel as nib
import numpy as np
import all_cost_functions as acf
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn import metrics

def compute_image_quality_metrics(*paths):
    
    print(f'computing image quality metrics for {paths[2]}\n')
    
    image_reference = nib.load(os.path.join(paths[0], paths[2]))
    in_image = image_reference.get_fdata()
    
    main, ext = acf.get_file_name_and_extension(os.path.join(paths[0], paths[2]))
    json_file = main+'.image_quality_metrics.json'
    
    re_compute = False # Re-compute all metrics if this is True
    
    if os.path.exists(json_file) and os.path.getsize(json_file) > 0 and not re_compute:
        print(f'image quality metrics are already computed for {paths[2]} and saved at {json_file}\n')
    else:
        im_quality_msi = acf.msi(in_image, 'sagittal')
        print(f'MSI is {im_quality_msi}\n')
        
        im_quality_snr, im_quality_svnr, _ = acf.snr_and_svnr(paths[0], paths[2])
        print(f'SNR is {im_quality_snr}\nSVNR is {im_quality_svnr}\n')
        
        im_quality_cnr, im_quality_cvnr, im_quality_tctv = acf.cnr_and_cvnr_and_tctv(paths[0], paths[2])
        print(f'CNR is {im_quality_cnr}\nCVNR is {im_quality_cvnr}\nTCTV is {im_quality_tctv}\n')
        
        im_quality_fwhm = acf.fwhm(paths[0], paths[2])
        print(f'FWHM for {paths[1]} is {im_quality_fwhm}\n')
        
        im_quality_ent = acf.ent(paths[0], paths[2])
        print(f'Entropy is {im_quality_ent}\n')
        
        save_image_quality_metrics = {'subject_ID': paths[1], 'image_file': paths[2], 'MSI': im_quality_msi, 'SNR': im_quality_snr, 'SVNR': im_quality_svnr, 'CNR': im_quality_cnr, 'CVNR': im_quality_cvnr, 'TCTV': im_quality_tctv, 'FWHM': im_quality_fwhm, 'ENT': im_quality_ent}
        
        with open(json_file, 'w') as file:
            json.dump(save_image_quality_metrics, file, indent = 4)
            
        print(f'image quality metrics were computed for {paths[2]} and saved at {json_file}\n')
        
def do_clustering(feature_matrix, sub_id):
    '''
    Parameters
    ----------
    feature_matrix : float
        matrix of image quality metrics, each metric is arranged as a column.
    sub_id : str
        id of the subject.

    Returns
    -------
    labels for each subject based on unsupervised learning (clustering in this case).
    '''
    # feature scaling in advance
    scalar = MinMaxScaler()
    feature_matrix_scaled = scalar.fit_transform(feature_matrix)
    
    # 1. K-means and Hierrarchial clustering (agglomerative)
    sse = []
    sil = []
    sil_h = []
    for k in range(1,10):
        kmeans = KMeans(n_clusters=k, init = 'k-means++').fit(feature_matrix_scaled)
        hier = AgglomerativeClustering(n_clusters = k).fit(feature_matrix_scaled)
        sse.append(kmeans.inertia_)
        if k > 1:
            sil.append(metrics.silhouette_score(feature_matrix_scaled, kmeans.labels_))
            sil_h.append(metrics.silhouette_score(feature_matrix_scaled,  hier.labels_))
    
    # plotting elbow method sse and silhouette score for K-Means
    if True:
        plt.subplot(131)
        plt.plot(range(1,10), sse)
        plt.title('SSE vs No.of.Clusters for K-means')
        plt.xlabel('no.of.clusters')
        plt.ylabel('sum of squared errors')
        plt.show()
        
    if True:
        plt.subplot(132)
        plt.plot(range(2,10), sil)
        plt.title('Silhouette score vs No.of.Clusters for K-means')
        plt.xlabel('no.of.clusters')
        plt.ylabel('Silhouette score')
        plt.show()
    
    # plotting silhouette score for hierarchial
    if True:
        plt.subplot(133)
        plt.plot(range(2,10), sil_h)
        plt.title('Silhouette score vs No.of.Clusters for Hierarchial')
        plt.xlabel('no.of.clusters')
        plt.ylabel('Silhouette score')
        plt.show()
        
    # # 2. Hierarchial clustering (agglomerative)
    # hier = AgglomerativeClustering(n_clusters = None, distance_threshold = 5).fit(feature_matrix_scaled)
    # print(f'no.of clusters from Hierarchial Clustering are {hier.n_clusters_}\n')

def main(data_dir, subject):
    '''
    Parameters
    ----------
    data_dir : str
        path to the data directory.
    subject : str
        subject ID.

    Returns
    -------
    a json with image_quality_metrics.

    '''
    raw_image_path = os.path.join(data_dir, subject, 'anat')
    
    if os.path.exists(raw_image_path):
        images = os.listdir(raw_image_path)
        for image in images:
            if any([image.endswith('hrT1.nii.gz'), image.endswith('hrT1.M.nii.gz')]): # hrT1
                compute_image_quality_metrics(raw_image_path, subject, image)
            elif any([image.endswith('hrT2.nii.gz'), image.endswith('hrT2.M.nii.gz')]): # hrT2
                compute_image_quality_metrics(raw_image_path, subject, image)
            elif any([image.endswith('hrFLAIR.nii.gz'), image.endswith('hrFLAIR.M.nii.gz')]): # hrFLAIR
                compute_image_quality_metrics(raw_image_path, subject, image)
            elif any([image.endswith('hrPD.nii.gz'), image.endswith('hrPD.M.nii.gz')]): # hrPD
                compute_image_quality_metrics(raw_image_path, subject, image)
                    
cluster = False
    
if cluster:
    data_dir = sys.argv[1]
    subject = sys.argv[2]
    main(data_dir, subject)
else:
    data_dir = '/usr/users/tummala/HCP-YA'
    subjects = sorted(os.listdir(data_dir))
    
    # image quality metrics
    subject_id = []
    msi = []
    snr = [] 
    svnr = [] 
    cnr = [] 
    cvnr = [] 
    tctv = [] 
    fwhm = [] 
    ent = []

    for subject in subjects:
        
        main(data_dir, subject)
        
        # Loading image quality features into a matrix for clustering
        raw_image_path = os.path.join(data_dir, subject, 'anat')
        
        if os.path.exists(raw_image_path):
            print(f'checking image quality for {subject}++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n')
            for json_file in os.listdir(raw_image_path):
                if json_file.endswith('image_quality_metrics.json') and json_file.find('hrT1') != -1:
                    with open(raw_image_path+'/'+json_file) as in_json:
                        data = json.load(in_json)
                        
                    subject_id.append(data['subject_ID'])
                    msi.append(data['MSI'])
                    snr.append(data['SNR'])
                    svnr.append(data['SVNR'])
                    cnr.append(data['CNR'])
                    cvnr.append(data['CVNR'])
                    tctv.append(data['TCTV'])
                    fwhm.append(data['FWHM'])
                    ent.append(data['ENT'])
        
    im_quality_matrix = np.transpose(np.row_stack((msi, snr, svnr, cnr, cvnr, tctv, fwhm, ent)))
    do_clustering(im_quality_matrix, subject_id)
    
    
                
    
    