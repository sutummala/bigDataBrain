# created by Sudhakar on August 2020 
# validate structural image quality metrics


import os
import seaborn as sns
import json
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import MaxAbsScaler
import matplotlib.pyplot as plt
from sklearn import metrics
import check_registration as cr

plot_kwds = {'alpha' : 0.25, 's' : 80, 'linewidths':0}

# only possible to visualize when dealing with just 2 features        
def plot_clusters(data, algorithm, args, kwds):
    '''
    Parameters
    ----------
    data : float
        matrix of features
    algorithm : str
        type of clustering algorithm.
    args : str
        any other arguments.
    kwds : any
        key word arguments.

    Returns
    -------
    a figure with clusters color coded.

    '''
    labels = algorithm(*args, **kwds).fit_predict(data)
    palette = sns.color_palette('deep', np.unique(labels).max() + 1)
    colors = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in labels]
    plt.scatter(data.T[0], data.T[1], c=colors, **plot_kwds)
    frame = plt.gca()
    frame.axes.get_xaxis().set_visible(False)
    frame.axes.get_yaxis().set_visible(False)
    plt.title('Clusters found by {}'.format(str(algorithm.__name__)), fontsize=24)
    #plt.text(-0.5, 0.7, 'Clustering took {:.2f} s'.format(end_time - start_time), fontsize=14)
        
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
    scalar = MaxAbsScaler()
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
    if False:
        plt.figure()
        plt.plot(range(1,10), sse)
        plt.title('SSE vs No.of.Clusters for K-means')
        plt.xlabel('no.of.clusters')
        plt.ylabel('sum of squared errors')
        plt.show()
    
    # plotting silhouette score for k-means
    if True:
        plt.figure()
        plt.subplot(121)
        plt.plot(range(2,10), sil)
        plt.title('Silhouette score vs No.of.Clusters for K-means')
        plt.xlabel('no.of.clusters')
        plt.ylabel('Silhouette score')
        plt.show()
    
    # plotting silhouette score for hierarchial
    if True:
        plt.subplot(122)
        plt.plot(range(2,10), sil_h)
        plt.title('Silhouette score vs No.of.Clusters for Hierarchial')
        plt.xlabel('no.of.clusters')
        plt.ylabel('Silhouette score')
        plt.show()
        
    # # 2. Hierarchial clustering (agglomerative)
    # hier = AgglomerativeClustering(n_clusters = None, distance_threshold = 5).fit(feature_matrix_scaled)
    # print(f'no.of clusters from Hierarchial Clustering are {hier.n_clusters_}\n')


data_dir = '/usr/users/tummala/bigdata1'
subjects = sorted(os.listdir(data_dir))

# subject IDs and image quality metrics
subject_id = []
msi = []
snr = [] 
svnr = [] 
cnr = [] 
cvnr = [] 
tctv = [] 
fwhm = [] 
ent = []
    
quality_flag = []
subjects_with_quality_flag = []

unsupervised = False

quality_label = []

for subject in subjects:
    
 # Loading image quality features into a matrix for clustering
    raw_image_path = os.path.join(data_dir, subject, 'anat')
        
    if os.path.exists(raw_image_path):
        for json_file in os.listdir(raw_image_path):
            if json_file.endswith('image_quality_metrics.json') and json_file.find('hrFLAIR') != -1:
                with open(raw_image_path+'/'+json_file) as in_json:
                    data = json.load(in_json)
                    subject_id.append(data['subject_ID'])
                    msi.append(data['MSI']) # mean signal intensity (computed along sagittal)
                    snr.append(data['SNR']) # signal to noise ratio
                    svnr.append(data['SVNR']) # signal variance to noise variance ratio
                    cnr.append(data['CNR']) # contrast to noise ratio
                    cvnr.append(data['CVNR']) # contrast variance to noise variance ratio
                    tctv.append(data['TCTV']) # tissue contrast to tissue variance ratio
                    fwhm.append(data['FWHM']) # full width half maximum
                    ent.append(data['ENT']) # entropy
            elif json_file.endswith('hrFLAIR.json'):
                with open(raw_image_path+'/'+json_file) as in_meta_json:
                    meta_data = json.load(in_meta_json)
                    
                    if meta_data['NMRI-Technical'] == ['good']:
                        quality_label.append(0)
                    else:
                        quality_label.append(1)
                        
                    quality_flag.append(meta_data['NMRI-Technical'])
                    subjects_with_quality_flag.append(meta_data['Subject'])
                    
im_quality_matrix = np.transpose(np.row_stack((snr, cnr, tctv, fwhm))) # creating a matrix by combining different quality metrics
                   
if unsupervised:
    print(f'doing clustering to find the number of clusters automatically\n')                        
    do_clustering(im_quality_matrix, subject_id)
else:
    print(f'doing supervised learning classification by making use of available quality flags\n')
    quality_label = np.array(quality_label)
    
    data1 = im_quality_matrix[quality_label == 0]
    data2 = im_quality_matrix[quality_label == 1]
    
    cr.combinational_cost(np.transpose(data1), np.transpose(data2), 'image_quality_check', 'T1', 2)
     