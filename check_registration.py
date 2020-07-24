# code created by Sudhakar on May 2020
# check registration


import os
import all_plots as ap
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt


subpath1 = '/usr/users/tummala/bigdata1'
subpath2 = '/usr/users/tummala/HCP-YA'

voi_size = ''
step_size = ''
# subpath3 = '/usr/users/tummala/bigdata'
# subpath4 = '/usr/users/tummala/HCP-YA-test'

def remove_nan(data):
    '''
    Parameters
    ----------
    data : list
        input list, could have nan values.

    Returns
    -------
    type: list
        all nan values from the input will be removed.
    '''
    data = np.array(data)
    return data[~np.isnan(data)]

def get_coreg_cost_vectors(cost_func, subpath, tag):
    '''
    Parameters
    ----------
    cost_func : str
        cost name, ncc:normalized correlation coefficient, nmi: normalized mutual information.
    subpath : str
        path containing all subjects.
    tag : str
        returns a tag to identify the image type.

    Returns
    -------
    type: str
        returns global and local cost vectors for all subjects under study.
    '''
    subjects = os.listdir(subpath)
    global_cost_vector = []
    local_cost_vector = []
    for index, subject in enumerate(subjects, start=1):
        cost_folder = subpath+'/'+subject+'/cost'+str(voi_size)+str(step_size)
        data_files = os.listdir(cost_folder)
        for data_file in data_files:
            if 'alignedToT1' in data_file and (tag in data_file and cost_func in data_file):
                cost_data = np.loadtxt(cost_folder+'/'+data_file)
                global_cost_vector.append(cost_data[0])
                local_cost_vector.append(cost_data[1])
    return remove_nan(global_cost_vector), remove_nan(local_cost_vector)

def get_cost_vectors(cost_func, reg_type, subpath, tag):
    '''
    Parameters
    ----------
    cost_func : str
        cost name, ncc:normalized correlation coefficient, nmi: normalized mutual information.
    reg_type: str
        registration type, align for rigid and mni for affine
    subpath : str
        path containing all subjects.
    tag : str
        returns a tag to identify the image type.

    Returns
    -------
    type: str
        returns global and local cost vectors for all subjects under study.
    '''
    subjects = os.listdir(subpath)
    global_cost_vector = []
    local_cost_vector = []
    for index, subject in enumerate(subjects, start=1):
        cost_folder = subpath+'/'+subject+'/cost'+str(voi_size)+str(step_size)
        #print('{}-{}, {}-{}'.format(index, subject, reg_type, cost_func))
        data_files = os.listdir(cost_folder)
        for data_file in data_files:
            if reg_type in data_file and (tag in data_file and cost_func in data_file):
                if not 'alignedToT1' in data_file:
                    cost_data = np.loadtxt(cost_folder+'/'+data_file)
                    global_cost_vector.append(cost_data[0])
                    local_cost_vector.append(cost_data[1])
    return remove_nan(global_cost_vector), remove_nan(local_cost_vector)

def get_coreg_test_cost_vectors(cost_func, subpath, tag):
    '''
    Parameters
    ----------
    cost_func : str
        cost name, ncc:normalized correlation coefficient, nmi: normalized mutual information.
    subpath : str
        path containing all subjects.
    tag : str
        returns a tag to identify the image type.

    Returns
    -------
    type: str
        returns global and local test cost vectors for all subjects under study.
    '''
    subjects = os.listdir(subpath)
    global_cost_vector = []
    local_cost_vector = []
    for index, subject in enumerate(subjects, start=1):
        if tag == 'hrT2':
            cost_folder = subpath+'/'+subject+'/test_cost_T2_T1'+str(voi_size)+str(step_size)
        elif tag == 'hrFLAIR':
            cost_folder = subpath+'/'+subject+'/test_cost_FLAIR_T1'+str(voi_size)+str(step_size)

        if os.path.exists(cost_folder) and os.listdir(cost_folder):
            data_files = os.listdir(cost_folder)
            for data_file in data_files:
                if (tag in data_file and cost_func in data_file): 
                    #print(reg_type, tag, cost_func)
                    cost_data = np.loadtxt(cost_folder+'/'+data_file)
                    global_cost_vector.append(cost_data[0])
                    local_cost_vector.append(cost_data[1])
    return remove_nan(global_cost_vector), remove_nan(local_cost_vector)

def get_test_cost_vectors(cost_func, reg_type, subpath, tag):
    '''
    Parameters
    ----------
    cost_func : str
        cost name, ncc:normalized correlation coefficient, nmi: normalized mutual information.
    reg_type: str
        registration type, align for rigid and mni for affine
    subpath : str
        path containing all subjects.
    tag : str
        returns a tag to identify the image type.

    Returns
    -------
    type: str
        returns global and local test cost vectors for all subjects under study.
    '''
    subjects = os.listdir(subpath)
    global_cost_vector = []
    local_cost_vector = []
    for index, subject in enumerate(subjects, start=1):
        if tag == 'hrT1':
            if reg_type == 'align':
                cost_folder = subpath+'/'+subject+'/test_cost_T1_align'+str(voi_size)+str(step_size)
            elif reg_type == 'mni':
                cost_folder = subpath+'/'+subject+'/test_cost_T1_mni'+str(voi_size)+str(step_size)
        elif tag == 'hrT2':
            if reg_type == 'align':
                cost_folder = subpath+'/'+subject+'/test_cost_T2_align'+str(voi_size)+str(step_size)
            elif reg_type == 'mni':
                cost_folder = subpath+'/'+subject+'/test_cost_T2_mni'+str(voi_size)+str(step_size)
        elif tag == 'hrFLAIR':
            if reg_type == 'align':
                cost_folder = subpath+'/'+subject+'/test_cost_FLAIR_align'+str(voi_size)+str(step_size)
            elif reg_type == 'mni':
                cost_folder = subpath+'/'+subject+'/test_cost_FLAIR_mni'+str(voi_size)+str(step_size)
        
        if os.path.exists(cost_folder) and os.listdir(cost_folder):
            data_files = os.listdir(cost_folder)
            for data_file in data_files:
                if (tag in data_file and cost_func in data_file):
                    #print(reg_type, tag, cost_func)
                    cost_data = np.loadtxt(cost_folder+'/'+data_file)
                    global_cost_vector.append(cost_data[0])
                    local_cost_vector.append(cost_data[1])
    return remove_nan(global_cost_vector), remove_nan(local_cost_vector)

def compute_cutoff_auc(data1, data2, *tags):
    '''computes cut-off point and AUC for given cost and reg type from data1 (normal vlaues), data2 (test values)'''
       
    labels = np.concatenate([np.ones(len(data1)), np.zeros(len(data2))])
    print(f'{len(data1)}, {len(data2)}')
    fpr, tpr, thresholds = metrics.roc_curve(labels, np.concatenate([data1, data2]), pos_label = 1)
    
    print(f'Threshold for {tags[2]}-{tags[1]}-{tags[3]}-{tags[0]} is: {thresholds[np.argmax(tpr-fpr)]}, sensitivity (recall) is: {tpr[np.argmax(tpr-fpr)]}, specificity is: {1-fpr[np.argmax(tpr-fpr)]}, fall-out is: {fpr[np.argmax(tpr-fpr)]}, AUC is: {metrics.auc(fpr, tpr)}\n')
    

def combinational_cost(data1, data2, reg_type, image_tag):
    '''
    Parameters
    ----------
    data1 : arrays
        matrix of all costs of group1.
    data2 : arrays
        matrix of all costs of group2.
    reg_type : str
        registration type, either rigid (align) or affine (affine).

    Returns
    -------
    accuracy of the combinational cost function for identifying mis-registrations.

    '''
    print(f'plotting classifier comparison for {image_tag}-{reg_type}')
    
    # transposing and creating labels for data1    
    X_normal = np.transpose(data1)
    x_normal_label = np.zeros(len(X_normal))
    
    # transposing and creating labels for data2    
    X_misaligned = np.transpose(data2)
    x_misaligned_label = np.ones(len(X_misaligned))
    
    # combining data1 and data2 and the corresponding labels    
    X = np.concatenate((X_normal, X_misaligned))
    y = np.concatenate((x_normal_label, x_misaligned_label))
    
    # splitting the data and labels into training and testing     
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
    
    # scaling the costs (features) to make sure the ranges of individual features are same to avoid the effect of features that have relatively large values. It may not be necessary in this case    
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    # 1. Linear Discriminant Analysis classifier
    lda = LDA(solver = 'eigen', shrinkage = 'auto', n_components = 1)
    # X_train = lda.fit_transform(X_train, y_train)
    # X_test = lda.transform(X_test)
    # fitting the lda classifier using training data and then predicting on testing 
    lda.fit(X_train, y_train)
    y_pred_lda = lda.predict(X_test)
    
    # 2. Random Forest classifier (it could be done in LDA transformed space if you have large number of features)  
    rfc = RandomForestClassifier(criterion = 'entropy', max_depth = 2, random_state = 0)
    # fitting the rfc classfier using training data, and then predicting on testing
    rfc.fit(X_train, y_train)
    y_pred_rfc = rfc.predict(X_test)
    
    # 3. Support Vector Machine classifier
    svm = SVC(random_state = 0)
    svm.fit(X_train, y_train)
    y_pred_svm = svm.predict(X_test)
    
    # 4. Gaussian Naive Bayes classifier
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    y_red_gnb = gnb.predict(X_test)
    
    lda_disp = metrics.plot_roc_curve(lda, X_test, y_test)
    svm_disp = metrics.plot_roc_curve(svm, X_test, y_test, ax = lda_disp.ax_)
    gnb_disp = metrics.plot_roc_curve(gnb, X_test, y_test, ax = lda_disp.ax_)
    rfc_disp = metrics.plot_roc_curve(rfc, X_test, y_test, ax = lda_disp.ax_)
    rfc_disp.figure_.suptitle(f"ROC curve comparison {image_tag}")

    plt.show()
    
    # # confusion matrix and calculating the accuracy
    # cm = metrics.confusion_matrix(y_test, y_pred)
    # print(cm)
    # print('Accuracy' + str(metrics.accuracy_score(y_test, y_pred)))    
        

if __name__ == '__main__':
    
    costs = ['ncc', 'nmi', 'cor']
    reg_types = ['align', 'mni']

    for reg_type in reg_types:
        combine_cost_vector_T1 = []
        combine_test_cost_vector_T1 = []
        
        combine_cost_vector_T2 = []
        combine_test_cost_vector_T2 = []
        
        combine_cost_vector_FLAIR = []
        combine_test_cost_vector_FLAIR = []
        for cost in costs:
            # getting normal values for hrT1, hrT2 and hrFLAIR for bigdata 
            global_cost_vector_bigdata_T1, local_cost_vector_bigdata_T1 = get_cost_vectors(cost, reg_type, subpath1, 'hrT1') # T1 to MNI
            global_cost_vector_bigdata_FLAIR, local_cost_vector_bigdata_FLAIR = get_cost_vectors(cost, reg_type, subpath1, 'hrFLAIR') # FLAIR to MNI
            global_cost_vector_bigdata_FLAIRtoT1, local_cost_vector_bigdata_FLAIRtoT1 = get_coreg_cost_vectors(cost, subpath1, 'hrFLAIR') # FLAIR brain to T1 brain (only align)
            
            combine_cost_vector_T1.append(local_cost_vector_bigdata_T1) # T1 data
            combine_cost_vector_FLAIR.append(local_cost_vector_bigdata_FLAIR) # FLAIR data
            
            # HCP-YA
            global_cost_vector_hcpya_T1, local_cost_vector_hcpya_T1 = get_cost_vectors(cost, reg_type, subpath2, 'hrT1') # T1 to MNI
            global_cost_vector_hcpya_T2, local_cost_vector_hcpya_T2 = get_cost_vectors(cost, reg_type, subpath2, 'hrT2') # T2 to MNI
            global_cost_vector_hcpya_T2toT1, local_cost_vector_hcpya_T2toT1 = get_coreg_cost_vectors(cost, subpath2, 'hrT2') # T2 brain to T1 brain (only align)
            
            combine_cost_vector_T2.append(local_cost_vector_hcpya_T2) # T2 data 
    
            if False:
                # plotting normal values for T1, T2 and FLAIR
                ap.plot_cost([global_cost_vector_bigdata_T1, global_cost_vector_hcpya_T1, global_cost_vector_hcpya_T2, global_cost_vector_bigdata_FLAIR], cost,
                          ['T1', 'T1(hcp)', 'T2(hcp)', 'FLAIR'], f'global-{reg_type}') # plotting global cost
            if False:
                ap.plot_cost([local_cost_vector_bigdata_T1, local_cost_vector_hcpya_T1, local_cost_vector_hcpya_T2, local_cost_vector_bigdata_FLAIR], cost,
                          ['T1', 'T1(hcp)', 'T2(hcp)', 'FLAIR'], f'local-{reg_type}') # plotting local cost
            
            # getting test values for hrT1, hrT2 and hrFLAIR for bigdata 
            global_test_cost_vector_bigdata_T1, local_test_cost_vector_bigdata_T1 = get_test_cost_vectors(cost, reg_type, subpath1, 'hrT1') # T1 to MNI
            global_test_cost_vector_bigdata_FLAIR, local_test_cost_vector_bigdata_FLAIR = get_test_cost_vectors(cost, reg_type, subpath1, 'hrFLAIR') # FLAIR to MNI
            global_test_cost_vector_bigdata_FLAIRtoT1, local_test_cost_vector_bigdata_FLAIRtoT1 = get_coreg_test_cost_vectors(cost, subpath1, 'hrFLAIR') # FLAIR brain to T1 brain (only align)
            
            combine_test_cost_vector_T1.append(local_test_cost_vector_bigdata_T1) # T1 test data
            combine_test_cost_vector_FLAIR.append(local_test_cost_vector_bigdata_FLAIR) # FLAIR test data
            
            # HCPYA
            global_test_cost_vector_hcpya_T1, local_test_cost_vector_hcpya_T1 = get_test_cost_vectors(cost, reg_type, subpath2, 'hrT1') # T1 to MNI
            global_test_cost_vector_hcpya_T2, local_test_cost_vector_hcpya_T2 = get_test_cost_vectors(cost, reg_type, subpath2, 'hrT2') # T2 to MNI
            global_test_cost_vector_hcpya_T2toT1, local_test_cost_vector_hcpya_T2toT1 = get_coreg_test_cost_vectors(cost, subpath2, 'hrT2') # T2 to T1 (only align)
            
            combine_test_cost_vector_T2.append(local_test_cost_vector_hcpya_T2) # T2 test data
            
            if False:
                ap.plot_cost([local_cost_vector_bigdata_T1, local_test_cost_vector_bigdata_T1], cost,
                          ['T1', 'T1-test', 'T1(local)', 'T1-test(local)'], f'Big-Data {reg_type}') # plotting local cost for bigdata T1
            if False:
                ap.plot_cost([local_cost_vector_bigdata_FLAIR, local_test_cost_vector_bigdata_FLAIR], cost,
                          ['FLAIR', 'FLAIR-test', 'FLAIR(local)', 'FLAIR-test(local)'], f'Big-Data {reg_type}') # plotting local cost for bigdata FLAIR
            if False and reg_type == 'align':
                ap.plot_cost([local_cost_vector_bigdata_FLAIRtoT1, local_test_cost_vector_bigdata_FLAIRtoT1], cost,
                          ['FLAIR-T1', 'FLAIR-T1-test'], f'Big-Data-Align') # plotting local cost for bigdata FLAIR
                
            # Plotting for HCP-YA 
            if False:    
                ap.plot_cost([local_cost_vector_hcpya_T1, local_test_cost_vector_hcpya_T1], cost,
                          ['T1', 'T1-test', 'T1(local)', 'T1-test(local)'], f'HCP-YA {reg_type}') # plotting local cost for HCPYA T1
            if False:
                ap.plot_cost([local_cost_vector_hcpya_T2, local_test_cost_vector_hcpya_T2], cost,
                          ['T2', 'T2-test', 'T2(local)', 'T2-test(local)'], f'HCP-YA {reg_type}') # plotting local cost for HCPYA T2
            if False and reg_type == 'align':
                ap.plot_cost([local_cost_vector_hcpya_T2toT1, local_test_cost_vector_hcpya_T2toT1], cost,
                          ['T2-T1', 'T2-T1-test'], f'HCP-YA-Align') # plotting local cost for HCPYA T2
            
            # # Computing cut-off point and AUC for normal and test
            # print('doing for Big Data\n')
            
            # #compute_cutoff_auc(global_cost_vector_bigdata_T1, global_test_cost_vector_bigdata_T1, cost, reg_type, 'hrT1', 'global') # T1 to MNI
            # compute_cutoff_auc(local_cost_vector_bigdata_T1, local_test_cost_vector_bigdata_T1, cost, reg_type, 'hrT1', 'local') # T1 to MNI
            # print('----------------------------------------------------------------------------------------------------')
            
            # #compute_cutoff_auc(global_cost_vector_bigdata_FLAIR, global_test_cost_vector_bigdata_FLAIR, cost, reg_type, 'hrFLAIR', 'global') # FLAIR to MNI
            # compute_cutoff_auc(local_cost_vector_bigdata_FLAIR, local_test_cost_vector_bigdata_FLAIR, cost, reg_type, 'hrFLAIR', 'local') # FLAIR to MNI
            # print('----------------------------------------------------------------------------------------------------')
            
            # if reg_type == 'align':
            #     #compute_cutoff_auc(global_cost_vector_bigdata_FLAIRtoT1, global_test_cost_vector_bigdata_FLAIRtoT1, cost, 'T1', 'hrFLAIR', 'global') # FLAIR brain to T1 brain
            #     compute_cutoff_auc(local_cost_vector_bigdata_FLAIRtoT1, local_test_cost_vector_bigdata_FLAIRtoT1, cost, 'T1', 'hrFLAIR', 'local') # FLAIR brain to T1 brain
            #     print('----------------------------------------------------------------------------------------------------')
            
            # print('doing for HCP-YA\n')
            
            # #compute_cutoff_auc(global_cost_vector_hcpya_T1, global_test_cost_vector_hcpya_T1, cost, reg_type, 'hrT1', 'global') # T1 to MNI
            # compute_cutoff_auc(local_cost_vector_hcpya_T1, local_test_cost_vector_hcpya_T1, cost, reg_type, 'hrT1', 'local') # T1 to MNI
            # print('----------------------------------------------------------------------------------------------------')
            
            # #compute_cutoff_auc(global_cost_vector_hcpya_T2, global_test_cost_vector_hcpya_T2, cost, reg_type, 'hrT2', 'global') # T2 to MNI
            # compute_cutoff_auc(local_cost_vector_hcpya_T2, local_test_cost_vector_hcpya_T2, cost, reg_type, 'hrT2', 'local') # T2 to MNI
            # print('----------------------------------------------------------------------------------------------------')
            
            # if reg_type == 'align':
            #     #compute_cutoff_auc(global_cost_vector_hcpya_T2toT1, global_test_cost_vector_hcpya_T2toT1, cost, 'T1', 'hrT2', 'global') # T2 brain to T1 brain
            #     compute_cutoff_auc(local_cost_vector_hcpya_T2toT1, local_test_cost_vector_hcpya_T2toT1, cost, 'T1', 'hrT2', 'local') # T2 brain to T1 brain
        
        # calling combinational_cost method to design a classifier 
        combinational_cost(combine_cost_vector_T1, combine_test_cost_vector_T1, reg_type, 'T1')
        combinational_cost(combine_cost_vector_T2, combine_test_cost_vector_T2, reg_type, 'T2')
        combinational_cost(combine_cost_vector_FLAIR, combine_test_cost_vector_FLAIR, reg_type, 'FLAIR')
        
        
        
        
        
