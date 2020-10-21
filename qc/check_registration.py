# code created by Sudhakar on May 2020 and modified on August 2020 to add different classifiers
# check registration


import os
import all_plots as ap
import numpy as np
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier as kNN
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle

plt.rcParams.update({'font.size': 22})
subpath1 = "/home/tummala/data/ABIDEre"
subpath2 = subpath1

voi_size = 3
step_size = 3 # stride
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
    '''computes cut-off point and AUC for given cost and reg type from data1 (normal values), data2 (test values)'''
    
    if False:
        data2 = data2[:np.shape(data1)[0]] # To make sizes of both classes same
    labels = np.concatenate([np.ones(len(data1)), np.zeros(len(data2))])
    print(f'{len(data1)}, {len(data2)}')
    fpr, tpr, thresholds = metrics.roc_curve(labels, np.concatenate([data1, data2]), pos_label = 1)
    
    print(f'Threshold for {tags[2]}-{tags[1]}-{tags[3]}-{tags[0]} is: {thresholds[np.argmax(tpr-fpr)]}, sensitivity (recall) is: {tpr[np.argmax(tpr-fpr)]}, specificity is: {1-fpr[np.argmax(tpr-fpr)]}, fall-out is: {fpr[np.argmax(tpr-fpr)]}, AUC is: {metrics.auc(fpr, tpr)}\n')
    
def classifier_accuracy(model, X_train, X_test, y_train, y_test):
    'get model (classifier) accuracy based on training and testing'
    
    model.fit(X_train, y_train)
    #print(f'F1-score is {metrics.f1_score(y_test, model.predict(X_test))}/n')
    #return model.score(X_test, y_test), metrics.roc_auc_score(y_test, model.predict_proba(X_test)[:,1])
    #precision, recall, thresholds = metrics.precision_recall_curve(y_test, model.predict_proba(X_test)[:,1])
    #print(f'area under PR curve is {metrics.auc(recall, precision)}')
    #print(X_test)
    #print(y_test-model.predict(X_test))
    return metrics.f1_score(y_test, model.predict(X_test)), metrics.roc_auc_score(y_test, model.predict_proba(X_test)[:,1]), model      

def combinational_cost(data1, data2, reg_type, image_tag, no_of_folds):
    '''
    Parameters
    ----------
    data1 : arrays
        matrix of all costs of group1 (normal). Each individual cost (feature) should be arrnaged as a column
    data2 : arrays
        matrix of all costs of group2 (abnormal). Each individual cost (feature) should be arrnaged as a column
    reg_type : str
        registration type, either rigid (6-dof) or affine (12-dof), or it could be non-linear.
    no_of_folds : int
        specify number of folds for nested cross-validation

    Returns
    -------
    accuracy and AUC of the combinational cost function based on different supervised-learning classifiers for identifying mis-registrations.

    '''
    print(f'classifier comparison for {image_tag}-{reg_type}--------------')
    
    # transposing and creating labels for data1    
    X_normal = np.transpose(data1) # to make each feature into a column
    x_normal_label = np.zeros(len(X_normal))
    print(f'number of correctly aligned images are {len(x_normal_label)}')
    # transposing and creating labels for data2
    if False:    
        X_misaligned = np.transpose(data2)[:np.shape(X_normal)[0], :]
    else:
        X_misaligned = np.transpose(data2)
    x_misaligned_label = np.ones(len(X_misaligned))
    print(f'number of misaligned images are {len(x_misaligned_label)}')
    
    # combining data1 and data2 and the corresponding labels    
    X = np.concatenate((X_normal, X_misaligned))
    y = np.concatenate((x_normal_label, x_misaligned_label))
       
    # scaling the costs (features) to make sure the ranges of individual features are same to avoid the effect of features that have relatively large values. It may not be necessary in this case as all these 3 costs lie between 0 and 1  
    scale = MaxAbsScaler()
    X = scale.fit_transform(X)
    
    # K-fold cross validation, n_splits specifies the number of folds
    folds = StratifiedKFold(n_splits = no_of_folds)
    
    scores_lda = []
    scores_qda = [] 
    scores_rfc = []
    scores_svm = []
    scores_gnb = []
    scores_knn = []
    scores_lor = []
    scores_ada = []
    scores_gra = []
    scores_ann = []
    
    auc_lda = []
    auc_qda = [] 
    auc_rfc = []
    auc_svm = []
    auc_gnb = []
    auc_knn = []
    auc_lor = []
    auc_ada = []
    auc_gra = []
    auc_ann = []
    
    for train_index, test_index in folds.split(X, y):
        
        X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
    
        # 1. Linear Discriminant Analysis Classifier
        lda = LDA(solver = 'eigen', shrinkage = 'auto', n_components = 1)
        score_lda, roc_auc_lda, model_lda = classifier_accuracy(lda, X_train, X_test, y_train, y_test)
        scores_lda.append(score_lda) # F1-score
        auc_lda.append(roc_auc_lda) # AUC
        
        # 1a. Quadratic Discriminant Analysis Classifier
        qda = QDA()
        score_qda, roc_auc_qda, model_qda = classifier_accuracy(qda, X_train, X_test, y_train, y_test)
        scores_qda.append(score_qda)
        auc_qda.append(roc_auc_qda)
        
        # 2. Random Forest Classifier (it could be done in LDA transformed space if you have large number of features)
        rfc = RandomForestClassifier(criterion = 'gini', n_estimators = 100)
        score_rfc, roc_auc_rfc, model_rfc = classifier_accuracy(rfc, X_train, X_test, y_train, y_test)
        scores_rfc.append(score_rfc)
        auc_rfc.append(roc_auc_rfc)
        
        # 3. Support Vector Machine Classifier
        svc = SVC(kernel = 'linear', gamma = 'scale', probability = True)
        score_svm, roc_auc_svm, model_svm = classifier_accuracy(svc, X_train, X_test, y_train, y_test)
        scores_svm.append(score_svm)
        auc_svm.append(roc_auc_svm)
        #print(svc.coef_)
        
        # 4. Gaussian Naive Bayes Classifier
        gnb = GaussianNB()
        score_gnb, roc_auc_gnb, model_gnb = classifier_accuracy(gnb, X_train, X_test, y_train, y_test)
        scores_gnb.append(score_gnb)
        auc_gnb.append(roc_auc_gnb)
        
        # 5. k-Nearest Neighbour Classifier
        knn = kNN(n_neighbors = 15)
        score_knn, roc_auc_knn, model_knn = classifier_accuracy(knn, X_train, X_test, y_train, y_test)
        scores_knn.append(score_knn)
        auc_knn.append(roc_auc_knn)
        
        # 6. Logistic Regression Classifier
        lor = LogisticRegression()
        score_lor, roc_auc_lor, model_lor = classifier_accuracy(lor, X_train, X_test, y_train, y_test)
        scores_lor.append(score_lor)
        auc_lor.append(roc_auc_lor)
        
        # 7. Ada Boost Classifier
        ada = AdaBoostClassifier(n_estimators = 100)
        score_ada, roc_auc_ada, model_ada = classifier_accuracy(ada, X_train, X_test, y_train, y_test)
        scores_ada.append(score_ada)
        auc_ada.append(roc_auc_ada)
        
        # 7a. Gradient Boosting Classifier
        gra = GradientBoostingClassifier(random_state = 0)
        score_gra, roc_auc_gra, model_gra = classifier_accuracy(gra, X_train, X_test, y_train, y_test)
        scores_gra.append(score_gra)
        auc_gra.append(roc_auc_gra)
        
        # 8. Arteficial Neural Network (Deep Learning)
        # model_ann = tf.keras.models.Sequential()
        # model_ann.add(tf.keras.layers.Dense(units = np.shape(X_train)[1] + 1, activation = 'relu', input_shape = (np.shape(X_train)[1],))) # input_shape takes height of the input layer which is usually fed during first dense layer allocation
        # model_ann.add(tf.keras.layers.Dense(units = np.shape(X_train)[1] + 1, activation = 'relu')) # hidden layer
        # model_ann.add(tf.keras.layers.Dense(units = np.shape(X_train)[1] + 2, activation = 'relu')) # hidden layer
        # model_ann.add(tf.keras.layers.Dense(units = np.shape(X_train)[1] + 1, activation = 'relu')) # hidden layer
        # model_ann.add(tf.keras.layers.Dense(units = 2, activation = 'softmax')) # hidden layer
        # model_ann.compile(optimizer = 'sgd', loss = 'binary_crossentropy', metrics = ['accuracy']) # compile the neural network
        # model_ann.fit(X_train, y_train, epochs = 20) # fit the neural network on the training data
        # scores_ann.append(model_ann.evaluate(X_test, y_test)) # network accuracy
        # auc_ann.append(metrics.roc_auc_score(y_test, model_ann.predict_proba(X_test)[:, 1])) # network AUC
        
    # Note: 'cross_val_score' method from sklearn could be used directly on the classifier model to avoid the above for loop. Further, f1-score could be used instead of accuracy metric if number of positive samples (mis-aligned) are low.
    
    print(f'accuracy using LDA classifier for {image_tag}-{reg_type} is: {np.average(scores_lda)}, AUC is: {np.average(auc_lda)}\n')
    print(f'accuracy using QDA classifier for {image_tag}-{reg_type} is: {np.average(scores_qda)}, AUC is: {np.average(auc_qda)}\n')
    print(f'accuracy using RandomForest classifier for {image_tag}-{reg_type} is: {np.average(scores_rfc)}, AUC is: {np.average(auc_rfc)}\n')
    print(f'accuracy using SVM classifier for {image_tag}-{reg_type} is: {np.average(scores_svm)}, AUC is: {np.average(auc_svm)}\n')
    print(f'accuracy using Naive Bayes classifier for {image_tag}-{reg_type} is: {np.average(scores_gnb)}, AUC is: {np.average(auc_gnb)}\n')
    print(f'accuracy using kNN classifier for {image_tag}-{reg_type} is: {np.average(scores_knn)}, AUC is: {np.average(auc_knn)}\n')
    print(f'accuracy using Logistic Regression classifier for {image_tag}-{reg_type} is: {np.average(scores_lor)}, AUC is: {np.average(auc_lor)}\n')
    print(f'accuracy using Ada Boost classifier for {image_tag}-{reg_type} is: {np.average(scores_ada)}, AUC is: {np.average(auc_ada)}\n')
    print(f'accuracy using Gradient boosting classifier for {image_tag}-{reg_type} is: {np.average(scores_gra)}, AUC is: {np.average(auc_gra)}\n')
    #print(f'accuracy using ANN for {image_tag}-{reg_type} is: {np.average(scores_ann)}, AUC is: {np.average(auc_ann)}\n')
    
    save_model = '/home/tummala/mri/ml_classifier_models_checking_reg'
    
    if not os.path.exists(save_model):
        os.makedirs(save_model)
    
    # saving the trained model, e.g. shown for saving ada boost classifier model and minmax scaling model
    pickle.dump(scale, open(save_model+'/'+'scale_'+reg_type+image_tag, 'wb'))
    pickle.dump(model_lda, open(save_model+'/'+'lda_'+reg_type+image_tag, 'wb'))
    pickle.dump(model_qda, open(save_model+'/'+'qda_'+reg_type+image_tag, 'wb'))
    pickle.dump(model_rfc, open(save_model+'/'+'rfc_'+reg_type+image_tag, 'wb'))
    pickle.dump(model_svm, open(save_model+'/'+'svm_'+reg_type+image_tag, 'wb'))
    pickle.dump(model_gnb, open(save_model+'/'+'gnb_'+reg_type+image_tag, 'wb'))
    pickle.dump(model_knn, open(save_model+'/'+'knn_'+reg_type+image_tag, 'wb'))
    pickle.dump(model_lor, open(save_model+'/'+'lor_'+reg_type+image_tag, 'wb'))
    pickle.dump(model_ada, open(save_model+'/'+'ada_boost_'+reg_type+image_tag, 'wb'))
    # pickle.load method could be used to load the model for later use and predict method of the seved model to categorize new cases
    
    # plotting ROC curve for Sensitivity/Specificity all above classifiers
    if False:
        lda_disp = metrics.plot_roc_curve(lda, X_test, y_test)
        qda_disp = metrics.plot_roc_curve(qda, X_test, y_test, ax = lda_disp.ax_)
        svm_disp = metrics.plot_roc_curve(svc, X_test, y_test, ax = lda_disp.ax_)
        #nsvm_disp = metrics.plot_roc_curve(nsvm, X_test, y_test, ax = lda_disp.ax_)
        gnb_disp = metrics.plot_roc_curve(gnb, X_test, y_test, ax = lda_disp.ax_)
        rfc_disp = metrics.plot_roc_curve(rfc, X_test, y_test, ax = lda_disp.ax_)
        ada_disp = metrics.plot_roc_curve(ada, X_test, y_test, ax = lda_disp.ax_)
        knn_disp = metrics.plot_roc_curve(knn, X_test, y_test, ax = lda_disp.ax_)
        #knn_disp.figure_.suptitle(f"ROC curve comparison {image_tag}-{reg_type}")
        
    # plotting Precision-Recall ROC curve for all above classifiers
    if False:
        lda_disp = metrics.precision_recall_curve(lda, X_test, y_test)
        qda_disp = metrics.precision_recall_curve(qda, X_test, y_test, ax = lda_disp.ax_)
        svm_disp = metrics.precision_recall_curve(svc, X_test, y_test, ax = lda_disp.ax_)
        #nsvm_disp = metrics.plot_roc_curve(nsvm, X_test, y_test, ax = lda_disp.ax_)
        gnb_disp = metrics.precision_recall_curve(gnb, X_test, y_test, ax = lda_disp.ax_)
        rfc_disp = metrics.precision_recall_curve(rfc, X_test, y_test, ax = lda_disp.ax_)
        ada_disp = metrics.precision_recall_curve(ada, X_test, y_test, ax = lda_disp.ax_)
        knn_disp = metrics.precision_recall_curve(knn, X_test, y_test, ax = lda_disp.ax_)
        #knn_disp.figure_.suptitle(f"ROC curve comparison {image_tag}-{reg_type}")

    # plt.show()
    
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
            
            if reg_type == 'align':
                global_cost_vector_bigdata_FLAIRtoT1, local_cost_vector_bigdata_FLAIRtoT1 = get_coreg_cost_vectors(cost, subpath1, 'hrFLAIR') # FLAIR brain to T1 brain (only align)
            
            combine_cost_vector_T1.append(local_cost_vector_bigdata_T1) # T1 data
            combine_cost_vector_FLAIR.append(local_cost_vector_bigdata_FLAIR) # FLAIR data
            
            # HCP-YA
            global_cost_vector_hcpya_T1, local_cost_vector_hcpya_T1 = get_cost_vectors(cost, reg_type, subpath2, 'hrT1') # T1 to MNI
            global_cost_vector_hcpya_T2, local_cost_vector_hcpya_T2 = get_cost_vectors(cost, reg_type, subpath2, 'hrT2') # T2 to MNI
            
            if reg_type == 'align':
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
            
            # Computing cut-off point and AUC for normal and test
            print('doing for Big Data\n')
            
            compute_cutoff_auc(global_cost_vector_bigdata_T1, global_test_cost_vector_bigdata_T1, cost, reg_type, 'hrT1', 'global') # T1 to MNI
            compute_cutoff_auc(local_cost_vector_bigdata_T1, local_test_cost_vector_bigdata_T1, cost, reg_type, 'hrT1', 'local') # T1 to MNI
            print('----------------------------------------------------------------------------------------------------')
            
            #compute_cutoff_auc(global_cost_vector_bigdata_FLAIR, global_test_cost_vector_bigdata_FLAIR, cost, reg_type, 'hrFLAIR', 'global') # FLAIR to MNI
            if False:
                compute_cutoff_auc(local_cost_vector_bigdata_FLAIR, local_test_cost_vector_bigdata_FLAIR, cost, reg_type, 'hrFLAIR', 'local') # FLAIR to MNI
                print('----------------------------------------------------------------------------------------------------')
            
            if False:
                compute_cutoff_auc(global_cost_vector_bigdata_FLAIRtoT1, global_test_cost_vector_bigdata_FLAIRtoT1, cost, 'T1', 'hrFLAIR', 'global') # FLAIR brain to T1 brain
                compute_cutoff_auc(local_cost_vector_bigdata_FLAIRtoT1, local_test_cost_vector_bigdata_FLAIRtoT1, cost, 'T1', 'hrFLAIR', 'local') # FLAIR brain to T1 brain
                print('----------------------------------------------------------------------------------------------------')
            
            print('doing for HCP-YA\n')
            
            compute_cutoff_auc(global_cost_vector_hcpya_T1, global_test_cost_vector_hcpya_T1, cost, reg_type, 'hrT1', 'global') # T1 to MNI
            compute_cutoff_auc(local_cost_vector_hcpya_T1, local_test_cost_vector_hcpya_T1, cost, reg_type, 'hrT1', 'local') # T1 to MNI
            print('----------------------------------------------------------------------------------------------------')
            
            #compute_cutoff_auc(global_cost_vector_hcpya_T2, global_test_cost_vector_hcpya_T2, cost, reg_type, 'hrT2', 'global') # T2 to MNI
            if False:
                compute_cutoff_auc(local_cost_vector_hcpya_T2, local_test_cost_vector_hcpya_T2, cost, reg_type, 'hrT2', 'local') # T2 to MNI
                print('----------------------------------------------------------------------------------------------------')
            
            if False:
                compute_cutoff_auc(global_cost_vector_hcpya_T2toT1, global_test_cost_vector_hcpya_T2toT1, cost, 'T1', 'hrT2', 'global') # T2 brain to T1 brain
                compute_cutoff_auc(local_cost_vector_hcpya_T2toT1, local_test_cost_vector_hcpya_T2toT1, cost, 'T1', 'hrT2', 'local') # T2 brain to T1 brain
        
        # calling combinational_cost method to design a classifier 
        combinational_cost(combine_cost_vector_T1, combine_test_cost_vector_T1, reg_type, 'T1', 5) 
        #combinational_cost(combine_cost_vector_T2, combine_test_cost_vector_T2, reg_type, 'T2', 5)
        #combinational_cost(combine_cost_vector_FLAIR, combine_test_cost_vector_FLAIR, reg_type, 'FLAIR', 5)
        
        
        
        
        
