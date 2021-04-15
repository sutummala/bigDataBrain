# code created by Sudhakar on May 2020 and modified on August 2020 to add different ML classifiers
# check registration

import os
import all_plots as ap
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, GridSearchCV, cross_val_score, train_test_split
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler, PowerTransformer, QuantileTransformer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier as kNN
import matplotlib.pyplot as plt
import PipelineProfiler
import pickle
from mpl_toolkits.mplot3d import Axes3D
#from autosklearn.classification import AutoSklearnClassifier

plt.rcParams.update({'font.size': 22})

subpath1 = '/home/tummala/data/HCP-100'
#subpath2 = '/media/tummala/TUMMALA/Work/Data/ABIDE-failed'
#subpath2 = '/media/tummala/TUMMALA/Work/Data/ABIDE-validate'
#subpath2 = '/home/tummala/data/HCP-100re'
subpath2 = '/media/tummala/TUMMALA/Work/Data/IXI-Re'

voi_size = 7
step_size = 7 # stride
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
        cost name, ncc: normalized correlation coefficient, nmi: normalized mutual information.
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
        if np.any(np.isnan(local_cost_vector)):
            print(f'NaN test cost values are found for {subject}')       
    return remove_nan(global_cost_vector), remove_nan(local_cost_vector)

def get_cost_vectors(cost_func, reg_type, subpath, tag):
    '''
    Parameters
    ----------
    cost_func : str
        cost name, ncc: normalized correlation coefficient, nmi: normalized mutual information.
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
    #print(f'no.of subjects {len(subjects)}')
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
                    if np.isnan(cost_data[1]):
                        print('{cost_func} cost value is nan for {subject}')
        if np.any(np.isnan(local_cost_vector)):
            print(f'NaN cost values are found for {subject}')
    #print(f'length of local cost vector {len(local_cost_vector)}')
    return remove_nan(global_cost_vector), remove_nan(local_cost_vector)

def get_coreg_test_cost_vectors(cost_func, subpath, tag):
    '''
    Parameters
    ----------
    cost_func : str
        cost name, ncc: normalized correlation coefficient, nmi: normalized mutual information.
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
            if np.any(np.isnan(local_cost_vector)):
                print(f'NaN cost values are found for {subject}')        
    return remove_nan(global_cost_vector), remove_nan(local_cost_vector)

def get_test_cost_vectors(cost_func, reg_type, subpath, tag):
    '''
    Parameters
    ----------
    cost_func : str
        cost name, ncc: normalized correlation coefficient, nmi: normalized mutual information.
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
                    if np.isnan(cost_data[1]):
                        print('{cost_func} cost value is nan for {subject}')
                    #print(f'{subject}{cost_data[1]}')
            if np.any(np.isnan(local_cost_vector)):
                print(f'NaN test cost values are found for {subject}')                
    return remove_nan(global_cost_vector), remove_nan(local_cost_vector)

def compute_cutoff_auc(data1, data2, *tags):
    '''computes cut-off point and AUC for given cost and reg type from data1 (normal values), data2 (test values)'''
    
    labels_predict = []
    
    balance_groups = 0
    
    if balance_groups:
        data2 = data2[:np.shape(data1)[0]] # To make sizes of both classes same
    labels = np.concatenate([np.zeros(len(data1)), np.ones(len(data2))])
    data = np.concatenate([data1, data2]).reshape(-1, 1)
    sc = MinMaxScaler()
    data = sc.fit_transform(data)
    data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size = 0.25, stratify = labels, shuffle = True)
    #print(f'{len(data1)}, {len(data2)}')
    fpr, tpr, thresholds = metrics.roc_curve(labels_train, data_train, pos_label = 0)
    threshold = thresholds[np.argmax(tpr-fpr)]
    
    #labels_predict = data_test[data_test > threshold] # predicting the labels on the testset based on the threshold on the training set
    
    for j in range(len(labels_test)):
        if data_test[j] > threshold:
            labels_predict.append(0)
        else:
            labels_predict.append(1)
                
    print(f'F1-score and Balanced Accuracy on test-set for {tags[2]}-{tags[1]}-{tags[3]}-{tags[0]} are {metrics.f1_score(labels_test, labels_predict)}, {metrics.balanced_accuracy_score(labels_test, labels_predict)}')
    #print(f'Threshold for {tags[2]}-{tags[1]}-{tags[3]}-{tags[0]} is: {thresholds[np.argmax(tpr-fpr)]}, sensitivity (recall) is: {tpr[np.argmax(tpr-fpr)]}, specificity is: {1-fpr[np.argmax(tpr-fpr)]}, fall-out is: {fpr[np.argmax(tpr-fpr)]}, AUC is: {metrics.auc(fpr, tpr)}\n')
    
def classifier_accuracy(model, X_train, X_test, y_train, y_test):
    'get model (classifier) accuracy based on training and testing'
    
    model.fit(X_train, y_train)
    #print(f'F1-score is {metrics.f1_score(y_test, model.predict(X_test))}/n')
    #return model.score(X_test, y_test), metrics.roc_auc_score(y_test, model.predict_proba(X_test)[:,1]), model
    #precision, recall, thresholds = metrics.precision_recall_curve(y_test, model.predict_proba(X_test)[:,1])
    #print(f'area under PR curve is {metrics.auc(recall, precision)}')
    #print(X_test)
    #print(y_test-model.predict(X_test))
    tn, fp, fn, tp = metrics.confusion_matrix(y_test, model.predict(X_test)).ravel()
    specificity = tn/(tn+fp)
    
    return metrics.balanced_accuracy_score(y_test, model.predict(X_test)), metrics.roc_auc_score(y_test, model.predict_proba(X_test)[:,1]), model      

def visualize_cost_values(data, labels, standardize):
    'function to visualize the features in 3D'
    
    fig = plt.figure()
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev = 48, azim = 134)
    #ax = fig.add_subplot(111, projection='3d')
    if standardize:
        data = StandardScaler().fit_transform(data)
            
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c = labels.astype(float), edgecolor = 'k')
    #ax.scatter(X_misaligned[:, 0], X_misaligned[:, 1], X_misaligned[:, 2], c='g', marker='+')
        
    #ax.text3D(X[y == 1, 0].mean(), X[y == 1, 1].mean(), X[y == 1, 2].mean(), 'misaligned', horizontalalignment='center', bbox=dict(alpha=.2, edgecolor='w', facecolor='w'))
        
    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    ax.set_xlabel('NCC')
    ax.set_ylabel('NMI')
    ax.set_zlabel('CR')
    ax.dist = 12
    plt.show()

def combinational_cost(data1, data2, data3, data4, reg_type, image_tag, no_of_folds, number_rep):
    '''
    Parameters
    ----------
    data1 : arrays
        matrix of all costs of group1 (normal) for training/testing. Each individual cost (feature) should be arranged as a column
    data2 : arrays
        matrix of all costs of group2 (abnormal) for training/testing. Each individual cost (feature) should be arranged as a column
    data3 : arrays
        matrix of all costs of group1 (normal) for validation. Each individual cost (feature) should be arranged as a column
    data4 : arrays
        matrix of all costs of group2 (abnormal) for validation. Each individual cost (feature) should be arranged as a column
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
    print(f'number of correctly aligned images for cross-validation are {len(x_normal_label)}')
    
    balance_data = 0 # Since the generated misaligned images are 5 times higher, the two classes can be balanced by considering this flag variable one
    
    # transposing and creating labels for data2
    if balance_data:    
        X_misaligned = np.transpose(data2)[:np.shape(X_normal)[0], :]
    else:
        X_misaligned = np.transpose(data2)    
    x_misaligned_label = np.ones(len(X_misaligned))
    print(f'number of misaligned images for cross-validation are {len(x_misaligned_label)}')
    
    # data for validation, combining data3 and data4 
    X_normal_val = np.transpose(data3)
    x_normal_val_label = np.zeros(len(X_normal_val))
    print(f'number of images for testing are {len(x_normal_val_label)}')
    
    if balance_data:  
        X_misaligned_val = np.transpose(data4)[:np.shape(X_normal_val)[0], :]
    else:
        X_misaligned_val = np.transpose(data4)
    x_misaligned_val_label = np.ones(len(X_misaligned_val))
    #print(f'number of misaligned images for validation are {len(x_misaligned_val_label)}')
    
    #X_val = np.concatenate((X_normal_val, X_misaligned_val))
    #y_val = np.concatenate((x_normal_val_label, x_misaligned_val_label))
    X_val = X_normal_val
    y_val = x_normal_val_label
    
    # combining data1 and data2 and the corresponding labels    
    X = np.concatenate((X_normal, X_misaligned))
    y = np.concatenate((x_normal_label, x_misaligned_label))
    
    visualize_costs = 0 # This will do a 3D plot to visualize the costs
    
    if visualize_costs:
        visualize_cost_values(X, y, standardize = 1)
        
    # print('cost values (min, mean, max) before scaling')
    # for a in range(3):
    #     print(np.min(X[:, a]), np.mean(X[:, a]), np.max(X[:, a]), np.min(X_val[:, a]), np.mean(X_val[:, a]), np.max(X_val[:, a]))
        
    # scaling the costs (features) to make sure the ranges of individual features are same to avoid the effect of features that have relatively large values. It may not be necessary in this case as all these 3 costs lie between 0 and 1  
    #scale = QuantileTransformer(n_quantiles = 10, output_distribution = 'uniform') # Subtracting the mean and dividing with standard deviation
    scale = StandardScaler()
    scale_test = StandardScaler()
    
    # scale.fit(X)
    # X = scale.transform(X)
    
    #X_val = scale_test.fit_transform(X_val) # fit_transform is necessary here instead of just transform
    
    # print('cost values (min, mean, max) after scaling')
    # X1 = StandardScaler().fit_transform(X) # Making a copy for standardization
    # for a in range(3):
    #     print(np.min(X1[:, a]), np.median(X1[:, a]), np.max(X1[:, a]), np.min(X_val[:, a]), np.median(X_val[:, a]), np.max(X_val[:, a]))
    # X = np.concatenate((X, X_val))
    # y = np.concatenate((y, y_val))
    
    unsupervised_learning = 0 # reated models without giving labels, 
    
    if unsupervised_learning:
        kmeans = KMeans(n_clusters = 2, random_state = 0).fit(X)
        visualize_cost_values(X, 1-kmeans.labels_, standardize = 1)
        print(f'balanced accuracy using KMeans Clustering algorithm is {metrics.balanced_accuracy_score(y, 1-kmeans.labels_)}')
        print(f'tested BA score using KMeans Clustering algorithm is {metrics.balanced_accuracy_score(y_val, 1-kmeans.predict(X_val))}')
        
        agglo = AgglomerativeClustering(n_clusters = 2).fit(X)
        visualize_cost_values(X, 1-agglo.labels_, standardize  = 1)
        print(f'balanced accuracy using Agglomerative Clustering algorithm is {metrics.balanced_accuracy_score(y, 1-agglo.labels_)}')
        #print(f'tested BA score using Agglomerative Clustering algorithm is {metrics.balanced_accuracy_score(y_val, 1-agglo.predict(X_val))}')
    
    # doing Grid Search CV for hyper-parameter tuning
    
    #X_gridCV = StandardScaler().fit_transform(X)
    X_gridCV = X
    
    print('doing hyperparameter tuning of the ML models using Grid Search Cross-Validation')
    
    lda_parameters = {'solver': ('svd', 'lsqr', 'eigen'), 'tol': [0.001, 0.0001, 0.00001]}
    gridCV_lda = GridSearchCV(LDA(n_components = 1), lda_parameters, refit = True, scoring = ('balanced_accuracy'), cv = 5).fit(X_gridCV, y)
    print(f'F1-score for LDA {gridCV_lda.best_params_} is {gridCV_lda.best_score_}')
    
    rfc_parameters = {'n_estimators': [1, 10, 20, 40, 60, 80, 100], 'criterion': ('gini', 'entropy'), 'max_depth': [2, 4, 6, 8, 10], 'max_features':('auto', 'sqrt', 'log2')}
    gridCV_rfc = GridSearchCV(RandomForestClassifier(), rfc_parameters, refit = True, scoring = ('balanced_accuracy'), cv = 5).fit(X_gridCV, y)
    print(f'F1-score for Random Forest {gridCV_rfc.best_params_} is {gridCV_rfc.best_score_}')
    
    svc_parameters = {'kernel':('linear', 'poly', 'rbf', 'sigmoid'), 'gamma':[1e-3, 1e-4], 'C': [1, 10, 100, 1000]}
    gridCV_svc = GridSearchCV(SVC(), svc_parameters, refit = True, scoring = ('balanced_accuracy'), cv = 5).fit(X_gridCV, y)
    print(f'F1-score for Support Vector Machine {gridCV_svc.best_params_} is {gridCV_svc.best_score_}')
    
    knn_parameters = {'n_neighbors':[3, 7, 11, 15, 19, 21], 'weights': ('uniform', 'distance'), 'algorithm':('auto', 'ball_tree', 'kd_tree', 'brute')}
    gridCV_knn = GridSearchCV(kNN(), knn_parameters, refit = True, scoring = ('balanced_accuracy'), cv = 5).fit(X_gridCV, y)
    print(f'F1-score for K Nearest Neighbors {gridCV_knn.best_params_} is {gridCV_knn.best_score_}')
    
    ada_parameters = {'n_estimators':[1, 10, 20, 40, 60, 80, 100], 'learning_rate': [0.8, 0.9, 1]}
    gridCV_ada = GridSearchCV(AdaBoostClassifier(), ada_parameters, refit = True, scoring = ('balanced_accuracy'), cv = 5).fit(X_gridCV, y)
    print(f'F1-score for Adaptive Boosting {gridCV_ada.best_params_} is {gridCV_ada.best_score_}')
    
    # Repeated K-fold cross validation, n_splits specifies the number of folds, n_repeats specifies the no.of repetetions
    folds = RepeatedStratifiedKFold(n_splits = no_of_folds, n_repeats = number_rep)
    
    scores_lda = []
    scores_qda = [] 
    scores_rfc = []
    scores_svm = []
    scores_gnb = []
    scores_knn = []
    scores_lor = []
    scores_ada = []
    scores_gra = []
    scores_automl = []
    
    auc_lda = []
    auc_qda = [] 
    auc_rfc = []
    auc_svm = []
    auc_gnb = []
    auc_knn = []
    auc_lor = []
    auc_ada = []
    auc_gra = []
    auc_automl = []
    
    
    for train_index, test_index in folds.split(X, y):
        
        X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
        
        # X_train = scale.fit_transform(X_train) # scaling is implemented on X_train and the transformation is implemented on the X_test
        # X_test = scale.transform(X_test)
        
        #X_train, X_test, y_train, y_test = train_test_split(X_1, y_1, test_size = 0.20, stratify = y_1, shuffle = True)
    
        # 1. Linear Discriminant Analysis Classifier
        lda = LDA(solver = 'eigen', shrinkage = 'auto', n_components = 1)
        score_lda, roc_auc_lda, model_lda = classifier_accuracy(lda, X_train, X_test, y_train, y_test)
        # print('F1-score for LDA', {metrics.f1_score(y_val, LDA(solver = 'eigen', shrinkage = 'auto', n_components = 1).fit(X, y).predict(X_val))})
        scores_lda.append(score_lda) # F1-score
        auc_lda.append(roc_auc_lda) # AUC
        #auc_lda.append(metrics.f1_score(y_validation, LDA(solver = 'eigen', shrinkage = 'auto', n_components = 1).fit(X_1, y_1).predict(X_validation))) # AUC
        
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
        svc = SVC(kernel = 'rbf', gamma = 'scale', probability = True)
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
        
        # # 7a. Gradient Boosting Classifier
        # gra = GradientBoostingClassifier(random_state = 0)
        # score_gra, roc_auc_gra, model_gra = classifier_accuracy(gra, X_train, X_test, y_train, y_test)
        # scores_gra.append(score_gra)
        # auc_gra.append(roc_auc_gra)
        
        # 8. Auto Sklearn Classifier (for automatic model selection with hyperparameter tuning)
        # automl = AutoSklearnClassifier(time_left_for_this_task = 50, per_run_time_limit = 10)
        # score_automl, roc_auc_automl, model_automl = classifier_accuracy(automl, X_train, X_test, y_train, y_test)
        # scores_automl.append(score_automl)
        # auc_automl.append(roc_auc_automl)
        
    # Note: 'cross_val_score' method from sklearn could be used directly on the classifier model to avoid the above for loop. Further, f1-score could be used instead of accuracy metric if number of positive samples (mis-aligned) are low.
    if False:
        print(f'accuracy using LDA classifier for {image_tag}-{reg_type} is: {np.average(scores_lda)}, AUC is: {np.average(auc_lda)}')
        #print(f'accuracy using QDA classifier for {image_tag}-{reg_type} is: {np.average(scores_qda)}, AUC is: {np.average(auc_qda)}\n')
        print(f'accuracy using RandomForest classifier for {image_tag}-{reg_type} is: {np.average(scores_rfc)}, AUC is: {np.average(auc_rfc)}')
        print(f'accuracy using SVM classifier for {image_tag}-{reg_type} is: {np.average(scores_svm)}, AUC is: {np.average(auc_svm)}')
        print(f'accuracy using Naive Bayes classifier for {image_tag}-{reg_type} is: {np.average(scores_gnb)}, AUC is: {np.average(auc_gnb)}')
        print(f'accuracy using kNN classifier for {image_tag}-{reg_type} is: {np.average(scores_knn)}, AUC is: {np.average(auc_knn)}')
        #print(f'accuracy using Logistic Regression classifier for {image_tag}-{reg_type} is: {np.average(scores_lor)}, AUC is: {np.average(auc_lor)}\n')
        print(f'accuracy using Ada Boost classifier for {image_tag}-{reg_type} is: {np.average(scores_ada)}, AUC is: {np.average(auc_ada)}')
        #print(f'accuracy using Gradient boosting classifier for {image_tag}-{reg_type} is: {np.average(scores_gra)}, AUC is: {np.average(auc_gra)}\n')
        #print(f'accuracy using AutoML Classifier for {image_tag}-{reg_type} is: {np.average(scores_automl)}, AUC is: {np.average(auc_automl)}\n')
        
    save_model = '/home/tummala/mri/ml_classifier_models_checking_reg'
    
    if not os.path.exists(save_model):
        os.makedirs(save_model)
    
    # saving the trained model, e.g. shown for saving ada boost classifier model and minmax scaling model
    #scale_all = StandardScaler().fit(X)
    #X = scale_all.transform(X)
    
    #pickle.dump(scale_all, open(save_model+'/'+'scale_'+reg_type+image_tag, 'wb'))
    pickle.dump(gridCV_lda, open(save_model+'/'+'lda_'+reg_type+image_tag, 'wb'))
    pickle.dump(gridCV_rfc, open(save_model+'/'+'rfc_'+reg_type+image_tag, 'wb'))
    pickle.dump(gridCV_svc, open(save_model+'/'+'svm_'+reg_type+image_tag, 'wb'))
    pickle.dump(GaussianNB().fit(X, y), open(save_model+'/'+'gnb_'+reg_type+image_tag, 'wb'))
    pickle.dump(gridCV_knn, open(save_model+'/'+'knn_'+reg_type+image_tag, 'wb'))
    pickle.dump(gridCV_ada, open(save_model+'/'+'ada_boost_'+reg_type+image_tag, 'wb'))
    
    # automl_model = AutoSklearnClassifier(time_left_for_this_task = 50, per_run_time_limit = 10).fit(X, y)
    # pickle.dump(automl_model, open(save_model+'/'+'automl_'+reg_type+image_tag, 'wb'))
    # pickle.load method could be used to load the model for later use and predict method of the seved model to categorize new cases
    
    # plotting ROC curve for Sensitivity/Specificity all above classifiers
    subjects_test = os.listdir(subpath2)
    
    for index, subject in enumerate(subjects_test, start=1):
        global_cost_vector = []
        local_cost_vector = []
        cost_folder = subpath2+'/'+subject+'/cost'+str(voi_size)+str(step_size)
        #print('{}-{}, {}-{}'.format(index, subject, reg_type, cost_func))
        data_files = os.listdir(cost_folder)
        for data_file in data_files:
            if reg_type in data_file and (image_tag in data_file):
                if not 'alignedToT1' in data_file:
                    cost_data = np.loadtxt(cost_folder+'/'+data_file)
                    global_cost_vector.append(cost_data[0])
                    local_cost_vector.append(cost_data[1])
        
        if not local_cost_vector:
            print(f'cost vector is empty for {subject}')
            continue
        sample = np.reshape(np.array(local_cost_vector), (1,3))
        reg_quality = gridCV_rfc.predict_proba(sample)[0][0]*100
        
        if reg_quality < 50:
            print(f'Quality of {reg_type} registration for {subject} using RFC is {reg_quality}')
        
    
    if False:
        lda_disp = metrics.plot_roc_curve(gridCV_lda, X_val, y_val, drop_intermediate = False)
        print('F1-score for LDA', {metrics.accuracy_score(y_val, gridCV_lda.predict(X_val))})
        #qda_disp = metrics.plot_roc_curve(qda, X_test, y_test, ax = lda_disp.ax_)
        svm_disp = metrics.plot_roc_curve(gridCV_svc, X_val, y_val, ax = lda_disp.ax_)
        print('F1-score for SVM', {metrics.accuracy_score(y_val, gridCV_svc.predict(X_val))})
        #nsvm_disp = metrics.plot_roc_curve(nsvm, X_test, y_test, ax = lda_disp.ax_)
        gnb_disp = metrics.plot_roc_curve(GaussianNB().fit(X, y), X_val, y_val, ax = lda_disp.ax_)
        print('F1-score for GNB', {metrics.accuracy_score(y_val, GaussianNB().fit(X, y).predict(X_val))})
        knn_disp = metrics.plot_roc_curve(gridCV_knn, X_val, y_val, ax = lda_disp.ax_)
        print('F1-score for kNN', {metrics.accuracy_score(y_val, gridCV_knn.predict(X_val))})
        rfc_disp = metrics.plot_roc_curve(gridCV_rfc, X_val, y_val, ax = lda_disp.ax_)
        print('F1-score for RFC', {metrics.accuracy_score(y_val, gridCV_rfc.predict(X_val))})
        #print(y_val)
        #print(RandomForestClassifier(criterion = 'gini', n_estimators = 100).fit(X, y).predict_proba(X_val)[:, 1])
        metrics.plot_confusion_matrix(gridCV_rfc, X_val, y_val, colorbar = False)
        plt.show()
        ada_disp = metrics.plot_roc_curve(gridCV_ada, X_val, y_val, ax = lda_disp.ax_)
        print('F1-score for Ada Boost', {metrics.accuracy_score(y_val, gridCV_ada.predict(X_val))})
        # automl_disp = metrics.plot_roc_curve(automl_model, X_val, y_val, ax = lda_disp.ax_)
        # print('F1-score for AutoML Classifier', {metrics.balanced_accuracy_score(y_val, automl_model.predict(X_val))})
        # print(automl_model.sprint_statistics())
        
        # # Plotting the sklearn models using PipelineProfiler 
        # profiler_data = PipelineProfiler.import_autosklearn(automl_model)
        # PipelineProfiler.plot_pipeline_matrix(profiler_data)
        # plt.show()
        
        #knn_disp.figure_.suptitle(f"ROC curve comparison {image_tag}-{reg_type}")
        
    # plotting Precision-Recall ROC curve for all above classifiers
    if False:
        lda_disp = metrics.plot_precision_recall_curve(LDA(solver = 'eigen', shrinkage = 'auto', n_components = 1).fit(X, y), X_val, y_val)
        #qda_disp = metrics.precision_recall_curve(qda, X_test, y_test, ax = lda_disp.ax_)
        svm_disp = metrics.plot_precision_recall_curve(SVC(kernel = 'linear', gamma = 'scale', probability = True).fit(X, y), X_val, y_val, ax = lda_disp.ax_)
        #nsvm_disp = metrics.plot_roc_curve(nsvm, X_test, y_test, ax = lda_disp.ax_)
        gnb_disp = metrics.plot_precision_recall_curve(GaussianNB().fit(X, y), X_val, y_val, ax = lda_disp.ax_)
        knn_disp = metrics.plot_precision_recall_curve(kNN(n_neighbors = 15).fit(X, y), X_val, y_val, ax = lda_disp.ax_)
        rfc_disp = metrics.plot_precision_recall_curve(RandomForestClassifier(criterion = 'gini', n_estimators = 100).fit(X, y), X_val, y_val, ax = lda_disp.ax_)
        ada_disp = metrics.plot_precision_recall_curve(AdaBoostClassifier(n_estimators = 100).fit(X, y), X_val, y_val, ax = lda_disp.ax_)
        #knn_disp.figure_.suptitle(f"ROC curve comparison {image_tag}-{reg_type}")

    # plt.show()
    
    # # confusion matrix and calculating the accuracy
    # cm = metrics.confusion_matrix(y_test, y_pred)
    # print(cm)
    # print('Accuracy' + str(metrics.accuracy_score(y_test, y_pred)))    
        
if __name__ == '__main__':
    
    costs = ['ncc', 'nmi', 'cor']
    reg_types = ['align', 'mni']
    
    local = 1 # to choose local (1) or global (0) costs

    for reg_type in reg_types:
        combine_cost_vector_T1 = []
        combine_test_cost_vector_T1 = []
        
        combine_cost_vector_T1_val = []
        combine_test_cost_vector_T1_val = []
        
        combine_cost_vector_T2 = []
        combine_test_cost_vector_T2 = []
        
        combine_cost_vector_T2_val = []
        combine_test_cost_vector_T2_val = []
        
        combine_cost_vector_FLAIR = []
        combine_test_cost_vector_FLAIR = []
        
        for cost in costs:
            # getting normal values for hrT1, hrT2 and hrFLAIR for bigdata 
            global_cost_vector_bigdata_T1, local_cost_vector_bigdata_T1 = get_cost_vectors(cost, reg_type, subpath1, 'hrT1') # T1 to MNI
            global_cost_vector_bigdata_T2, local_cost_vector_bigdata_T2 = get_cost_vectors(cost, reg_type, subpath1, 'hrT2') # FLAIR to MNI
            
            if reg_type == 'align':
                global_cost_vector_bigdata_FLAIRtoT1, local_cost_vector_bigdata_FLAIRtoT1 = get_coreg_cost_vectors(cost, subpath1, 'hrFLAIR') # FLAIR brain to T1 brain (only align)
            
            if local:
                combine_cost_vector_T1.append(local_cost_vector_bigdata_T1) # T1 data
                combine_cost_vector_T2.append(local_cost_vector_bigdata_T2) # T2 data
            else:
                combine_cost_vector_T1.append(global_cost_vector_bigdata_T1) # T1 data
                combine_cost_vector_T2.append(global_cost_vector_bigdata_T2) # T2 data
            
            # HCP-YA
            global_cost_vector_hcpya_T1, local_cost_vector_hcpya_T1 = get_cost_vectors(cost, reg_type, subpath2, 'hrT1') # T1 to MNI
            global_cost_vector_hcpya_T2, local_cost_vector_hcpya_T2 = get_cost_vectors(cost, reg_type, subpath2, 'hrT2') # T2 to MNI
            
            if local:
                combine_cost_vector_T1_val.append(local_cost_vector_hcpya_T1)
                combine_cost_vector_T2_val.append(local_cost_vector_hcpya_T2)
            else:
                combine_cost_vector_T1_val.append(global_cost_vector_hcpya_T1)
                combine_cost_vector_T2_val.append(global_cost_vector_hcpya_T2)
            
            if reg_type == 'align':
                global_cost_vector_hcpya_T2toT1, local_cost_vector_hcpya_T2toT1 = get_coreg_cost_vectors(cost, subpath2, 'hrT2') # T2 brain to T1 brain (only align)
    
            if False:
                # plotting normal values for T1, T2 and FLAIR
                ap.plot_cost([global_cost_vector_bigdata_T1, global_cost_vector_hcpya_T1, global_cost_vector_hcpya_T2, global_cost_vector_bigdata_T2], cost,
                          ['T1', 'T1(hcp)', 'T2(hcp)', 'FLAIR'], f'global-{reg_type}') # plotting global cost
            if False:
                ap.plot_cost([local_cost_vector_bigdata_T1, local_cost_vector_hcpya_T1, local_cost_vector_hcpya_T2, local_cost_vector_bigdata_T2], cost,
                          ['T1', 'T1(hcp)', 'T2(hcp)', 'FLAIR'], f'local-{reg_type}') # plotting local cost
            
            # getting test values for hrT1, hrT2 and hrFLAIR for bigdata 
            global_test_cost_vector_bigdata_T1, local_test_cost_vector_bigdata_T1 = get_test_cost_vectors(cost, reg_type, subpath1, 'hrT1') # T1 to MNI
            global_test_cost_vector_bigdata_FLAIR, local_test_cost_vector_bigdata_FLAIR = get_test_cost_vectors(cost, reg_type, subpath1, 'hrT2') # FLAIR to MNI
            global_test_cost_vector_bigdata_FLAIRtoT1, local_test_cost_vector_bigdata_FLAIRtoT1 = get_coreg_test_cost_vectors(cost, subpath1, 'hrFLAIR') # FLAIR brain to T1 brain (only align)
            
            if local:
                combine_test_cost_vector_T1.append(local_test_cost_vector_bigdata_T1) # T1 test data
                combine_test_cost_vector_T2.append(local_test_cost_vector_bigdata_FLAIR) # FLAIR test data
            else:
                combine_test_cost_vector_T1.append(global_test_cost_vector_bigdata_T1) # T1 test data
                combine_test_cost_vector_T2.append(global_test_cost_vector_bigdata_FLAIR) # FLAIR test data
            
            # HCPYA
            global_test_cost_vector_hcpya_T1, local_test_cost_vector_hcpya_T1 = get_test_cost_vectors(cost, reg_type, subpath2, 'hrT1') # T1 to MNI
            global_test_cost_vector_hcpya_T2, local_test_cost_vector_hcpya_T2 = get_test_cost_vectors(cost, reg_type, subpath2, 'hrT2') # T2 to MNI
            global_test_cost_vector_hcpya_T2toT1, local_test_cost_vector_hcpya_T2toT1 = get_coreg_test_cost_vectors(cost, subpath2, 'hrT2') # T2 to T1 (only align)
            
            if local:
                combine_test_cost_vector_T1_val.append(local_test_cost_vector_hcpya_T1) # T1 validate data
                combine_test_cost_vector_T2_val.append(local_test_cost_vector_hcpya_T2) # T2 test data
            else:
                combine_test_cost_vector_T1_val.append(global_test_cost_vector_hcpya_T1) # T1 validate data
                combine_test_cost_vector_T2_val.append(global_test_cost_vector_hcpya_T2) # T2 test data
            
            if False:
                ap.plot_cost([local_cost_vector_bigdata_T1, local_test_cost_vector_bigdata_T1], cost,
                          ['T1', 'T1-test', 'T1(local)', 'T1-test(local)'], f'Big-Data {reg_type}') # plotting local cost for bigdata T1
            if False:
                ap.plot_cost([local_cost_vector_bigdata_T2, local_test_cost_vector_bigdata_FLAIR], cost,
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
            #print('doing for Big Data\n')
            
            #compute_cutoff_auc(global_cost_vector_bigdata_T1, global_test_cost_vector_bigdata_T1, cost, reg_type, 'hrT1', 'global') # T1 to MNI
            #compute_cutoff_auc(local_cost_vector_bigdata_T1, local_test_cost_vector_bigdata_T1, cost, reg_type, 'hrT1', 'local') # T1 to MNI
            #print('----------------------------------------------------------------------------------------------------')
            
            #compute_cutoff_auc(global_cost_vector_bigdata_FLAIR, global_test_cost_vector_bigdata_FLAIR, cost, reg_type, 'hrFLAIR', 'global') # FLAIR to MNI
            if False:
                compute_cutoff_auc(local_cost_vector_bigdata_T2, local_test_cost_vector_bigdata_FLAIR, cost, reg_type, 'hrFLAIR', 'local') # FLAIR to MNI
                print('----------------------------------------------------------------------------------------------------')
            
            if False:
                compute_cutoff_auc(global_cost_vector_bigdata_FLAIRtoT1, global_test_cost_vector_bigdata_FLAIRtoT1, cost, 'T1', 'hrFLAIR', 'global') # FLAIR brain to T1 brain
                compute_cutoff_auc(local_cost_vector_bigdata_FLAIRtoT1, local_test_cost_vector_bigdata_FLAIRtoT1, cost, 'T1', 'hrFLAIR', 'local') # FLAIR brain to T1 brain
                print('----------------------------------------------------------------------------------------------------')
            
            #print('doing for HCP-YA\n')
            
            #compute_cutoff_auc(global_cost_vector_hcpya_T1, global_test_cost_vector_hcpya_T1, cost, reg_type, 'hrT1', 'global') # T1 to MNI
            #compute_cutoff_auc(local_cost_vector_hcpya_T1, local_test_cost_vector_hcpya_T1, cost, reg_type, 'hrT1', 'local') # T1 to MNI
            #print('----------------------------------------------------------------------------------------------------')
            
            #compute_cutoff_auc(global_cost_vector_hcpya_T2, global_test_cost_vector_hcpya_T2, cost, reg_type, 'hrT2', 'global') # T2 to MNI
            if False:
                compute_cutoff_auc(local_cost_vector_hcpya_T2, local_test_cost_vector_hcpya_T2, cost, reg_type, 'hrT2', 'local') # T2 to MNI
                print('----------------------------------------------------------------------------------------------------')
            
            if False:
                compute_cutoff_auc(global_cost_vector_hcpya_T2toT1, global_test_cost_vector_hcpya_T2toT1, cost, 'T1', 'hrT2', 'global') # T2 brain to T1 brain
                compute_cutoff_auc(local_cost_vector_hcpya_T2toT1, local_test_cost_vector_hcpya_T2toT1, cost, 'T1', 'hrT2', 'local') # T2 brain to T1 brain
        
        # calling combinational_cost method to design a classifier 
        combinational_cost(combine_cost_vector_T1, combine_test_cost_vector_T1, combine_cost_vector_T1_val, combine_test_cost_vector_T1_val, reg_type, 'T1', 5, 1)
        combinational_cost(combine_cost_vector_T2, combine_test_cost_vector_T2, combine_cost_vector_T2_val, combine_test_cost_vector_T2_val, reg_type, 'T2', 5, 1) 
        #combinational_cost(combine_cost_vector_T2, combine_test_cost_vector_T2, reg_type, 'T2', 5)
        #combinational_cost(combine_cost_vector_FLAIR, combine_test_cost_vector_FLAIR, reg_type, 'FLAIR', 5)
        
        
        
        
        
