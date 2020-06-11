# code created by Sudhakar on May 2020
# computing cost functions for checking registration

import numpy as np
import scipy.stats

# 1. Sum of squared differences(SSD)
def ssd(refimage, movingimage):
    
    if refimage.shape != movingimage.shape:
        print('images shape mismatch')
    else:
        return np.sum((np.ndarray.flatten(refimage)-np.ndarray.flatten(movingimage))**2)
    
# 2. Cross Correlation 
def cc(refimage, movingimage):
    
    if refimage.shape != movingimage.shape:
        print('images shape mismatch')
    else:
        return np.sum(np.abs(np.correlate(np.ndarray.flatten(refimage), np.ndarray.flatten(movingimage), 'full')))

# 3. Normalized Cross Correlation 
def ncc(refimage, movingimage, cor_type):
    
    if refimage.shape != movingimage.shape:
        print('images shape mismatch')
    else:
        if cor_type == 'pearson':
            return np.corrcoef(np.ndarray.flatten(refimage), np.ndarray.flatten(movingimage))[0,1] # pearson's correlation coefficient (this could also be implemented using scipy.stats.pearsonr)
        elif cor_type == 'spearman':
            return scipy.stats.spearmanr(np.ndarray.flatten(refimage), np.ndarray.flatten(movingimage))[0] # spearman correlation coefficient
    
# Entropy for MI and NMI
def entropy(hist):
    
    hist_normalized = hist/float(np.sum(hist))
    hist_normalized = hist_normalized[np.nonzero(hist_normalized)]
    return -sum(hist_normalized * np.log2(hist_normalized))

# 4. Mutual Information 
def mi(refimage, movingimage):
    
    if refimage.shape != movingimage.shape:
        print('images shape mismatch')
    else:
        hist_joint = np.histogram2d(np.ndarray.flatten(refimage), np.ndarray.flatten(movingimage), bins = 10)[0]
        hist_ref = np.histogram(np.ndarray.flatten(refimage), bins = 10)[0]
        hist_moving = np.histogram(np.ndarray.flatten(movingimage), bins = 10)[0]
        
        entropy_joint = entropy(hist_joint)
        entropy_ref = entropy(hist_ref)
        entropy_moving = entropy(hist_moving)
        return entropy_ref + entropy_moving - entropy_joint, entropy_ref, entropy_moving

# 5. Normalized Mutual Information 
def nmi(refimage, movingimage):
    
    mutual_info, ent_ref, ent_moving = mi(refimage, movingimage)
    return mutual_info/((ent_ref + ent_moving)*0.5)



