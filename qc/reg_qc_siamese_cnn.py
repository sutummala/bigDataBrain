# Code created by Sudhakar on Nov 2020
# Siamese Network with Deep Transfer Learning and fine tuning Framework for fully automatic quality control of (rigid, affine) registrations

import os
import tensorflow as tf
from tensorflow.keras import backend as K
import tensorflow_addons as tfa
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
import nibabel as nib
import numpy as np
import random

def contrastive_loss(y_true, y_pred):
    
    margin = 1
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)

def generate_pairs(X, y):
    
    image_list = np.split(X, X.shape[0])
    label_list = np.split(y, len(y))
    
    left_input = []
    right_input = []
    targets = []
    
    pairs = 5 # Generating five pairs for each image
    
    for i in range(len(label_list)):
        for _ in range(pairs):
            compare_to = i
            while compare_to == i: #Make sure it's not comparing to itself
                compare_to = random.randint(0, X.shape[0]-1)
            left_input.append(image_list[i])
            right_input.append(image_list[compare_to])
            if label_list[i] == label_list[compare_to]:# They are the same
                targets.append(1.)
            else:# Not the same
                targets.append(0.)
                
    left_input = np.squeeze(np.array(left_input))
    right_input = np.squeeze(np.array(right_input))
    targets = np.squeeze(np.array(targets))
    
    return left_input, right_input, targets
                
def SiameseNetwork(model, input_shape):
    
    moving_input = tf.keras.Input(input_shape)
    ref_input = tf.keras.Input(input_shape)
    
    encoded_moving = model(moving_input)
    encoded_ref = model(ref_input)

    L1_layer = tf.keras.layers.Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))    
    L1_distance = L1_layer([encoded_moving, encoded_ref]) # L1-norm

    prediction = tf.keras.layers.Dense(1, activation='sigmoid')(L1_distance)
    siamesenet = tf.keras.Model(inputs = [moving_input, ref_input], outputs = prediction)
    
    return siamesenet

data_dir = '/home/tummala/data/HCP-100re' # Path to the subjects data directory
subjects = sorted(os.listdir(data_dir)) # list of subject IDs' 

ref_path = '/home/tummala/mri/tools/fsl/data/standard' # FSL template
ref_brain = ref_path+'/MNI152_T1_1mm.nii.gz' # Whole brain MNI

voi_size = 70
required_voi_size = [224, 224, 3] # suitable for vgg16 and resnet50, could be 299,299,3 for Inception V3

ref_image = nib.load(ref_brain)
ref_image_data = ref_image.get_fdata()

x, y, z =  np.shape(ref_image_data)

required_ref_voi = ref_image_data[round(x/2)-voi_size:round(x/2)+voi_size:2, round(y/2)-voi_size:round(y/2)+voi_size:2, round(z/2)-voi_size:round(z/2)+voi_size:2]
required_ref_voi = np.ndarray.flatten(required_ref_voi)[:np.prod(required_voi_size)].reshape(required_voi_size)
#required_ref_voi = (required_ref_voi - np.min(required_ref_voi))/(np.max(required_ref_voi) - np.min(required_ref_voi)) # scaling to bring the values between zero and one

reference = []
correctly_aligned = []
mis_aligned = []

tag = 'hrT1'

# preparing training and testing data set

for subject in subjects:
    # correctly aligned images
    align_path = os.path.join(data_dir, subject, 'align')
    align_path_images = os.listdir(align_path)
    # mis-aligned images
    test_path = os.path.join(data_dir, subject, 'test_imgs_T1_align33')
    test_path_images = os.listdir(test_path)
    
    for align_path_image in align_path_images:
        if tag in align_path_image and align_path_image.endswith('reoriented.align.nii'):
            input_image = nib.load(os.path.join(align_path, align_path_image))
            input_image_data = input_image.get_fdata()
            
            x, y, z =  np.shape(input_image_data)
            required_slice = input_image_data[round(x/2)-voi_size:round(x/2)+voi_size:2, round(y/2)-voi_size:round(y/2)+voi_size:2, round(z/2)-voi_size:round(z/2)+voi_size:2]
            required_slice = np.ndarray.flatten(required_slice)[:np.prod(required_voi_size)].reshape(required_voi_size)
            #required_slice = (required_slice - np.min(required_slice))/(np.max(required_slice) - np.min(required_slice))
            correctly_aligned.append(np.array(required_slice))
            print(f'correctly-aligned {np.shape(correctly_aligned)}')
            
    for test_path_image in test_path_images:
        test_input_image = nib.load(os.path.join(test_path, test_path_image))
        test_input_image_data = test_input_image.get_fdata()
        
        x, y, z =  np.shape(test_input_image_data)
        required_slice_test = test_input_image_data[round(x/2)-voi_size:round(x/2)+voi_size:2, round(y/2)-voi_size:round(y/2)+voi_size:2, round(z/2)-voi_size:round(z/2)+voi_size:2]
        required_slice_test = np.ndarray.flatten(required_slice_test)[:np.prod(required_voi_size)].reshape(required_voi_size)
        #required_slice_test = (required_slice_test - np.min(required_slice_test))/(np.max(required_slice_test) - np.min(required_slice_test))
        
        if np.shape(mis_aligned)[0] < np.shape(correctly_aligned)[0]:
            mis_aligned.append(np.array(required_slice_test))
    print(f'mis-aligned {np.shape(mis_aligned)}')

print('data is ready for deep cnn')

y_cor = np.ones(np.shape(correctly_aligned)[0])
y_incor = np.zeros(np.shape(mis_aligned)[0])

y_true = np.concatenate((y_cor, y_incor))
#y_true = tf.keras.utils.to_categorical(y_true, 2)
X = np.concatenate((correctly_aligned, mis_aligned))
#X = X.reshape(list(X.shape) + [1])

# Generate pairs
left_input, right_input, targets = generate_pairs(X, y_true)

np.save('/home/tummala/data/left', left_input)
np.save('/home/tummala/data/right', right_input)
np.save('/home/tummala/data/labels', targets)

y1 = tf.convert_to_tensor(y_true, dtype = 'int32')
X1 = tf.convert_to_tensor(X)

img_shape = np.shape(required_slice)
reference = np.repeat(required_ref_voi[np.newaxis, :, :, :], np.shape(X)[0], axis=0) # repeating reference to match with the sample size of moving images

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
preprocess_input = tf.keras.applications.vgg16.preprocess_input

X = preprocess_input(X) 
reference = preprocess_input(reference)

base_model = tf.keras.applications.VGG16(include_top = False, input_shape = img_shape, weights = 'imagenet')

base_model.trainable =  False

# y_pred = tf.linalg.norm(ref_features_flat - image_features_flat, axis=1)
# tfa.losses.contrastive_loss(y_true, y_pred)

output = base_model.layers[-1].output
output_flat = global_average_layer(output)
base_model = tf.keras.Model(base_model.input, output_flat)

folds = RepeatedStratifiedKFold(n_splits = 2, n_repeats = 1)

for train_index, test_index in folds.split(X, y_true):
    X_train_moving, X_test_moving, y_train, y_test = X[train_index], X[test_index], y_true[train_index], y_true[test_index]
    X_train_ref, X_test_ref = reference[train_index], reference[test_index]
    
if True:
    model = tf.keras.Sequential()
    model.add(base_model)
    model.add(tf.keras.layers.Dense(512, activation='sigmoid'))
    model.add(tf.keras.layers.Dropout(0.2))
    
if True:
    siamese_model = SiameseNetwork(model, img_shape)

    base_learning_rate = 0.0001
    initial_epochs = 10
    siamese_model.compile(optimizer = tf.keras.optimizers.Adam(lr = base_learning_rate), loss = tfa.losses.contrastive_loss, metrics = tf.keras.metrics.Recall())
    siamese_model.summary()
    history = siamese_model.fit((X[0], X[1]), y_true, batch_size = 8,
                        epochs = initial_epochs, verbose = 1,
                        validation_data = None) 

if True:
    base_model.trainable = True
    
    # Let's take a look to see how many layers are in the base model
    print("Number of layers in the base model: ", len(base_model.layers))
    
    # Fine-tune from this layer onwards
    fine_tune_at = int(0.7 * len(base_model.layers))
    
    # Freeze all the layers before the `fine_tune_at` layer
    for layer in base_model.layers[:fine_tune_at]:
      layer.trainable =  False
    
    siamese_model.compile(optimizer = tf.keras.optimizers.Adam(lr = base_learning_rate/10), loss = tfa.losses.contrastive_loss, metrics = ['accuracy'])
    
    siamese_model.summary()
    
    fine_tune_epochs = 20
    total_epochs =  initial_epochs + fine_tune_epochs
    
    history_fine = siamese_model.fit([X_train_moving, X_train_ref], y_train, batch_size = 32,
                              epochs=total_epochs,
                              initial_epoch = history.epoch[-1],
                              validation_data = ([X_test_moving, X_test_ref], y_test))
