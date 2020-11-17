# Code created by Sudhakar on Nov 2020
# Siamese Network with Deep Transfer Learning and fine tuning Framework for automatic quality control of (rigid, affine) registrations

import os
import tensorflow as tf
from tensorflow.backend import K
import tensorflow_addons as tfa
from tensorflow.keras import models, layers
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, cross_val_score
import nibabel as nib
import numpy as np

def contrastive_loss(y_true, y_pred):
    margin = 1
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)

def SiameseNetwork(base_model, input_shape):
    
    moving_input = tf.keras.Input(input_shape)

    ref_input = tf.keras.Input(input_shape)
    
    encoded_moving = base_model(moving_input)
    encoded_ref = base_model(ref_input)

    L1_layer = tf.keras.layers.Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))    
    L1_distance = L1_layer([encoded_moving, encoded_ref])

    prediction = tf.keras.Dense(1,activation='sigmoid')(L1_distance)
    siamesenet = tf.keras.Model(inputs=[moving_input, ref_input],outputs=prediction)
    
    return siamesenet

data_dir = '/Volumes/TUMMALA/Work/Data/ABIDE-crossval/' # Path to the subjects data directory
subjects = sorted(os.listdir(data_dir)) # list of subject IDs' 

ref_path = "/usr/local/fsl/data/standard" # FSL template
ref_brain = ref_path+'/MNI152_T1_1mm.nii.gz' # Whole brain MNI

voi_size = 70
required_voi_size = [224, 224, 3]

ref_image = nib.load(ref_brain)
ref_image_data = ref_image.get_fdata()

x, y, z =  np.shape(ref_image_data)

required_ref_voi = ref_image_data[round(x/2)-voi_size:round(x/2)+voi_size:2, round(y/2)-voi_size:round(y/2)+voi_size:2, round(z/2)-voi_size:round(z/2)+voi_size:2]

reference_ref_voi = np.ndarray.flatten(required_ref_voi)[:np.prod(required_voi_size)].reshape(required_voi_size)
reference_voi = []
reference_voi.append(required_ref_voi)

correctly_aligned = []
mis_aligned = []

tag = 'hrT1'

# preparing training and testing data set
for subject in subjects:
    # correctly aligned images
    align_path = os.path.join(data_dir, subject, 'mni')
    align_path_images = os.listdir(align_path)
    # mis-aligned images
    test_path = os.path.join(data_dir, subject, 'test_imgs_T1_mni33')
    test_path_images = os.listdir(test_path)
    
    for align_path_image in align_path_images:
        if tag and align_path_image.endswith('reoriented.mni.nii'):
            input_image = nib.load(os.path.join(align_path, align_path_image))
            input_image_data = input_image.get_fdata()
            
            x, y, z =  np.shape(input_image_data)
            required_slice = input_image_data[round(x/2)-voi_size:round(x/2)+voi_size:2, round(y/2)-voi_size:round(y/2)+voi_size:2, round(z/2)-voi_size:round(z/2)+voi_size:2]
            required_slice = np.ndarray.flatten(required_slice)[:np.prod(required_voi_size)].reshape(required_voi_size)
            #required_slice = required_slice/np.max(required_slice)
            correctly_aligned.append(np.array(required_slice))
            print(f'correctly-aligned {np.shape(correctly_aligned)}')
            
    for test_path_image in test_path_images:
        test_input_image = nib.load(os.path.join(test_path, test_path_image))
        test_input_image_data = test_input_image.get_fdata()
        
        x, y, z =  np.shape(test_input_image_data)
        required_slice_test = test_input_image_data[round(x/2)-voi_size:round(x/2)+voi_size:2, round(y/2)-voi_size:round(y/2)+voi_size:2, round(z/2)-voi_size:round(z/2)+voi_size:2]
        required_slice_test = np.ndarray.flatten(required_slice_test)[:np.prod(required_voi_size)].reshape(required_voi_size)
        #required_slice_test = required_slice_test/np.max(required_slice_test)
        
        if np.shape(mis_aligned)[0] < np.shape(correctly_aligned)[0]:
            mis_aligned.append(np.array(required_slice_test))
    print(f'mis-aligned {np.shape(mis_aligned)}')

print('data is ready for deep cnn')

y_cor = np.zeros(np.shape(correctly_aligned)[0])
y_incor = np.ones(np.shape(mis_aligned)[0])

y_true = np.concatenate((y_cor, y_incor))
#y_true = tf.keras.utils.to_categorical(y_true, 2)
X = np.concatenate((correctly_aligned, mis_aligned))
#X = X.reshape(list(X.shape) + [1])

y1 = tf.convert_to_tensor(y_true, dtype = 'int32')
X1 = tf.convert_to_tensor(X)

# Transfer-learning starts

img_shape = np.shape(required_slice)

ref_voi_all = tf.keras.backend.repeat_elements(reference_voi, rep=np.shape(X)[0], axis = 0)

preprocess_input = tf.keras.applications.vgg16.preprocess_input()

X = preprocess_input(X) 
ref_voi_all = preprocess_input(ref_voi_all)

base_model = tf.keras.applications.VGG16(include_top = False, input_shape = img_shape, weights = 'imagenet')

# y_pred = tf.linalg.norm(ref_features_flat - image_features_flat, axis=1)
# tfa.losses.contrastive_loss(y_true, y_pred)

folds = RepeatedStratifiedKFold(n_splits = 2, n_repeats = 1)

for train_index, test_index in folds.split(X, y_true):
    X_train_moving, X_test_moving, y_train, y_test = X[train_index], X[test_index], y_true[train_index], y_true[test_index]
    X_train_ref, X_test_ref = ref_voi_all(train_index), ref_voi_all(test_index)
    
if True:
    inputs = tf.keras.Input(shape=img_shape)
    x1 = base_model(inputs, training = False)
    x1 = tf.keras.layers.GlobalAveragePooling2D(x1)
    x1 = tf.keras.layers.Dense(256, activation = 'relu')(x1)
    outputs = tf.keras.layers.Dropout(0.2)(x1)
    base_model = tf.keras.Model(inputs, outputs)

if True:
    siamese_model = SiameseNetwork(base_model, img_shape)

    base_learning_rate = 0.00001
    initial_epochs = 10
    siamese_model.compile(optimizer = tf.keras.optimizers.Adam(lr=base_learning_rate), loss = contrastive_loss, metrics = ['accuracy'])
    siamese_model.summary()
    history = siamese_model.fit(([X_train_moving, X_train_ref], y_train), batch_size = 32,
                        epochs = initial_epochs,
                        shuffle = True,
                        validation_data = ([X_test_moving, X_test_ref], y_test))

if True:
    base_model.trainable = True
    
    # Let's take a look to see how many layers are in the base model
    print("Number of layers in the base model: ", len(base_model.layers))
    
    # Fine-tune from this layer onwards
    fine_tune_at = int(0.7 * len(base_model.layers))
    
    # Freeze all the layers before the `fine_tune_at` layer
    for layer in base_model.layers[:fine_tune_at]:
      layer.trainable =  False
    
    siamese_model.compile(optimizer = tf.keras.optimizers.Adam(lr=base_learning_rate), loss = contrastive_loss, metrics = ['accuracy'])
    
    base_model.summary()
    
    fine_tune_epochs = 100
    total_epochs =  initial_epochs + fine_tune_epochs
    
    history_fine = siamese_model.fit(([X_train_moving, X_train_ref], y_train), batch_size = 32,
                              epochs=total_epochs,
                              initial_epoch = history.epoch[-1],
                              validation_data = ([X_test_moving, X_test_ref], y_test))


