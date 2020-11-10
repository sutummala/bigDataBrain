# Code created by Sudhakar on Oct/Nov 2020
# Deep CNN Framework for automatic quality control of (rigid, affine) registrations

import os
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import models, layers
import nibabel as nib
import numpy as np

data_dir = '/Volumes/TUMMALA/Work/Data/ABIDE-validate/' # Path to the subjects data directory
subjects = sorted(os.listdir(data_dir)) # list of subject IDs' 

ref_path = "/usr/local/fsl/data/standard" # FSL template
ref_brain = ref_path+'/MNI152_T1_1mm.nii.gz' # Whole brain MNI

ref_image = nib.load(ref_brain)
ref_image_data = ref_image.get_fdata()

x, y, z =  np.shape(ref_image_data)

required_ref_voi = ref_image_data[round(x/2)-80:round(x/2)+80, round(y/2)-80:round(y/2)+80, round(z/2)-1:round(z/2)+2]

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
            required_slice = input_image_data[round(x/2)-40:round(x/2)+40, round(y/2)-40:round(y/2)+40, round(z/2)-40:round(z/2)+40]
            #required_slice = required_slice/np.max(required_slice)
            correctly_aligned.append(np.array(required_slice))
            print(f'correctly-aligned {np.shape(correctly_aligned)}')
            
    for test_path_image in test_path_images:
        test_input_image = nib.load(os.path.join(test_path, test_path_image))
        test_input_image_data = input_image.get_fdata()
        
        x, y, z =  np.shape(test_input_image_data)
        required_slice_test = test_input_image_data[round(x/2)-40:round(x/2)+40, round(y/2)-40:round(y/2)+40, round(z/2)-40:round(z/2)+40]
        #required_slice_test = required_slice_test/np.max(required_slice_test)
        
        if np.shape(mis_aligned)[0] < np.shape(correctly_aligned)[0]:
            mis_aligned.append(np.array(required_slice_test))
        print(f'mis-aligned {np.shape(mis_aligned)}')

print('data is ready for deep cnn')

# Build a Deep CNN framework

# Convolution and pooling layers
model = models.Sequential()
model.add(layers.Conv3D(32, (3, 3, 3), activation='relu', input_shape=(80, 80, 80) + (1,)))
model.add(layers.MaxPooling3D((2, 2, 2)))
model.add(layers.Conv3D(64, (3, 3, 3), activation='relu'))
model.add(layers.MaxPooling3D((2, 2, 2)))
model.add(layers.Conv3D(64, (3, 3, 3), activation='relu'))
model.add(layers.MaxPooling3D((2, 2, 2)))
model.add(layers.Conv3D(64, (3, 3, 3), activation='relu'))

# Fully connected layers
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
#model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(2, activation = 'sigmoid'))

model.summary()
model.compile(optimizer='adam', loss = tf.keras.losses.BinaryCrossentropy(from_logits = True), metrics=['accuracy'])

y_cor = np.zeros(np.shape(correctly_aligned)[0])
y_incor = np.ones(np.shape(mis_aligned)[0])

y_true = np.concatenate((y_cor, y_incor))
y_true = tf.keras.utils.to_categorical(y_true, 2)
X = np.concatenate((correctly_aligned, mis_aligned))
X = X.reshape(list(X.shape) + [1])

y1 = tf.convert_to_tensor(y_true, dtype = 'int32')
X1 = tf.convert_to_tensor(X)

model.fit(X1, y1, epochs=10)

# img_shape = np.shape(required_ref_voi)

# preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

# base_model = tf.keras.applications.MobileNetV2(input_shape=img_shape,
#                                                 include_top=False,
#                                                 weights='imagenet')
# base_model.trainable = False

# ref_features = base_model(preprocess_input(tf.convert_to_tensor(reference_voi)))
# image_features = base_model(preprocess_input(X1))

# print(ref_features.shape)

# global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
# image_features_flat = global_average_layer(image_features)
# ref_features_flat = global_average_layer(ref_features)
# ref_features_flat = tf.keras.backend.repeat_elements(ref_features_flat, rep=np.shape(image_features_flat)[0], axis=0)

# print(image_features_flat.shape)
# print(ref_features_flat.shape)

# y_pred = tf.linalg.norm(ref_features_flat - image_features_flat, axis=1)
# tfa.losses.contrastive_loss(y_true, y_pred)

# if True:
#     prediction_layer = tf.keras.layers.Dense(1, activation = 'sigmoid')
#     inputs = tf.keras.Input(shape=img_shape)
#     x1 = preprocess_input(inputs)
#     x1 = base_model(x1, training = False)
#     x1 = global_average_layer(x1)
#     x1 = tf.keras.layers.Dropout(0.2)(x1)
#     outputs = prediction_layer(x1)
#     model = tf.keras.Model(inputs, outputs)

# if True:
#     base_learning_rate = 0.0001
#     model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
#                   loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
#                   metrics=['accuracy'])
    
#     model.summary()
    
#     initial_epochs = 10
    
#     # loss0, accuracy0 = model.evaluate(X1, y1)
#     #training_data = tf.keras.data.Dataset(X1, y1)
    
#     history = model.fit(X1, y1, batch_size = 8,
#                         epochs = initial_epochs,
#                         validation_data = None)

# if True:
#     base_model.trainable = True
    
#     # Let's take a look to see how many layers are in the base model
#     print("Number of layers in the base model: ", len(base_model.layers))
    
#     # Fine-tune from this layer onwards
#     fine_tune_at = 10
    
#     # Freeze all the layers before the `fine_tune_at` layer
#     for layer in base_model.layers[:fine_tune_at]:
#       layer.trainable =  False
    
#     model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
#                   optimizer = tf.keras.optimizers.RMSprop(lr=base_learning_rate/10),
#                   metrics=['accuracy'])
    
#     model.summary()
    
#     fine_tune_epochs = 10
#     total_epochs =  initial_epochs + fine_tune_epochs
    
#     history_fine = model.fit(X1, y1, batch_size = 4,
#                              epochs=total_epochs,
#                              initial_epoch = history.epoch[-1],
#                              validation_data = None)


# #X = X.reshape(-1, 218, 182, 1)

# #X1 = tf.keras.utils.normalize(X, axis = 2)

# #history = model.fit(X, y, epochs = 100, validation_data = None)