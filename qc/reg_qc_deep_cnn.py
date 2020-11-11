# Code created by Sudhakar on Oct/Nov 2020
# Deep CNN Framework for automatic quality control of (rigid, affine) registrations

import os
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import models, layers
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, cross_val_score
import nibabel as nib
import numpy as np

data_dir = '/home/tummala/data/HCP-100re' # Path to the subjects data directory
subjects = sorted(os.listdir(data_dir)) # list of subject IDs' 

ref_path = "/home/tummala/mri/tools/fsl/data/standard" # FSL template
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
            required_slice = input_image_data[round(x/2)-80:round(x/2)+80, round(y/2)-80:round(y/2)+80, round(z/2)-1:round(z/2)+2]
            #required_slice = required_slice/np.max(required_slice)
            correctly_aligned.append(np.array(required_slice))
            print(f'correctly-aligned {np.shape(correctly_aligned)}')
            
    for test_path_image in test_path_images:
        test_input_image = nib.load(os.path.join(test_path, test_path_image))
        test_input_image_data = test_input_image.get_fdata()
        
        x, y, z =  np.shape(test_input_image_data)
        required_slice_test = test_input_image_data[round(x/2)-80:round(x/2)+80, round(y/2)-80:round(y/2)+80, round(z/2)-1:round(z/2)+2]
        #required_slice_test = required_slice_test/np.max(required_slice_test)
        
        if np.shape(mis_aligned)[0] < np.shape(correctly_aligned)[0]:
            mis_aligned.append(np.array(required_slice_test))
    print(f'mis-aligned {np.shape(mis_aligned)}')

print('data is ready for deep cnn')

# Build a Deep CNN framework

# # Convolution and pooling layers
# model = models.Sequential()
# model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(218, 182, 1)))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# # Fully connected layers
# model.add(layers.Flatten())
# model.add(layers.Dense(64, activation='relu'))
# #model.add(layers.Dense(16, activation='relu'))
# model.add(layers.Dense(2, activation = 'softmax'))

# model.summary()
# model.compile(optimizer='adam', loss = tf.keras.losses.BinaryCrossentropy(from_logits = True), metrics=['accuracy'])

y_cor = np.zeros(np.shape(correctly_aligned)[0])
y_incor = np.ones(np.shape(mis_aligned)[0])

y_true = np.concatenate((y_cor, y_incor))
X = np.concatenate((correctly_aligned, mis_aligned))

y1 = tf.convert_to_tensor(y_true, dtype = 'int32')
X1 = tf.convert_to_tensor(X)

img_shape = np.shape(required_ref_voi)

preprocess_input = tf.keras.applications.inception_v3.preprocess_input

base_model = tf.keras.applications.InceptionV3(input_shape = img_shape,
                                                include_top = False,
                                                weights = 'imagenet')
base_model.trainable = False

ref_features = base_model(preprocess_input(tf.convert_to_tensor(reference_voi)))
image_features = base_model(preprocess_input(X1))

print(ref_features.shape)

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
image_features_flat = global_average_layer(image_features)

if False:
    proto_tensor = tf.make_tensor_proto(image_features_flat)
    image_features_array = tf.make_ndarray(proto_tensor) # Numpy array
    
ref_features_flat = global_average_layer(ref_features)
ref_features_flat = tf.keras.backend.repeat_elements(ref_features_flat, rep=np.shape(image_features_flat)[0], axis=0)

print(image_features_flat.shape)
print(ref_features_flat.shape)

y_pred = tf.linalg.norm(ref_features_flat - image_features_flat, axis=1)
tfa.losses.contrastive_loss(y_true, y_pred)

print('transfer learing')

output = base_model.layers[-1].output
output_flat = global_average_layer(output)
base_model = tf.keras.Model(base_model.input, output_flat)

if True:
    model = tf.keras.Sequential()
    model.add(base_model)
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid')) 

folds = RepeatedStratifiedKFold(n_splits = 2, n_repeats = 1)

for train_index, test_index in folds.split(X, y_true):
    X_train, X_test, y_train, y_test = X[train_index], X[test_index], y_true[train_index], y_true[test_index]

if True:
    base_learning_rate = 0.0001
    model.compile(optimizer=tf.keras.optimizers.Adam(lr = base_learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    model.summary()
    
    initial_epochs = 20
    
    # loss0, accuracy0 = model.evaluate(X1, y1)
    #training_data = tf.keras.data.Dataset(X1, y1)
    
    history = model.fit(preprocess_input(X_train), y_train, batch_size = 32,
                        epochs = initial_epochs,
                        validation_data = (preprocess_input(X_test), y_test))

if True:
    print('fine tuning')
    base_model.trainable = True
    
    # Let's take a look to see how many layers are in the base model
    print("Number of layers in the base model: ", len(base_model.layers))
    
    # Fine-tune from this layer onwards
    fine_tune_at = int(0.7 * len(base_model.layers))
    
    # Freeze all the layers before the `fine_tune_at` layer
    for layer in base_model.layers[:fine_tune_at]:
      layer.trainable =  False
    
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  optimizer = tf.keras.optimizers.RMSprop(lr=base_learning_rate/10),
                  metrics=['accuracy'])
    
    model.summary()
    
    fine_tune_epochs = 150
    total_epochs =  initial_epochs + fine_tune_epochs
    
    history_fine = model.fit(preprocess_input(X_train), y_train, batch_size = 32,
                             epochs=total_epochs,
                             initial_epoch = history.epoch[-1],
                             validation_data = (preprocess_input(X_test), y_test))


#X = X.reshape(-1, 218, 182, 1)

#X1 = tf.keras.utils.normalize(X, axis = 2)

#history = model.fit(X, y, epochs = 100, validation_data = None)