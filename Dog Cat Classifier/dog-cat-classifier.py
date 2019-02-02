from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
#import numpy as np
#from keras.preprocessing import image

# Convolutional Layers 
classifier = Sequential()
classifier.add( Conv2D( 64, (4,4), input_shape = (64, 64, 3), activation = 'relu') )
classifier.add( BatchNormalization() )
classifier.add( MaxPooling2D(pool_size = (2,2)) )

classifier.add( Conv2D( 64, (4,4), input_shape = (64, 64, 3), activation = 'relu') )
classifier.add( BatchNormalization() )
classifier.add( MaxPooling2D(pool_size = (2,2)) )

classifier.add( Flatten() )

# Adding the fully connected(dense) layers
classifier.add( Dense(units = 128, activation = 'relu') )
classifier.add( Dropout(0.2) )
classifier.add( Dense(units = 128, activation = 'relu') )
classifier.add( Dropout(0.2) )
classifier.add( Dense(units = 1, activation = 'sigmoid') )

classifier.compile( optimizer = 'adam',
                    loss = 'binary_crossentropy',
                    metrics = ['accuracy'])

trainingGenerator = ImageDataGenerator( rescale = 1./255,
                                        rotation_range = 7,
                                        horizontal_flip = True,
                                        shear_range = 0.2,
                                        height_shift_range = 0.07,
                                        zoom_range = 0.2)

testGenerator = ImageDataGenerator( rescale = 1./255 )

trainingDatabase = trainingGenerator.flow_from_directory( 'dataset/training_set',
                                                          target_size = (64, 64),
                                                          batch_size = 32,
                                                          class_mode = 'binary')

testDatabase = testGenerator.flow_from_directory( 'dataset/test_set',
                                                  target_size = (64, 64),
                                                  batch_size = 32,
                                                  class_mode = 'binary')

classifier.fit_generator( trainingDatabase, 
                          steps_per_epoch = 4000 / 8,
                          epochs = 10, 
                          validation_data = testDatabase,
                          validation_steps = 1000 / 8 )

# Save the network to avoid training again
classifierJSON = classifier.to_json()
with open('dog-cat-classifier.json', 'w') as json_file:
    json_file.write(classifierJSON)
classifier.save_weights('dog-cat-classifier-weights.h5')

trainingDatabase.class_indices

