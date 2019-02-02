import numpy as np
from keras.models import model_from_json
from keras.preprocessing import image

file = open('dog-cat-classifier.json', 'r')
networkStructure = file.read()
file.close()

classifier = model_from_json(networkStructure)
classifier.load_weights('dog-cat-classifier-weights.h5')

testImage = image.load_img('dataset/luke.jpg', target_size = (64, 64))
testImage = image.img_to_array(testImage)
testImage /= 255
testImage = np.expand_dims(testImage, axis = 0)

prevision = classifier.predict(testImage)

print('Cat') if prevision > 0.5 else print('Dog')