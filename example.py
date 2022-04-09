# Import the Conv2D and Flatten layers and instantiate model
from keras import Sequential, Model
from keras.layers import Conv2D, Flatten, Dense
from keras.preprocessing import image
from keras.applications.resnet import preprocess_input, ResNet50, decode_predictions
import numpy as np

model = Sequential()

# Add a convolutional layer of 32 filters of size 3x3
model.add(Conv2D(filters=32, input_shape=(28, 28, 1), kernel_size=3, activation='relu'))

# Add a convolutional layer of 16 filters of size 3x3
model.add(Conv2D(filters=16, kernel_size=3, activation='relu'))

# Flatten the previous layer output
model.add(Flatten())

# Add as many outputs as classes with softmax activation
model.add(Dense(10, activation='softmax'))

model.summary()

layer_outputs = [layer.output for layer in model.layers[:2]]

# Build a model using the model input and the layer outputs
activation_model = Model(inputs = model.inputs, outputs=layer_outputs)

# Load the image with the right target size for your model
img = image.load_img("Kris.jpg", target_size=(224, 224))

# Turn it into an array
img_array = image.img_to_array(img)
print("start")
print(img_array)
print("img_array printing finished")

# Expand the dimensions of the image
img_expanded = np.expand_dims(img_array, axis=0)

# Pre-process the img in the same way original images were
img_ready = preprocess_input(img_expanded)
print(img_ready)

model = ResNet50(weights='imagenet')

# Predict with ResNet50 on your already processed img
preds = model.predict(img_ready)

# Decode predictions
print('Predicted:', decode_predictions(preds, top=3)[0])