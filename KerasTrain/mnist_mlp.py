from __future__ import print_function

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from keras.utils import plot_model
import matplotlib.pyplot as plt

print(keras.__version__)

batch_size = 128
num_classes = 10
epochs = 100

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape)
x_train = x_train.reshape(60000, 784).astype("float32")
x_test = x_test.reshape(10000, 784).astype("float32")

x_train /= 255
x_test /= 255

print(x_train.shape[0], "Train samples")
print(x_test.shape[0], "Test samples")

print(y_train.shape)

# convert class vector to binary class matrix
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Dense(512, activation="relu", input_shape=(784,)))
model.add(Dropout(0.2))
model.add(Dense(512, activation=keras.backend.relu))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation="softmax"))

model.summary()

model.compile(loss="categorical_crossentropy",
              optimizer=keras.optimizers.RMSprop(),
              metrics=["accuracy"])

plot_model(model, to_file="model.png", show_shapes=True)

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)

print(history.history.keys())

x = history.history['acc']
fig = plt.figure()
plt.plot(history.history['acc'], "b")
plt.plot(history.history["val_acc"], "r")
plt.title("model accuracy")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.legend(["train", "test"])

plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("model loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["train", "test"])
plt.show()

print("Test loss:", score[0])
print("Test accuracy:", score[1])

plt.show()
