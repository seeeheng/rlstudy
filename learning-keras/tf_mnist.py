import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt

fashion_mnist = tf.keras.datasets.fashion_mnist
(train_x, train_y),(test_x, test_y) = fashion_mnist.load_data()
print(train_x[0].shape) # fashion_mnist is a 28x28 grayscale image.

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'] # 10 different labels.

class FashionMnist(tf.keras.Model):
	def __init__(self):
		super(FashionMnist, self).__init__()
		self.flatten = tf.keras.layers.Flatten(input_shape=(28,28))
		self.fc1 = tf.keras.layers.Dense(128, activation='relu')
		self.out = tf.keras.layers.Dense(10) # no activation cause this is the output.

	def call(self, x):
		x = self.flatten(x)
		x = self.fc1(x)
		x = self.out(x)
		return x

	def predict(self, x):
		results = self.call(x)
		predictions = tf.nn.softmax(results).numpy()
		return predictions

model = FashionMnist()
train_x, test_x = train_x/255.0, test_x/255.0

plt.figure(figsize=(10,10))
for i in range(25):
	plt.subplot(5,5,i+1) # 5 rows, 5 columns
	plt.xticks([])
	plt.yticks([])
	plt.grid(False)
	plt.imshow(train_x[i], cmap=plt.cm.binary)
	plt.xlabel(class_names[train_y[i]])

loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
model.fit(train_x, train_y, epochs=2)
model.evaluate(test_x, test_y, verbose=2)

# LEARNING POINT: using test_x[:1] and test_x[0] will be very different.
# test_x[:1] maintains the structure with batch, i.e. (1,28,28)
# test_x[0] doesn't, i.e. (28,28)
predictions = model.predict(test_x[:1])
print(np.argmax(predictions))