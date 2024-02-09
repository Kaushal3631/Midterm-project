
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

class FashionMNISTClassifier:
    def __init__(self):
        self.model = None
        self.probability_model = None
        self.class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                            'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    def load_data(self):
        fashion_mnist = keras.datasets.fashion_mnist
        (self.train_images, self.train_labels), (self.test_images, self.test_labels) = fashion_mnist.load_data()

        self.train_images = self.train_images / 255.0
        self.test_images = self.test_images / 255.0
        self.train_images = np.expand_dims(self.train_images, axis=-1)
        self.test_images = np.expand_dims(self.test_images, axis=-1)

    def build_model(self):
        self.model = keras.Sequential([
            keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(64, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(10)
        ])

        self.model.compile(optimizer='adam',
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                           metrics=['accuracy'])

    def train_model(self, epochs=10):
        self.model.fit(self.train_images, self.train_labels, epochs=epochs)

    def evaluate_model(self):
        test_loss, test_acc = self.model.evaluate(self.test_images, self.test_labels, verbose=2)
        print('\nTest accuracy:', test_acc)

    def build_probability_model(self):
        self.probability_model = tf.keras.Sequential([self.model, tf.keras.layers.Softmax()])

    def plot_image(self, i, predictions_array, true_label, img):
        true_label, img = true_label[i], img[i]
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])

        plt.imshow(img.squeeze(), cmap=plt.cm.binary)

        predicted_label = np.argmax(predictions_array)
        if predicted_label == true_label:
            color = 'blue'
        else:
            color = 'red'

        plt.xlabel("{} {:2.0f}% ({})".format(self.class_names[predicted_label],
                                              100*np.max(predictions_array),
                                              self.class_names[true_label]),
                                              color=color)

    def plot_value_array(self, i, predictions_array, true_label):
        true_label = true_label[i]
        plt.grid(False)
        plt.xticks(range(10))
        plt.yticks([])
        thisplot = plt.bar(range(10), predictions_array, color="#777777")
        plt.ylim([0, 1])
        predicted_label = np.argmax(predictions_array)

        thisplot[predicted_label].set_color('red')
        thisplot[true_label].set_color('blue')

    def plot_results(self, num_rows=5, num_cols=3):
        num_images = num_rows * num_cols
        plt.figure(figsize=(2*2*num_cols, 2*num_rows))
        predictions = self.probability_model.predict(self.test_images)

        for i in range(num_images):
            plt.subplot(num_rows, 2*num_cols, 2*i+1)
            self.plot_image(i, predictions[i], self.test_labels, self.test_images)
            plt.subplot(num_rows, 2*num_cols, 2*i+2)
            self.plot_value_array(i, predictions[i], self.test_labels)
        plt.tight_layout()
        plt.show()
