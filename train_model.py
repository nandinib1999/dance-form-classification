import cv2
import numpy as np
import os
import pandas as pd
from augment_images import augment_train_images
import utils
import argparse
import models
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

print(tf.__version__)

parser = argparse.ArgumentParser()
parser.add_argument("--train_dataset", type=str, default='train.csv', help="Dataset with training image filenames and labels")
parser.add_argument("--test_dataset", type=str, default="test.csv", help="Dataset with test image filenames and labels")
parser.add_argument("--train_images_folder", type=str, default="train", help="Folder where training images are stored.")
parser.add_argument("--test_images_folder", type=str, default="test", help="Folder where test images are stored.")
parser.add_argument("--test_split_size", type=float, default=0.2, help="For train and test split")
args = parser.parse_args()

def main():
	original_train = pd.read_csv(args.train_dataset)
	test = pd.read_csv(args.test_dataset)

	print("Original Training Dataset ", original_train.shape)
	print("Test Dataset ", test.shape)

	augment_train_images(args.train_images_folder)

	augmented_train = pd.read_csv('train_augmented.csv')
	print(augmented_train.shape)
	print(augmented_train.head())

	print("Original Dataset ")
	print(original_train['target'].value_counts())
	print("Augmented Dataset ")
	print(augmented_train['target'].value_counts())

	train_fnames = os.listdir(args.train_images_folder)
	test_fnames = os.listdir(args.test_images_folder)

	train_fnames = [x for x in train_fnames if x.endswith('.jpg')]
	test_fnames = [x for x in test_fnames if x.endswith('.jpg')]

	print("Training Images Count: ", len(train_fnames))
	print("Test Images Count: ", len(test_fnames))

	train_data, train_labels = utils.load_training_data(train_fnames, augmented_train, args.train_images_folder)
	train_data = np.array(train_data)
	train_labels = np.array(train_labels)
	print("Shape of training data: ", train_data.shape)
	print("Shape of training labels: ", train_labels.shape)

	le = LabelEncoder()
	train_labels=le.fit_transform(train_labels)
	np.save('classes.npy', le.classes_)

	X_train, X_val, y_train, y_val = train_test_split(train_data, train_labels, test_size=args.test_split_size, random_state=42)
	y_train = to_categorical(y_train, num_classes = 8)
	y_val = to_categorical(y_val, num_classes = 8)

	print("Shape of test_x: ",X_train.shape)
	print("Shape of train_y: ",y_train.shape)
	print("Shape of test_x: ",X_val.shape)
	print("Shape of test_y: ",y_val.shape)

	train_datagenerator = ImageDataGenerator(
        rescale=1. / 255
	) 

	val_datagenerator = ImageDataGenerator(
		rescale=1. / 255
	)

	train_datagenerator.fit(X_train)
	val_datagenerator.fit(X_val)

	num_classes = 8
	model = models.vgg_model(num_classes)

	print(model.summary())

	epochs = 30
	model.compile(optimizer='adam',
	              loss ='categorical_crossentropy',
	              metrics=['accuracy'])

	history = model.fit(X_train,y_train, batch_size=30, epochs=epochs, validation_data = (X_val,y_val))

	acc = history.history['accuracy']
	val_acc = history.history[ 'val_accuracy' ]
	loss = history.history[ 'loss' ]
	val_loss = history.history['val_loss' ]
	epochs = range(epochs)

	plt.figure(1)
	plt.plot(epochs, acc)
	plt.plot(epochs, val_acc)
	plt.title('Training and validation accuracy')
	plt.savefig('statics/accuracy.png')
	plt.show()

	plt.figure(2)
	plt.plot(epochs, loss)
	plt.plot(epochs, val_loss)
	plt.title('Training and validation loss')
	plt.savefig('statics/validation.png')
	plt.show()

	model.save('dance-form.h5')

if __name__ == '__main__':
	main()
