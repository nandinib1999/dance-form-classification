import cv2
import argparse
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default='dance-form.h5', help="Name/Path of the saved model weights")
parser.add_argument("--encoder_file", type=str, default="classes.npy", help="Name/Path of the saved LabelEncoder")
parser.add_argument("--image_path", type=str, help="Path of the test image")
args = parser.parse_args()

def predict_label_image(image_path, model, encoder):
  img = cv2.imread(image_path)
  resize_image = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
  resize_image = resize_image / 255
  input_img = np.array(resize_image)
  input_img = np.expand_dims(input_img, axis=0)
  y_pred = model(input_img)
  label_ind = np.argmax(y_pred)
  label = encoder.inverse_transform([label_ind])
  return label

def main():
	model = load_model(args.model_name)
	encoder = LabelEncoder()
	encoder.classes_ = numpy.load('classes.npy')
	predict = predict_label_image(args.image_path, model, encoder)
	print("Predicted Class is ", predict)

if __name__ == '__main__':
	main()
