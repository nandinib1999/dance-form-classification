import pandas as pd 
import cv2
import random
import os
import numpy as np
random.seed(1)

IMG_SIZE = 224

def fill(img, h, w):
    img = cv2.resize(img, (h, w), cv2.INTER_CUBIC)
    return img
        
def horizontal_shift(img, ratio=0.0):
    if ratio > 1 or ratio < 0:
        print('Value should be less than 1 and greater than 0')
        return img
    ratio = random.uniform(-ratio, ratio)
    w, h = img.shape[:2]
    to_shift = w*ratio
    if ratio > 0:
        img = img[:, :int(w-to_shift), :]
    if ratio < 0:
        img = img[:, int(-1*to_shift):, :]

    if img.shape[0] == 0:
      return None
    img = fill(img, h, w)
    return img

def brightness(img, value):
	bright = np.ones(img.shape , dtype="uint8") * value
	img = cv2.add(img,bright)
	return img

def flipping(img, flip_orientation):
	flip = cv2.flip(img,flip_orientation)
	return flip

def vertical_shift(img, ratio=0.0):
    if ratio > 1 or ratio < 0:
        print('Value should be less than 1 and greater than 0')
        return img
    ratio = random.uniform(-ratio, ratio)
    w, h = img.shape[:2]
    to_shift = h*ratio
    if ratio > 0:
        img = img[:int(h-to_shift), :, :]
    if ratio < 0:
        img = img[int(-1*to_shift):, :, :]

    if img.shape[0] == 0:
      return None
    img = fill(img, h, w)
    return img

def generate_image(img, label, image_name, train_dir):
	train_data = []
	value = random.uniform(0, 0.7)
	value2 = int(random.uniform(0, 60))
	orientation = random.choice([-1, 0, 1])

	vshift_img = vertical_shift(img, value)
	hshift_img = horizontal_shift(img, value)
	flip_img = flipping(img, orientation)
	bright_img = brightness(img, value2)

	aug_images = [vshift_img, hshift_img, flip_img, bright_img]
	for i in range(len(aug_images)):
		aimg = aug_images[i]
    if aimg not None:
      name_img = "aug_"+str(i)+"_"+image_name
      print('Writing ', name_img)
      aimg = cv2.resize(aimg,(IMG_SIZE,IMG_SIZE))
      cv2.imwrite(os.path.join(train_dir, name_img), aimg)
      train_data.append([name_img, label])

	return train_data


def augment_train_images(train_dir):
	list_of_images = os.listdir(train_dir)
  list_of_images = [x for x in list_of_images if 'aug_' not in x and x.endswith('.jpg')]
  print("Total Number of Images: ", len(list_of_images))
	train = pd.read_csv('train.csv')
	training_data = []
	for image in list_of_images:
		image_path = os.path.join(train_dir, image)
		img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
		if image in list(train['Image']):
			class_label = train.loc[train['Image'] == image, 'target'].values[0]
			training_data.append([image, class_label])
			augmented_images = generate_image(img, class_label, image, train_dir)
			training_data.extend(augmented_images)
			print(len(training_data))

	shuffled_training_data = random.sample(training_data, len(training_data))
	df = pd.DataFrame(shuffled_training_data, columns=['Image', 'target'])
	df.to_csv('train_augmented.csv', index=False)


if __name__ == '__main__':
	augment_train_images('./train')