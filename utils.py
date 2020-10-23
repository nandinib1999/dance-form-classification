import cv2
import os
import matplotlib.pyplot as plt

IMG_SIZE = 224

def load_training_data(list_of_images, train, train_dir):
  train_data = [] 
  train_label = [] 
  for image in list_of_images:
      image_path = os.path.join(train_dir, image)
      img_array = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
      print(image_path)
      new_array = cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
      if image in list(train['Image']):
          class_label = train.loc[train['Image'] == image, 'target'].values[0]
          train_data.append(new_array)
          train_label.append(class_label)
  return train_data, train_label


def load_test_data(list_of_images, test_dir):
    test_data = []
    for image in list_of_images:
        image_path = os.path.join(test_dir, image)
        test_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        resize_image = cv2.resize(test_image, (IMG_SIZE,IMG_SIZE))
        test_data.append(resize_image) 
            
    return test_data

def show_sample(images, labels):
    plt.figure(figsize=(12,12))
    for n in range(25):
        ax = plt.subplot(5,5,n+1)
        plt.imshow(images[n])
        plt.title(labels[n].title())
        plt.axis('off')