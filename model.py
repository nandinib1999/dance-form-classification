from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Model

def vgg_model(num_classes):
	base_model = tf.keras.applications.VGG16(input_shape = (224,224,3), weights = 'imagenet',include_top=False)
	for layer in base_model.layers:
	    layer.trainable = False
	x = base_model.output
	x = tf.keras.layers.GlobalAveragePooling2D()(x)
	x = Dense(4096, activation='relu',name='fc1')(x)
	x = Dropout(0.2)(x)
	x = Dense(2048, activation='relu',name='fc2')(x)
	x = Dropout(0.2)(x)
	out = Dense(num_classes, activation='softmax', name='output_layer')(x)
	model = Model(inputs=base_model.input, outputs=out)
	return model