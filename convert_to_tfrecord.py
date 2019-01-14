import tensorflow as tf 

import numpy as np

import glob

# from PIL import Image

import train

# Converting the values into features
# _int64 is used for numeric values

def _int64_feature(value):
	return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _float_feature(value):
	return tf.train.Feature(bytes_list=tf.train.FloatList(value=[value]))

# _bytes is used for string/char values

def _bytes_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

tfrecord_filename = 'something.tfrecords'

# Initiating the writer and creating the tfrecords file.

writer = tf.python_io.TFRecordWriter(tfrecord_filename)

# Loading the location of all files - image dataset
# Considering our image dataset has apple or orange
# The images are named as apple01.jpg, apple02.jpg .. , orange01.jpg .. etc.

dataset = train.TRAIN_DATASET

for index, data in enumerate(dataset):

	print('Writing... {}'.format(index))
	
	tfrecord_filename = index
	writer = tf.python_io.TFRecordWriter('{}_scannet.tfrecords'.format(index))

	pointset = data[0]
	labels = data[1]
	weights = data[2]

	feature = { 'label': _bytes_feature(labels.tostring()),
				'image': _bytes_feature(pointset.tostring()),
				'weights': _bytes_feature(weights.tostring()) }

	# Create an example protocol buffer
	example = tf.train.Example(features=tf.train.Features(feature=feature))

	# Writing the serialized example.
	writer.write(example.SerializeToString())

	writer.close()