import numpy as np
import tensorflow as tf

import train
from utils import provider


def generator():

    for el in train.TRAIN_DATASET:
        yield el


def generate_train_data():

	# Shuffle train samples
	dataset = train.TRAIN_DATASET
	data1 = tf.data.Dataset.from_tensor_slices(dataset[0])

	import pdb; pdb.set_trace()

	for ii in len(dataset):

		ps,seg,smpw = dataset[ii]
		aug_data = provider.rotate_point_cloud_z(ps)


def generate_test_data():
	pass


def generate_whole_test_data():
	pass


# data_arr = [
# 			{
# 			'int_data': 108,
# 			'float_data': 2.45,
# 			'str_data': 'String 100',
# 			'float_list_data': [256.78, 13.9]
# 			},
# 			{
# 			'int_data': 37,
# 			'float_data': 84.3,
# 			'str_data': 'String 200',
# 			'float_list_data': [1.34, 843.9, 65.22]
# 			}
# 			]


# def get_example_object(data_record):
	
# 	# tf.float32, tf.int32, tf.float32
# 	# Convert individual data into a list of int64 or float or bytes
# 	int_list1 = tf.train.Int64List(value = [data_record['int_data']])
# 	float_list1 = tf.train.FloatList(value = [data_record['float_data']])
# 	# Convert string data into list of bytes
# 	str_list1 = tf.train.BytesList(value = [data_record['str_data'].encode('utf-8')])
# 	float_list2 = tf.train.FloatList(value = data_record['float_list_data'])

# 	# Create a dictionary with above lists individually wrapped in Feature
# 	feature_key_value_pair = {
# 	'int_list1': tf.train.Feature(int64_list = int_list1),
# 	'float_list1': tf.train.Feature(float_list = float_list1),
# 	'str_list1': tf.train.Feature(bytes_list = str_list1),
# 	'float_list2': tf.train.Feature(float_list = float_list2)
# 	}

# 	# Create Features object with above feature dictionary
# 	features = tf.train.Features(feature = feature_key_value_pair)

# 	# Create Example object with features
# 	example = tf.train.Example(features = features)

# 	return example


if __name__ == "__main__":


	# with tf.python_io.TFRecordWriter('example.tfrecord') as tfwriter:

	# 	# Iterate through all records
	# 	for data_record in data_arr:

	# 		example = get_example_object(data_record)
	# 		# Append each example into tfrecord
	# 		tfwriter.write(example.SerializeToString())

	generate_train_data()    
