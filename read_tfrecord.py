import tensorflow as tf 
import glob

reader = tf.TFRecordReader()
filenames = glob.glob('*.tfrecords')
filename_queue = tf.train.string_input_producer(filenames)

_, serialized_example = reader.read(filename_queue)

feature_set = { 'label': tf.FixedLenFeature([], tf.string),
				'image': tf.FixedLenFeature([], tf.string),
				'weights': tf.FixedLenFeature([], tf.string) }

features = tf.parse_single_example( serialized_example, features= feature_set )
label = features['label']

with tf.Session() as sess:
	print sess.run([label])