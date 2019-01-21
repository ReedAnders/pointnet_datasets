import numpy as np
import tensorflow as tf 

import scannet_dataset


NUM_POINT = 8192
BATCH_SIZE = 32
NUM_CLASSES = 3

PRED_DATA_PATH = 'data/veggieplatter_data_real'

PRED_DATASET_WHOLE_SCENE = scannet_dataset.ScannetDatasetWholeScene_Pred(
    root=PRED_DATA_PATH, 
    npoints=NUM_POINT, 
    num_classes=NUM_CLASSES, 
    split='test')


is_training = False

is_continue_batch = False

extra_batch_data = np.zeros((0,NUM_POINT,3))
extra_batch_label = np.zeros((0,NUM_POINT))
extra_batch_smpw = np.zeros((0,NUM_POINT))

for batch_idx in range(BATCH_SIZE):
    if not is_continue_batch:
        # point_sets, semantic_segs, sample_weights = TEST_DATASET_WHOLE_SCENE
        batch_data, batch_label, batch_smpw, points_default = PRED_DATASET_WHOLE_SCENE[batch_idx]
        batch_data = np.concatenate((batch_data,extra_batch_data),axis=0)
        batch_label = np.concatenate((batch_label,extra_batch_label),axis=0)
        batch_smpw = np.concatenate((batch_smpw,extra_batch_smpw),axis=0)
    else:
        batch_data_tmp, batch_label_tmp, batch_smpw_tmp, _ = PRED_DATASET_WHOLE_SCENE[batch_idx]
        batch_data = np.concatenate((batch_data,batch_data_tmp),axis=0)
        batch_label = np.concatenate((batch_label,batch_label_tmp),axis=0)
        batch_smpw = np.concatenate((batch_smpw,batch_smpw_tmp),axis=0)

    if batch_data.shape[0]<BATCH_SIZE:
        is_continue_batch = True
        continue
    elif batch_data.shape[0]==BATCH_SIZE:
        is_continue_batch = False
        extra_batch_data = np.zeros((0,NUM_POINT,3))
        extra_batch_label = np.zeros((0,NUM_POINT))
        extra_batch_smpw = np.zeros((0,NUM_POINT))
    else:
        is_continue_batch = False
        extra_batch_data = batch_data[BATCH_SIZE:,:,:]
        extra_batch_label = batch_label[BATCH_SIZE:,:]
        extra_batch_smpw = batch_smpw[BATCH_SIZE:,:]
        batch_data = batch_data[:BATCH_SIZE,:,:]
        batch_label = batch_label[:BATCH_SIZE,:]
        batch_smpw = batch_smpw[:BATCH_SIZE,:]

data = (batch_data, batch_label, batch_smpw)

import pdb; pdb.set_trace()
# ops['is_training_pl']: is_training}

# Adapted from https://gist.github.com/blondegeek/724308125c0cdb795b617c7132148d12

# def _floatList_feature(value):
#   return tf.train.Feature(int64_list=tf.train.FloatList(value=value))

# def _int64_feature(value):
#   return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# class NumpyToTFHelper(object):
#     def __init__(self,filenames):
#         self.filenames = filenames
#         self.current_file = -1
#         self.current_index = 0 
#         self.current_array = None
#     def get_next(self):
#         if self.current_file == -1: 
#             self.current_file += 1
#             self.current_array = batch_data[self.current_file]
#         elif self.current_index == self.current_array.shape[0]-1:
#             if self.current_file == len(self.filenames) - 1:
#                 return None
#             self.current_file += 1
#             self.current_array = np.load(self.filenames[self.current_file])['arr_0']
#             self.current_index = 0 
#         self.current_index += 1 
#         # return single 16000 length record
#         cur = self.current_array[self.current_index -1] 
#         print(cur.shape)
#         print(type(cur))
#         return np.matrix(cur).A1

# files = ['CH_7381_8380_1478761500.9.npz']

filename = "data/tf_serialized/CH_int_array_TEST.tfrecords"
# print('Writing', filename)
writer = tf.python_io.TFRecordWriter(filename)
# N = NumpyToTFHelper(files) 
# n = N.get_next()

# while n is not None:
#     raw = n.tostring()
#     # remember to read back to numpy array, will need to specify that int was used.
#     example = tf.train.Example(features=tf.train.Features(feature={
#         'molecule':_int64list_feature(n)})) 
#     writer.write(example.SerializeToString())
#     n = N.get_next()
# writer.close()

for data, label, smpw in zip(batch_data, batch_label, batch_smpw):
	data_raw = data.tostring()
	label_raw = label.tostring()
	smpw_raw = smpw.tostring()
	example = tf.train.Example(features=tf.train.Features(feature={
        'pointclouds_pl':_bytes_feature(data_raw),
        'labels_pl':_bytes_feature(label_raw),
        'smpws_pl':_bytes_feature(smpw_raw)}
        # 'is_training_pl':_int64list_feature(is_training)}
        )) 
	writer.write(example.SerializeToString())
writer.close()