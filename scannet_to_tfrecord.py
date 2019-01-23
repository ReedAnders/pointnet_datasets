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

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

for index, (data, label, smpw) in enumerate(zip(batch_data, batch_label, batch_smpw)):


	filename = "data/tf_serialized/test_{}.tfrecords".format(index)
	writer = tf.python_io.TFRecordWriter(filename)

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