import pickle
import sys

from plyfile import PlyElement, PlyData
import numpy as np


pred_array = pickle.load(open('pred_vals.p', 'rb'))
pred_array = np.argmax(pred_array[3], 2)

input_array = pickle.load(open('input_vals.p', 'rb'))


def color(label, color):
	
	if label == 1 and color == 'r':
		return 255
	if label == 2 and color == 'g':
		return 255
	if label == 0 and color == 'b':
		return 255
	else:
		return 0

def create_colors(label_arr):
	
	result = []

	for index, value in enumerate(label_arr):

		temp_tuple = (
					color(value, 'r'),
					color(value, 'b'),
					color(value, 'g')
					)

		result.append(temp_tuple)

	return np.array(result, dtype=[('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])


for index in range(input_array.shape[0]):
	
	temp_labels = pred_array[index]

	temp_coords = input_array[index]
	temp_coords = [(x[0], x[1], x[2]) for x in temp_coords]

	vertex_color = create_colors(temp_labels)

	vertex = np.array(
					temp_coords,
					dtype=[('x', 'f8'), ('y', 'f8'), ('z', 'f8')]
					)
	n = len(vertex)
	assert len(vertex_color) == n

	vertex_all = np.empty(n, vertex.dtype.descr + vertex_color.dtype.descr)

	for prop in vertex.dtype.names:
	    vertex_all[prop] = vertex[prop]

	for prop in vertex_color.dtype.names:
	    vertex_all[prop] = vertex_color[prop]

	ply = PlyData([PlyElement.describe(vertex_all, 'vertex')], text=True)
	
	ply.write('pred/labled_{}.ply'.format(index))

	# with open('pred/labled_{}.ply'.format(index), mode='wb') as f:
	# 	PlyData([ply], byte_order='>').write(f)

	

	# el = PlyElement.describe(face, 'face_{}'.format(index))

	# with open('pred/labled_{}.ply'.format(index), mode='wb') as f:
	# 	PlyData([el], byte_order='>').write(f)

	# PlyData([el]).write('pred/labled_{}.ply'.format(index))




# vertex = numpy.array([(0, 0, 0),
#                       (0, 1, 1),
#                       (1, 0, 1),
#                       (1, 1, 0)],
#                      dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])

# vertex_color = numpy.array([(0, 255, 0),
#                             (0, 255, 255),
#                             (255, 0, 255),
#                             (255, 255, 0)],
#                            dtype=[('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])


# n = len(vertex)
# assert len(vertex_color) == n

# vertex_all = numpy.empty(n, vertex.dtype.descr + vertex_color.dtype.descr)

# for prop in vertex.dtype.names:
#     vertex_all[prop] = vertex[prop]

# for prop in vertex_color.dtype.names:
#     vertex_all[prop] = vertex_color[prop]

# ply = PlyData([PlyElement.describe(vertex_all, 'vertex')], text=True)
# ply.write(sys.stdout)