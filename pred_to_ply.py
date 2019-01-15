import pickle

import plyfile


pred_array = pickle.load(open('../pred_vals', 'rb'))

import pdb; pdb.set_trace()

labels = np.argmax(pred_array[3], 2)

face = numpy.array([([0, 1, 2], 255, 255, 255),
					([0, 2, 3], 255,   0,   0),
					([0, 1, 3],   0, 255,   0),
					([1, 2, 3],   0,   0, 255)],
					dtype=[('vertex_indices', 'i4', (3,)),
					('red', 'u1'), ('green', 'u1'),
					('blue', 'u1')])

el = PlyElement.describe(some_array, 'some_name')