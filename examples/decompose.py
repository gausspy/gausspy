
import gausspy.gp as gp
import time
import pickle


SCIENCE_DATA = 'agd_data_science.pickle'

g = gp.GaussianDecomposer()

#Two phase
g.set('phase', 'two')
g.set('SNR_thresh', [5.,5.])
g.set('SNR2_thresh', [5.,0.])
g.set('alpha1', 4.19)
g.set('alpha2', 6.45)
g.set('mode', 'conv')

t0 = time.time()
new_data = g.batch_decomposition(SCIENCE_DATA)
print 'Elapsed time [s]: ', int(time.time() - t0)
pickle.dump(new_data, open('agd_data_science_decomposed.pickle', 'w'))
