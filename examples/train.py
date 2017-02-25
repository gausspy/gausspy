
import gausspy.gp as gp
reload(gp)

TRAINING_DATA = 'agd_data.pickle'

g = gp.GaussianDecomposer()
g.load_training_data(TRAINING_DATA)


#One phase training
g.set('phase', 'one')
g.set('SNR_thresh', 5.)
g.set('SNR2_thresh', 5.)

g.train(alpha1_initial = 10., verbose = False, mode = 'conv',
                           learning_rate = 1.0, eps = 1.0, MAD = 0.1)
# F1=60%, Alpha = 6.56

#Two phase training
g.set('phase', 'two')
g.set('SNR_thresh', [5.,5.])
g.set('SNR2_thresh', [5.,0.])

g.train(alpha1_initial = 5.0, alpha2_initial = 7, plot=False,
                           verbose = False, mode = 'conv',
                           learning_rate = 1.0, eps = 1.0, MAD = 0.1)

# A=75%, 4.19, 6.45
