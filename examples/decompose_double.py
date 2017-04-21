
import gausspy.gp as gp
import time
import pickle

# Import data file. For "three phase", or absorption+emisison fits, 
# the data file should contain both spectra, saved as "x_values", "data_list" (absorption),
# "x_values_em" and "data_list_em" (emission) for each target.
data_file = 'example_data.pickle'
data = pickle.load(open(data_file))

g = gp.GaussianDecomposer()

# Set parameters for absorption fit
# In this case, two phase decomposition 
# using parameters trained for 21-SPONGE 
g.set('phase', 'two')
g.set('alpha1' , 1.12)
g.set('alpha2' , 2.73)

# Set alpha value for the one-phase
# fit to the emission residuals. In this case, 
# I used the one-phase trained alpha for 
# 21-SPONGE emission.
g.set('alpha_em', 2.1)

g.set('SNR_thresh', 3.)
g.set('SNR2_thresh', 3.)
g.set('plot', False)
g.set('verbose', False)

t0 = time.time()
data_decomposed = g.batch_decomposition(data_file)
print 'Elapsed time [s]: ', int(time.time() - t0)

# Include spectra in final output file
data_decomposed['x_values'] = data['x_values']
data_decomposed['data_list'] = data['data_list']
data_decomposed['x_values_em'] = data['x_values_em']
data_decomposed['data_list_em'] = data['data_list_em']

pickle.dump(data_decomposed, open('example_data_decomposed.pickle', 'w'))

