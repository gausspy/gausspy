# Plot GaussPy results for selections of cube LOS
import numpy as np
import pickle
import matplotlib.pyplot as plt


datafile = 'example_data_decomposed.pickle'
data = pickle.load(open(datafile))

N= len(data['x_values'])
fig = plt.figure(0,[8,9])

ploti=1
for i in range(N):
 
    if ploti > N: break
    # PLOTTING
    ax1 = fig.add_subplot(N, 2, 2 * (ploti-1) + 1)
    ax2 = fig.add_subplot(N, 2, 2 * (ploti-1) + 2)

    x = data['x_values'][i]
    y = data['data_list'][i]

    x_em = data['x_values_em'][i]
    y_em = data['data_list_em'][i]
    
    fit_fwhms = data['fwhms_fit'][i]
    fit_means = data['means_fit'][i]
    fit_amps = np.array(data['amplitudes_fit'][i], dtype=float)

    fit_fwhms_em = data['fwhms_fit_em'][i]
    fit_means_em = data['means_fit_em'][i]
    fit_amps_em = np.array(data['amplitudes_fit_em'][i], dtype=float)

    # Labels emission components: 1 = absorption-fitted, 0 = emission only
    fit_labels = np.array(data['fit_labels'][i], dtype=int)

    # Plot absorption components
    if len(fit_amps) > 0.:
        for j in range(len(fit_amps)):
            amp, fwhm, mean =  fit_amps[j], fit_fwhms[j], fit_means[j]
            yy = amp * np.exp(-4. * np.log(2) * (x-mean)**2 / fwhm**2)
            ax1.plot(x,yy,'-',lw=1.5,color='purple')

    ax1.plot(x, y, color='black')

    model = x_em * 0.
    # Plot emission components
    if len(fit_amps_em) > 0.:
        for j in range(len(fit_amps_em)):
            amp, fwhm, mean =  fit_amps_em[j], fit_fwhms_em[j], fit_means_em[j]
            yy = amp * np.exp(-4. * np.log(2) * (x_em-mean)**2 / fwhm**2)
	    color = 'purple' if fit_labels[j]==1 else 'red'
            ax2.plot(x_em,yy,'-',lw=1.5,color=color)
	    model = model + yy

    ts = (fit_amps_em[fit_labels > 0]) / (1. - np.exp(-1. * fit_amps))
    print 'Spin temperature: ', ts

    ax2.plot(x_em, y_em, color='black')
    ax2.plot(x_em, model, color='green')
	
    ax1.set_xlim(-60,60)
    ax2.set_xlim(-60,60)
    ax1.set_xlabel('Velocity (km/s)')
    ax2.set_xlabel('Velocity (km/s)')
    ax2.set_ylabel(r'$\rm T_B (K)$')
    ax1.set_ylabel(r'$\tau$')
    ploti = ploti + 1

plt.tight_layout()
plt.savefig('example_data_decomposed.pdf', format='pdf')      
#plt.show()
