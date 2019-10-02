####################################################################
##################### IMPORTING REQUIRED MODULES ###################

import os.path
import numpy as np
import rampy as rp
import pylab as plb
from matplotlib import gridspec
import fnmatch
import lmfit
from scipy import stats

####################################################################
########################## SET PARAMETERS ############################

file_baseg = 'g'
file_base2d = '2d'
file_base = 'flake'
file_suffix = '.txt'
fig_suffix = '.png'


first_file = 0
numberoffiles = len(fnmatch.filter(os.listdir('.'), '2d*.txt'))

#this ratio scales intensities of G and 2D peaks
ratio = 2 #change this to 1 if your measurements are the same length and accumulations


####################################################################
######################### DEFINING FUNCTIONS #######################

def lorentzian(x, a, f, l, y0=0): # [hwhm, peak center, intensity, y0] #
    numerator =  (0.5*l)**2
    denominator = ( x - (f) )**2 + (0.5*l)**2
    y = a*(numerator/denominator)+y0
    return y


def residual4(pars, x, data=None, eps=None):
    # unpack parameters:
    #  extract .value attribute for each parameter
    # a - amplitude; f - freaquency, l - FWHM?
    a1 = pars['a1'].value
    a2 = pars['a2'].value
    a3 = pars['a3'].value
    a4 = pars['a4'].value

    
    f1 = pars['f1'].value
    f2 = pars['f2'].value
    f3 = pars['f3'].value
    f4 = pars['f4'].value
    
    
    l1 = pars['l1'].value
    l2 = pars['l2'].value
    l3 = pars['l3'].value
    l4 = pars['l4'].value

    
    # lorentzian model
    
    peak1 = lorentzian(x,a1,f1,l1)
    peak2 = lorentzian(x,a2,f2,l2)
    peak3 = lorentzian(x,a3,f3,l3)
    peak4 = lorentzian(x,a4,f4,l4)
  
    
    model = peak1 + peak2 + peak3 + peak4
    
    if data is None:
        return model, peak1, peak2, peak3, peak4
    if eps is None:
        return (model - data)
    return (model - data)/eps


def residual2(pars, x, data=None, eps=None):
    # unpack parameters:
    #  extract .value attribute for each parameter
    # a - amplitude; f - freaquency, l - FWHM?
    a1 = pars['a1'].value
    a2 = pars['a2'].value
        
    f1 = pars['f1'].value
    f2 = pars['f2'].value
   
    l1 = pars['l1'].value
    l2 = pars['l2'].value

    
    # lorentzian model
    
    peak1 = lorentzian(x,a1,f1,l1)
    peak2 = lorentzian(x,a2,f2,l2)
  
    
    model = peak1 + peak2
    
    if data is None:
        return model, peak1, peak2
    if eps is None:
        return (model - data)
    return (model - data)/eps

def residual1(pars, x, data=None, eps=None):
    # unpack parameters:
    #  extract .value attribute for each parameter
    # a - amplitude; f - freaquency, l - FWHM?
    a1 = pars['a1'].value        
    f1 = pars['f1'].value   
    l1 = pars['l1'].value
    
    # lorentzian model
    
    peak1 = lorentzian(x,a1,f1,l1) 
    
    model = peak1
    
    if data is None:
        return model, peak1
    if eps is None:
        return (model - data)
    return (model - data)/eps


####################################################################
########################## LOADING DATA ############################
#initial arrays
allinall=[]

for e in range(first_file, 3): #numberoffiles

    # generate corresponding full file name
    full_fnameg = file_baseg + str(e) + file_suffix
    full_fname2d = file_base2d + str(e) + file_suffix

    if not (os.path.exists(full_fnameg) and os.path.exists(full_fname2d)):
        print ("no such file %s " % full_fnameg, full_fname2d)
        e =+ 1
    else:
        # read that file into an array
        filedatag = np.genfromtxt(full_fnameg, comments='#', delimiter='\t')
        filedata2d = np.genfromtxt(full_fname2d, comments='#', delimiter='\t')


        ####################################################################
        ########################## D, G, 2D peak fitting ############################

        #load data
        xg = filedatag[:,0]
        yg_org = filedatag[:,1]
        x2d = filedata2d[:,0]
        y2d_org = filedata2d[:,1]/ratio
        
        #smooth
        yg_s = rp.smooth(xg,yg_org,method="whittaker",Lambda=10)
        y2d_s = rp.smooth(x2d,y2d_org,method="whittaker",Lambda=10)        
        
        #remove background
            #g peak
        bir = np.array([(min(xg),1030),(1900,max(xg))])
        yg_cor, background = rp.baseline(xg,yg_s,bir,"arPLS",lam=10**8)
        yg_corr = yg_cor[:,0]
        
            #2d peak
        bir = np.array([(min(x2d),2550),(3100,max(x2d))])
        y2d_cor, background = rp.baseline(x2d,y2d_s,bir,"arPLS",lam=10**8)
        y2d_corr = y2d_cor[:,0]        
        
        #fix spectrum
        y = np.concatenate((y2d_corr,yg_corr))
        x = np.concatenate((x2d,xg))
        
                
        bir = np.array([(min(x),1050.),(1880.,2300.), (2400.,2500),(3050.,max(x))])
        yg_corrected, background = rp.baseline(x,y,bir,"arPLS",lam=10**8)
        y = yg_corrected[:,0]
              
        #normalise
        yg = rp.normalise(yg_corr,method="minmax")
        y2d = rp.normalise(y2d_corr,method="minmax")
        y = rp.normalise(y,method="minmax")
        
        
        ##fitting t-PA##

        # signal selection
        x_fit = xg[np.where((xg > 1050)&(xg < 1250))]
        y_fit = yg_org[np.where((xg > 1050)&(xg < 1250))]
        
        #signal processing
        bir = np.array([(1050., 1190)])
        y_fit, background = rp.baseline(x_fit,y_fit,bir,"poly",poly=1)
        
        y_fit = rp.smooth(x_fit,y_fit[:,0],method="whittaker",Lambda=10)
        y_fit = rp.normalise(y_fit,method="minmax")

        params = lmfit.Parameters()
#               (Name,  Value,  Vary,   Min,  Max,  Expr)
        params.add_many(('a1',   0.3,   True,  0,      1,  None),
                        ('f1',   1135,   True, 1100,   1150,  None),
                        ('l1',   40,   True,  20,      80,  None))

        result = lmfit.minimize(residual1, params, method ='least_squares', args=(x_fit, y_fit)) 

        model = lmfit.fit_report(result.params)
        yout, peak1 = residual1(result.params,x_fit)
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(y_fit, yout)
        r_tpa = r_value**2
        
        
        #g peak fitting
        
        if (result.params['a1'].value >= 0.55) and (r_tpa >= 0.25):
            params = lmfit.Parameters()
    #               (Name,  Value,  Vary,   Min,  Max,  Expr)
            params.add_many(('a1',   0.5,   True,  0,      1,  None),
                            ('f1',   1350,   True, 1330,   1370,  None),
                            ('l1',   40,   True,  0,      150,  None),
                            ('a2',  1,   True,  0.3,     1,  None),
                            ('f2',   1580,   True, 1550,   1600,  None),
                            ('l2',   20,   True,  0,   80,  None),
                            ('a3',   None,   True,  0,     0.6,  'a1/4'),
                            ('f3',   None,   True,  None,     None,  'f1+270'),
                            ('l3',   None,   True,  None,     None,  'l1*0.25'),
                            ('a4',  0.3,   True,  0,   0.7,  None),
                            ('f4',   1530,   True,  1505, 1560,  None),
                            ('l4',   50,   True,  20,     80,  None))

            result = lmfit.minimize(residual4, params, method = 'least_squares', args=(xg, yg))         
            modelg = lmfit.fit_report(result.params)
            youtg, peakg1,peakg2, peakg3, peakg4 = residual4(result.params,xg) # the different peaks

            #save results in a list
            best_para = [e, 'tpa']
            for name in result.params.valuesdict():
                best_para=best_para+[result.params[name].value]
                
        else:
            params = lmfit.Parameters()
    #               (Name,  Value,  Vary,   Min,  Max,  Expr)
            params.add_many(('a1',   0.5,   True,  0,      1,  None),
                            ('f1',   1350,   True, 1330,   1370,  None),
                            ('l1',   40,   True,  0,      150,  None),
                            ('a2',  1,   True,  0.3,     1,  None),
                            ('f2',   1580,   True, 1550,   1600,  None),
                            ('l2',   20,   True,  0,   80,  None),
                            ('a3',   None,   True,  0,     0.6,  'a1/4'),
                            ('f3',   None,   True,  None,     None,  'f1+270'),
                            ('l3',   None,   True,  None,     None,  'l1*0.25'),
                            ('a4',  0,   None,  None,   None,  None),
                            ('f4',  0,   None,  None,   None,  None),
                            ('l4',   0,   None,  None,   None,  None))

            result = lmfit.minimize(residual4, params, method = 'least_squares', args=(xg, yg))         
            modelg = lmfit.fit_report(result.params)
            youtg, peakg1,peakg2, peakg3, peakg4 = residual4(result.params,xg) # the different peaks

            #save results in a list
            best_para = [e, 'defected']
            for name in result.params.valuesdict():
                best_para=best_para+[result.params[name].value]
                
        



        #2D peak fitting
        params = lmfit.Parameters()
#               (Name,  Value,  Vary,   Min,  Max,  Expr)
        params.add_many(('a1',  0.5,   True,  0,     1,  None),
                        ('f1',   2690,   True, 2650,   2720,  None),
                        ('l1',   50,   True,  0,   180,  None),
                        ('a2',  0.2,   True,  0,     1,  None),
                        ('f2',   2940,   True, 2900,   3100,  None),
                        ('l2',   40,   True,  0,   180,  None))

        result = lmfit.minimize(residual2, params, method = 'leastsq', args=(x2d, y2d))         
        model2d = lmfit.fit_report(result.params)
        yout2d, peak2d1,peak2d2 = residual2(result.params,x2d) # the different peaks

        #save results in a list
        for name in result.params.valuesdict():
            best_para=best_para+[result.params[name].value]


# #         ############################## PLOTTING FULL RAMAN ############################
        fig = plb.figure(figsize=(12, 6)) 
        gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])

        ax0 = plb.subplot(gs[0])
        ax0.plot(xg, yg,'k', xg, youtg, 'r--', xg, peakg2, 'g:',xg, peakg3, 'g:', xg, peakg1, 'g:',  xg, peakg4, 'g:')
        plb.title('G and D peaks of flake %1.0f' %e, loc='right')
        ax0.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        plb.xlabel(r'$\omega$ (cm$^{-1}$)', fontsize=10)
        plb.ylabel('Intensity (arb.units)', fontsize=10)
        ax0.grid(True, which='major', ls='-', alpha=0.1)

        ax1 = plb.subplot(gs[1])
        ax1.plot(x2d, y2d,'k', x2d, yout2d, 'r--', x2d, peak2d1, 'g:', x2d, peak2d2, 'g:')
        plb.title("2D and D\'+D peaks of flake %1.0f" %e, loc='right')
        ax1.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        plb.xlabel(r'$\omega$ (cm$^{-1}$)', fontsize=10) 
        plb.ylabel('Intensity (arb.units)', fontsize=10)
        ax1.grid(True, which='major', ls='-', alpha=0.1)

        # save the figure to file
        plb.tight_layout()
        plb.savefig(file_base + str(e) + fig_suffix, format='png', dpi=300)
        #plb.show(block=False)  ## show plot ##
        plb.close()


        #saving fitting parameters
        allinall.append(best_para) #get values and sandart derivation

        ####################################################################
        ############################## SAVING FILES ###########################

        #saving normalised spectra
        spectrum = np.array((x, y)).T
        np.savetxt('spectrum'+str(e)+file_suffix, spectrum,fmt='%f', delimiter='\t', newline='\n') #


np.savetxt('allinall_tpa.csv', allinall, fmt = '%s', delimiter=',', newline='\n', 
           header = 'Flake,status,D_Intensity,D_Position,D_FWHM,G_Intensity,G_Position,G_FWHM,D\'_Intensity,D\'_Position,'+
		   'D\'_FWHM,tpa_Intensity,tpa_Position,tpa_FWHM,2D_Intensity,2D_Position,2D_FWHM,2D\'_Intensity,2D\'_Position,2D\'_FWHM')





