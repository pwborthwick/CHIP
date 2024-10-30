import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

import warnings
warnings.filterwarnings("ignore")

class hubble:

    def __init__(self):

        self.uc = {'c':299792458 ,'rho_c': 7.64e-10, 'H0':67.3e3,
                   'omega_m': 0.315, 'omega_r': 4.17e-14, 'omega_l':0.685,
                   'omega_k': 0.0}
        self.to = {'Mpc->m': 3.08567758e22, 'Gly->m':9.4605e24, 'y->s': 31556926 , 'G':1e9}
        self.uc['H0'] /= self.to['Mpc->m']
        self.uc['omega_r'] /= self.uc['rho_c']

        self.zero = 1e-50
        self.radial = +1
        self.f, self.lims, self.at, self.initial = self.cosmic_time, (), (), [0]
        self.mode = 'proper'
        self.horizons = {'event':0.0, 'particle':0.0, 'hubble':0.0}
        
        self.set_parameters()
        
    def set_parameters(self, a_span=range(1,10)):
        
        self.lims, self.at = (self.zero, 100), a_span
        epochs = self.integrator()
        scale_conversion = (list(epochs.y[0]/self.to['y->s']/self.to['G']))
        scale_conversion.insert(0, 0)

        self.a_scale = scale_conversion
        self.now = scale_conversion[1]

        self.f, self.lims, self.at = self.comoving_distance, (self.zero, 1), [1]
        rad = self.integrator()
        self.radius_now = rad.y[0][0]/self.to['Gly->m']
       
        return 
    
    def set_radial(self, direction='outward'):
        self.radial = +1 if direction == 'outward' else -1
        
    def hubble_parameter(self, a):
        return self.uc['H0']*np.sqrt(self.uc['omega_r']*a**-4 +
                                     self.uc['omega_m']*a**-3 +
                                     self.uc['omega_l']*a**0)

    def cosmic_time(self, a, t):
        H = self.hubble_parameter(a)
        return 1/(a*H)

    def comoving_distance(self, a, d):
        H = self.hubble_parameter(a) 
        return self.uc['c']/(a**2*H)*self.radial

    def integrator(self):
        value = solve_ivp(fun=self.f, t_span=self.lims, t_eval=self.at, y0=self.initial)
        return value


def light_cone(fig, ax, h):

    h.f = h.cosmic_time

    h.lims, h.at, h.initial = (h.zero, 9), np.linspace(h.zero, 9, 1000), [0]
    t_axis = h.integrator()
    t_axis = t_axis.y[0]/h.to['y->s']/h.to['G']

    h.set_radial(direction='inward')
    h.f = h.comoving_distance
    h.initial = [h.radius_now*h.to['Gly->m']]
    x_axis = h.integrator()
    h.set_radial(direction='outward')
    
    rp = x_axis.y[0] / h.to['Gly->m']
    if h.mode == 'proper': rp *= h.at
    
    ax.plot(rp, t_axis, linestyle='solid',  color='cyan', label='light cone') 
    ax.plot(-rp, t_axis, linestyle='solid',  color='cyan') 
    
def particle_horizon(fig, ax, h):

    h.f = h.cosmic_time
    h.lims, h.at, h.initial = (h.zero, 9), np.linspace(h.zero, 9, 1000), [0]

    t_axis = h.integrator().y[0]/h.to['y->s']/h.to['G']
    
    h.f = h.comoving_distance
    x_axis = h.integrator().y[0]/h.to['Gly->m']

    if h.mode == 'proper': x_axis *= h.at
    ax.plot(x_axis,  t_axis, linestyle='dotted', color='blue', label='particle horizon')
    ax.plot(-x_axis, t_axis, linestyle='dotted', color='blue') 
    ax.axvline(h.radius_now, 0, 1, linestyle='dashed', color='gray', label='')
    points = [[0, 0, h.radius_now, -h.radius_now],[0, h.now, h.now, h.now]]
    ax.scatter(points[0], points[1], color='cyan', zorder=10)

    h.horizons['particle'] = h.radius_now
    
def event_horizon(fig, ax, h):


    a = np.linspace(h.zero, 9, 1000)
    d = []
    t = []
    for x in a:

        h.f = h.comoving_distance
        h.lims, h.at, h.initial = (x,1000), [x], [0]
        p = solve_ivp(fun=h.f, t_span=h.lims, eval=h.at,y0=h.initial)

        h.f, h.lims = h.cosmic_time, (h.zero, x)
        q = solve_ivp(fun=h.f, t_span=h.lims, eval=h.at, y0=h.initial)

        if h.mode == 'proper':
            d.append(x*p.y[0][-1]/h.to['Gly->m'])
        else:
            d.append(p.y[0][-1]/h.to['Gly->m'])
        t.append(q.y[0][-1]/h.to['y->s']/h.to['G'])

    d = np.array(d)
    t = np.array(t)

    h.set_radial(direction='inward')
    h.f, h.lims, h.at = h.comoving_distance, (1, 1000), [0]
    p = solve_ivp(fun=h.f, t_span=h.lims, eval=h.at,y0=h.initial)
    h.set_radial(direction='outward')
    
    ax.axvline(-p.y[0][-1]/h.to['Gly->m'], 0, 1, linestyle='dashed', color='gray', label='')
    ax.fill_betweenx(t, d, -d, color='y',alpha=0.5)
    ax.plot(d,  t, color='r', linestyle='dashed', label='Event Horizon')
    ax.plot(-d, t, color='r', linestyle='dashed')

    h.horizons['event'] = -p.y[0][-1]/h.to['Gly->m']
    
def hubble_radius(fig, ax, h):

    a = np.logspace(np.log10(h.zero),np.log10(9),int(1000))
       
    # co-moving distance
    H = h.hubble_parameter(a)

    d = h.uc['c']/H/h.to['Gly->m']

    #proper distance
    if h.mode == 'co-moving': d = d/a

    h.f, h.lims, h.at, h.initial = h.cosmic_time, (h.zero, 9), a, [0]
    t_axis = h.integrator().y[0]/h.to['y->s']/h.to['G']
    
    ax.axvline(h.uc['c']/h.hubble_parameter(1)/h.to['Gly->m'], 0, 1, linestyle='dashed', color='gray', label='')

    ax.fill_betweenx(t_axis, d, -d, color='w',alpha=0.5)
    ax.plot(d,  t_axis, color='w')
    ax.plot(-d, t_axis, color='w', label='Hubble Radius')

    h.horizons['hubble'] = h.uc['c']/h.hubble_parameter(1)/h.to['Gly->m']

plt.style.use('dark_background')
fig, ax1 = plt.subplots(figsize=(1,1))

h = hubble()

h.mode = 'co-moving'
light_cone(fig, ax1, h)
particle_horizon(fig, ax1, h)

event_horizon(fig, ax1, h)
hubble_radius(fig, ax1, h)

plt.title('Cosmic Horizons', loc='left')
ax1.axhline(h.now, 0, 1, linestyle='dashed', color='gray', label='')
ax1.axvline(0,    0, 1, linestyle='dotted', color='gray')   

ax1.set_xlim([-60, 60])
ax1.set_ylim(0,50)

ax2 = ax1.secondary_yaxis('right')
ax2.set_yticks(h.a_scale,[0,1,2,3,4,5,6,7,8,9])
ax2.set_ylabel('Scale Factor (a)')
x_text = 'Proper Distance (Gly)' if h.mode == 'proper' else 'Comoving Distance (Gly)'
ax1.set_xlabel(x_text)
ax1.set_ylabel('Cosmic Time (Gyr)')
ax1.set_xticks(range(-50,51,10))
ax1.legend(shadow=True, fancybox=True)

plt.show()
