# NTC overall simulation and study

import numpy as np
import scipy.constants as spc
import scipy.optimize as spo
import scipy.special as sps
import matplotlib.pyplot as plt

class Bipole:
    def __init__(self):
        self.par = {}
        self.temperature = 0
    def r_model(self, ix):
        pass
    def g_model(self, vx):
        pass
    def thermal_model(self, vx, ix):
        pass

class Diode(Bipole):
    def __init__(self, Is=1e-15, Eta=1, Rth=100, Tdef=300, Model_type="pure_exp"):
        super().__init__()
        self.par['Is'] = Is
        self.par['Eta'] = Eta
        self.par['Rth'] = Rth
        self.par['Model_type'] = Model_type
        self.set_temperature(Tdef)
    def set_temperature(self, temperature):
        self.temperature = temperature
        # Compute thermal voltage according to current temperature
        self.Vt = spc.Boltzmann * self.temperature/spc.e
    def r_model(self, id):
        # Implement various models
        match self.par['Model_type']:
            case "pure_exp":
                return self.par['Eta'] * self.Vt * np.log(id/self.par['Is'])
            case "negative_saturation":
                return self.par['Eta'] * self.Vt * np.log(1.0 + id/self.par['Is'])
            case _:
                raise ValueError("Unimplemented diode model requested")
    def g_model(self, vd):
        # Implement various models
        match self.par['Model_type']:
            case "pure_exp":
                return self.par['Is'] * np.exp(vd/self.par['Eta']/self.Vt)
            case "negative_saturation":
                return self.par['Is'] * np.exp(vd/self.par['Eta']/self.Vt - 1.0)
            case _:
                raise ValueError("Unimplemented diode model requested")

class NTC(Bipole):
    def __init__(self, R0=10e3, Beta=3780, T0=spc.convert_temperature(25, 'Celsius', 'Kelvin'), Model_type='beta_value'):
        super().__init__()
        self.par['R0'] = R0
        self.par['Beta'] = Beta
        self.par['T0'] = T0
        self.par['Model_type'] = Model_type
        self.temperature = T0
    def r_model(self, ix, temperature=None):
        return self.r_value(temperature) * ix
    def g_model(self, vx, temperature=None):
        return 1/self.r_value(temperature) * vx
    def r_value(self, temperature=None):
        if np.all(temperature == None):
            temp_now = self.temperature
        else:
            temp_now = temperature
        match self.par['Model_type']:
            case "beta_value":
                return self.par['R0'] * np.exp(self.par['Beta'] * (1/temp_now - 1/self.par['T0']))
            case "steinhart_hart":
                x = 1/self.par['C'] * (self.par['A'] - 1/temp_now)
                y = np.sqrt((self.par['B']/(3*self.par['C']))**3 + x**2/4)
                return np.exp((y - x/2)**(1/3) - (y + x/2)**(1/3))
            case "polynomial":
                return self.par['R0'] * np.exp(np.polyval(self.par['DCBA'], 1/temp_now))
            case _:
                raise ValueError("Unimplemented NTC model requested")

# Simulation parameters
class NTC_simulation:
    def __init__(self):
        self.Tmin_deg = -50
        self.Tmax_deg = 150
        self.Nsim = 1001
        self.Vbias = 3.3
        self.Nj = 4
        self.diode0 = Diode(Is = 25e-9, Eta = 1.6)
        self.ntc0 = NTC()
        self.Tx = np.linspace(self.Tmin_deg, self.Tmax_deg, self.Nsim, endpoint=True)
        self.Tx = spc.convert_temperature(self.Tx, 'Celsius', 'Kelvin')
    def get_temperatures(self):
        return self.Tx
    def analysis_ideal_diode(self):
        vx = self.Nj * (self.diode0.r_model(self.Vbias/self.ntc0.par['R0']) - 
                        self.diode0.par['Eta'] * self.diode0.Vt * np.log(self.ntc0.r_value(self.Tx)/self.ntc0.par['R0']))
        ix = self.diode0.g_model(vx/self.Nj)
        return (vx, ix)
    def analysis_divider(self):
        vx = self.Vbias - self.Nj * self.diode0.par['Eta'] * self.diode0.Vt * sps.lambertw(
            self.ntc0.r_value(self.Tx)/(self.Nj * self.diode0.par['Eta'] * self.diode0.Vt) * 
            self.diode0.g_model(self.Vbias/self.Nj)
        )
        ix = self.diode0.g_model(vx/self.Nj)
        return (vx, ix)
    def sim_ideal_diode(self):
        # Ideal diode (diode voltage not affecting NTC voltage)
        ix = self.ntc0.g_model(self.Vbias, self.Tx)
        vx = self.Nj * self.diode0.r_model(ix)
        return (vx, ix) # output voltage and diodes current
    def sim_divider(self):
        # Asymmetrical voltage divider
        # cir = lambda ix: Vbias - ntc0.r_model(ix, Tx) - Nj*diode0.r_model(ix) # KVL
        cir = lambda vx: self.ntc0.g_model(self.Vbias - vx, self.Tx) - self.diode0.g_model(vx/self.Nj) # KCL
        vx = spo.fsolve(cir, np.ones_like(self.Tx))
        ix = self.diode0.g_model(vx/self.Nj)
        return (vx, ix)

# Simulation
mysim = NTC_simulation()
Tx = mysim.get_temperatures()
vout = np.empty((mysim.Nsim, 4))
intc = np.empty_like(vout)
vout[:,0], intc[:,0] = mysim.sim_ideal_diode()
vout[:,1], intc[:,1] = mysim.sim_divider()
vout[:,2], intc[:,2] = mysim.analysis_ideal_diode()
vout[:,3], intc[:,3] = mysim.analysis_divider()

# Results
plt.figure(1)
plt.plot(vout, 1/Tx)
plt.xlabel('Output voltage (V)')
plt.ylabel('Temperature reciprocal (1/K)')
plt.grid()
plt.show(block=False)

plt.figure(2)
plt.plot(vout, Tx)
plt.xlabel('Output voltage (V)')
plt.ylabel('Temperature (K)')
plt.grid()
plt.show(block=True)

print('Ciao')