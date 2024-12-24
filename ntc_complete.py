# NTC overall simulation and study

import numpy as np
import scipy.constants as spc
import scipy.optimize as spo
import scipy.special as sps
import matplotlib.pyplot as plt

class Bipole:
    def __init__(self):
        self.par = {}
        self.T_base = 0.0
    def r_model(self, ix, Tx=None, model="default"):
        pass
    def g_model(self, vx, Tx=None, model="default"):
        pass
    def thermal_model(self, v_or_i, input, Tx):
        if input == "current":
            ix = v_or_i
            p_loss = self.r_model(ix, Tx) * ix
        elif input == "voltage":
            vx = v_or_i
            p_loss = self.g_model(vx, Tx) * vx
        else:
            raise ValueError("Invalid model requested")
        # We are not computing the temperature, but we are writing an equation that must be solved
        return self.T_base + self.par['Rth'] * p_loss - Tx

class Diode(Bipole):
    def __init__(self, Is=1e-15, Eta=1, Rth=100, T_base=300, Model_type="pure_exp"):
        super().__init__()
        self.par['Is'] = Is
        self.par['Eta'] = Eta
        self.par['Rth'] = Rth
        self.par['Model_type'] = Model_type
        self.par['Eg'] = 1.12 * spc.eV # TODO: make it a parameter
        self.T_base = T_base
    def __update_TSEP(self, Tx):
        # Compute temperature sensitive parameters according to the specified temperature
        # If temperature not specified, use the one stored in the object
        if Tx is None:
            Tx = self.T_base
        self.Vt = spc.Boltzmann * Tx/spc.e
        self.Is = self.par['Is'] * np.exp(-self.par['Eg']/spc.Boltzmann/Tx)
    def r_model(self, ix, Tx=None, model="default"):
        self.__update_TSEP(Tx)
        # Implement various models
        # default can be overriden by the user to reuse expressions
        # Otherwise set by Model_type
        if model == "default":
            model = self.par['Model_type']
        match model:
            case "pure_exp":
                return self.par['Eta'] * self.Vt * np.log(ix/self.par['Is'])
            case "negative_saturation":
                return self.par['Eta'] * self.Vt * np.log(1.0 + ix/self.par['Is'])
            case _:
                raise ValueError("Unimplemented diode model requested")
    def g_model(self, vx, Tx=None, model="default"):
        if model == "default":
            model = self.par['Model_type']
        match model:
            case "pure_exp":
                return self.par['Is'] * np.exp(vx/self.par['Eta']/self.Vt)
            case "negative_saturation":
                return self.par['Is'] * np.exp(vx/self.par['Eta']/self.Vt - 1.0)
            case _:
                raise ValueError("Unimplemented diode model requested")

class NTC(Bipole):
    def __init__(self, R0=10e3, Beta=3780, T0=spc.convert_temperature(25, 'Celsius', 'Kelvin'), Model_type='beta_value'):
        super().__init__()
        self.par['R0'] = R0
        self.par['Beta'] = Beta
        self.par['T0'] = T0
        self.par['Model_type'] = Model_type
        self.T_base = T0
    def r_model(self, ix, Tx=None, model="default"):
        return self.r_value(Tx, model) * ix
    def g_model(self, vx, Tx=None, model="default"):
        return 1/self.r_value(Tx, model) * vx
    def r_value(self, Tx=None, model="default"):
        if Tx is None:
            Tx = self.T_base
        if model == "default":
            model = self.par['Model_type']
        match model:
            case "beta_value":
                return self.par['R0'] * np.exp(self.par['Beta'] * (1/Tx - 1/self.par['T0']))
            case "steinhart_hart":
                x = 1/self.par['C'] * (self.par['A'] - 1/Tx)
                y = np.sqrt((self.par['B']/(3*self.par['C']))**3 + x**2/4)
                return np.exp((y - x/2)**(1/3) - (y + x/2)**(1/3))
            case "polynomial":
                return self.par['R0'] * np.exp(np.polyval(self.par['DCBA'], 1/Tx))
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