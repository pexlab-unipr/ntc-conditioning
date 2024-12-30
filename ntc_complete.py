# NTC overall simulation and study

import numpy as np
import scipy.constants as spc
import scipy.optimize as spo
import scipy.special as sps
import matplotlib.pyplot as plt

class Bipole:
    def __init__(self):
        self.par = {}
    def r_model(self, ix, Tx, model="default"):
        pass
    def g_model(self, vx, Tx, model="default"):
        pass
    def thermal_model(self, v_or_i, input, Tstart, Tend, enable_selfheating=True):
        if enable_selfheating:
            Rth = self.par['Rth']
        else:
            Rth = 0
        if input == "current":
            ix = v_or_i
            p_loss = self.r_model(ix, Tend) * ix
        elif input == "voltage":
            vx = v_or_i
            p_loss = self.g_model(vx, Tend) * vx
        else:
            raise ValueError("Invalid model requested")
        # We are not computing the temperature, but we are writing an equation that must be solved
        # Tstart is the known starting point, Tend is the supposed actual temperature
        return Tstart + Rth * p_loss - Tend

class Diode(Bipole):
    def __init__(self, Is=1e-15, Eta=1, Rth=100, Model_type="pure_exp"):
        super().__init__()
        self.par['Is'] = Is
        self.par['Eta'] = Eta
        self.par['Rth'] = Rth
        self.par['Model_type'] = Model_type
        self.par['Eg'] = 1.12 * spc.eV # TODO: make it a parameter to account for non-silicon devices
    # Compute temperature sensitive parameters according to the specified temperature
    def Vt(self, Tx):
        return spc.Boltzmann * Tx/spc.e
    def Is(self, Tx):
        return self.par['Is'] * np.exp(-self.par['Eg']/spc.Boltzmann * (1/Tx - 1/300)) # TODO parameterize reference temperature for given saturation current
    def r_model(self, ix, Tx, model="default"):
        # Implement various models
        # default can be overriden by the user to reuse expressions
        # Otherwise set by Model_type
        if model == "default":
            model = self.par['Model_type']
        match model:
            case "pure_exp":
                return self.par['Eta'] * self.Vt(Tx) * np.log(ix/self.Is(Tx))
            case "negative_saturation":
                return self.par['Eta'] * self.Vt(Tx) * np.log(1.0 + ix/self.Is(Tx))
            case _:
                raise ValueError("Unimplemented diode model requested")
    def g_model(self, vx, Tx, model="default"):
        if model == "default":
            model = self.par['Model_type']
        match model:
            case "pure_exp":
                return self.Is(Tx) * np.exp(vx/self.par['Eta']/self.Vt(Tx))
            case "negative_saturation":
                return self.Is(Tx) * np.exp(vx/self.par['Eta']/self.Vt(Tx) - 1.0)
            case _:
                raise ValueError("Unimplemented diode model requested")

class NTC(Bipole):
    def __init__(self, R0=10e3, Beta=3780, T0=spc.convert_temperature(25, 'Celsius', 'Kelvin'), Rth= 300, Model_type='beta_value'):
        super().__init__()
        self.par['R0'] = R0
        self.par['Beta'] = Beta
        self.par['T0'] = T0
        self.par['Rth'] = Rth
        self.par['Model_type'] = Model_type
    def r_model(self, ix, Tx, model="default"):
        return self.r_value(Tx, model) * ix
    def g_model(self, vx, Tx, model="default"):
        return 1/self.r_value(Tx, model) * vx
    def r_value(self, Tx, model="default"):
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
        self.T_min_deg = -50
        self.T_max_deg = 150
        self.T_base_deg = 25
        self.N_sim = 1001
        self.V_bias = 3.3
        self.N_j = 4
        self.diode0 = Diode(Is = 25e-9, Eta = 1.6, Model_type="negative_saturation")
        self.ntc0 = NTC()
        self.T_eval = np.linspace(self.T_min_deg, self.T_max_deg, self.N_sim, endpoint=True)
        self.T_eval = spc.convert_temperature(self.T_eval, 'Celsius', 'Kelvin')
        self.T_base = spc.convert_temperature(self.T_base_deg, 'Celsius', 'Kelvin')
    def get_temperatures(self):
        return (self.T_eval, spc.convert_temperature(self.T_eval, 'Kelvin', 'Celsius'))
    def analysis_ideal_diode(self):
        vx = self.N_j * (self.diode0.r_model(self.V_bias/self.ntc0.par['R0'], self.T_base, model="pure_exp") - 
                        self.diode0.par['Eta'] * self.diode0.Vt(self.T_base) * np.log(self.ntc0.r_value(self.T_eval, model="beta_value")/self.ntc0.par['R0']))
        ix = self.diode0.g_model(vx/self.N_j, self.T_base, model="pure_exp")
        return (vx, ix)
    def analysis_divider(self):
        vx = self.V_bias - self.N_j * self.diode0.par['Eta'] * self.diode0.Vt(self.T_base) * sps.lambertw(
            self.ntc0.r_value(self.T_eval, model="beta_value")/(self.N_j * self.diode0.par['Eta'] * self.diode0.Vt(self.T_base)) * 
            self.diode0.g_model(self.V_bias/self.N_j, self.T_base, model="pure_exp")
        )
        ix = self.diode0.g_model(vx/self.N_j, self.T_base, model="pure_exp")
        return (vx, ix)
    def sim_ideal_diode(self):
        # Ideal diode (diode voltage not affecting NTC voltage)
        ix = self.ntc0.g_model(self.V_bias, self.T_eval)
        vx = self.N_j * self.diode0.r_model(ix, self.T_base)
        return (vx, ix) # output voltage and diodes current
    def sim_divider(self, NTC_selfheat=False, Diode_selfheat=False):
        # Asymmetrical voltage divider
        # cir = lambda ix: Vbias - ntc0.r_model(ix, Tx) - Nj*diode0.r_model(ix) # KVL
        eq_circuit  = lambda vx, T_NTC, T_diode: self.ntc0.g_model(self.V_bias - vx, T_NTC) - self.diode0.g_model(vx/self.N_j, T_diode) # KCL
        eq_ntc_sh   = lambda vx, T_NTC, T_NTC0: self.ntc0.thermal_model(self.V_bias - vx, "voltage", T_NTC0, T_NTC, NTC_selfheat)
        eq_diode_sh = lambda vx, T_diode, T_diode0: self.diode0.thermal_model(vx/self.N_j, "voltage", T_diode0, T_diode, Diode_selfheat)
        eqs_system  = lambda vx, T_NTC, T_diode, T_NTC0, T_diode0: np.array([
            eq_circuit(vx, T_NTC, T_diode), 
            eq_ntc_sh(vx, T_NTC, T_NTC0), 
            eq_diode_sh(vx, T_diode, T_diode0)
            ])
        x = np.vectorize(
            lambda T_meas: spo.fsolve(
                lambda x: eqs_system(
                    x[0], x[1], x[2], T_meas, self.T_base), 
                [0.5*self.N_j, T_meas, self.T_base], 
                diag=[0.5*self.N_j, T_meas, self.T_base]), 
            signature='()->(3)')(self.T_eval)
        vx = x[:,0]
        T_NTC = x[:,1]
        T_diode = x[:,2]
        ix = self.diode0.g_model(vx/self.N_j, T_diode)
        return (vx, ix, T_NTC, T_diode)

# Simulation
mysim = NTC_simulation()
Tx, Tx_deg = mysim.get_temperatures()
v_out = np.empty((mysim.N_sim, 4))
i_ntc = np.empty_like(v_out)
T_ntc = np.empty_like(v_out)
T_diode = np.empty_like(v_out)
v_out[:,0], i_ntc[:,0] = mysim.sim_ideal_diode()
v_out[:,1], i_ntc[:,1], T_ntc[:,1], T_diode[:,1] = mysim.sim_divider(NTC_selfheat=True, Diode_selfheat=True)
v_out[:,2], i_ntc[:,2] = mysim.analysis_ideal_diode()
v_out[:,3], i_ntc[:,3] = mysim.analysis_divider()

# Results
plt.figure(1)
plt.plot(v_out, 1/Tx, label=['ideal diode (sim)', 'divider (sim)', 'ideal diode (model)', 'divider (model)'])
plt.xlabel('Output voltage (V)')
plt.ylabel('Temperature reciprocal (1/K)')
plt.legend()
plt.grid()
plt.show(block=False)

plt.figure(2)
plt.plot(v_out, Tx)
plt.xlabel('Output voltage (V)')
plt.ylabel('Temperature (K)')
plt.grid()
plt.show(block=False)

plt.figure(3)
plt.plot(Tx_deg, T_ntc[:,1] - Tx, label='NTC')
plt.plot(Tx_deg, T_diode[:,1] - mysim.T_base, label='diode')
plt.xlabel('Measured temperature (°C)')
plt.ylabel('Component overtemperature (K)')
plt.legend()
plt.grid()
plt.show(block=True)

print('Ciao')