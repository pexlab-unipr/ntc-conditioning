# NTC overall simulation and study

import numpy as np
import scipy.constants as spc
import scipy.optimize as spo
import scipy.special as sps
import matplotlib.pyplot as plt

def curvature(x, y):
    # Compute the curvature of a curve defined by x and y
    yp = np.gradient(y, x)
    ypp = np.gradient(yp, x)
    return np.abs(ypp) / (1 + yp**2)**(3/2)

def curvature2(x, y):
    # Compute the first derivatives
    dx = np.diff(x, prepend=x[0])
    dy = np.diff(y, prepend=y[0])
    yp = dy / dx
    # Compute the second derivatives
    dyp = np.diff(yp, prepend=yp[0])
    ypp = dyp / dx
    # Compute the curvature
    curvature = np.abs(ypp) / (1 + yp**2)**(3/2)
    return curvature

def fake_curvature(x, y):
    # Not a real curvature, but a measure of the slope of the curve
    yp = np.diff(y, axis=0) / np.diff(x, axis=0) #np.gradient(y, x)
    return yp

def multi_curvature(x, y):
    xb, yb = np.broadcast_arrays(x, y[:,np.newaxis])
    res = np.array([curvature2(xb[:,i], yb[:,i]) for i in range(yb.shape[1])])
    return res.T

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
    def __init__(self, R0=10e3, Beta=3988, T0=spc.convert_temperature(25, 'Celsius', 'Kelvin'), Rth=333, Model_type='beta_value'):
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
    def __init__(self, V_bias, N_j, diode, ntc, name="NTC_simulation"):
        self.T_min_deg = -50
        self.T_max_deg = 150
        self.T_base_deg = 25
        self.N_sim = 101
        self.V_bias = V_bias
        self.N_j = N_j
        self.diode = diode
        self.ntc = ntc
        self.name = name
        self.T_eval = np.linspace(self.T_min_deg, self.T_max_deg, self.N_sim, endpoint=True)
        self.T_eval = spc.convert_temperature(self.T_eval, 'Celsius', 'Kelvin')
        self.T_base = spc.convert_temperature(self.T_base_deg, 'Celsius', 'Kelvin')
    def get_temperatures(self):
        return (self.T_eval, spc.convert_temperature(self.T_eval, 'Kelvin', 'Celsius'))
    def analysis_ideal_diode(self):
        vx = self.N_j * (self.diode.r_model(self.V_bias/self.ntc.par['R0'], self.T_base, model="pure_exp") - 
                        self.diode.par['Eta'] * self.diode.Vt(self.T_base) * np.log(self.ntc.r_value(self.T_eval, model="beta_value")/self.ntc.par['R0']))
        ix = self.diode.g_model(vx/self.N_j, self.T_base, model="pure_exp")
        return (vx, ix)
    def analysis_divider(self):
        vx = self.V_bias - self.N_j * self.diode.par['Eta'] * self.diode.Vt(self.T_base) * sps.lambertw(
            self.ntc.r_value(self.T_eval, model="beta_value")/(self.N_j * self.diode.par['Eta'] * self.diode.Vt(self.T_base)) * 
            self.diode.g_model(self.V_bias/self.N_j, self.T_base, model="pure_exp")
        )
        ix = self.diode.g_model(vx/self.N_j, self.T_base, model="pure_exp")
        return (vx, ix)
    def sim_ideal_diode(self):
        # Ideal diode (diode voltage not affecting NTC voltage)
        ix = self.ntc.g_model(self.V_bias, self.T_eval)
        vx = self.N_j * self.diode.r_model(ix, self.T_base)
        return (vx, ix) # output voltage and diodes current
    def sim_divider(self, NTC_selfheat=False, Diode_selfheat=False):
        # Asymmetrical voltage divider
        # cir = lambda ix: Vbias - ntc0.r_model(ix, Tx) - Nj*diode0.r_model(ix) # KVL
        eq_circuit  = lambda vx, T_NTC, T_diode: self.ntc.g_model(self.V_bias - vx, T_NTC) - self.diode.g_model(vx/self.N_j, T_diode) # KCL
        eq_ntc_sh   = lambda vx, T_NTC, T_NTC0: self.ntc.thermal_model(self.V_bias - vx, "voltage", T_NTC0, T_NTC, NTC_selfheat)
        eq_diode_sh = lambda vx, T_diode, T_diode0: self.diode.thermal_model(vx/self.N_j, "voltage", T_diode0, T_diode, Diode_selfheat)
        eqs_system  = lambda vx, T_NTC, T_diode, T_NTC0, T_diode0: np.array([
            eq_circuit(vx, T_NTC, T_diode), 
            eq_ntc_sh(vx, T_NTC, T_NTC0), 
            eq_diode_sh(vx, T_diode, T_diode0)
            ])
        x = np.vectorize(
            lambda T_meas: spo.fsolve(
                lambda x: eqs_system(
                    x[0], x[1], x[2], T_meas, self.T_base), 
                [0.75*self.N_j, T_meas, self.T_base], 
                diag=[0.75*self.N_j, T_meas, self.T_base]), 
            signature='()->(3)')(self.T_eval)
        vx = x[:,0]
        T_NTC = x[:,1]
        T_diode = x[:,2]
        ix = self.diode.g_model(vx/self.N_j, T_diode)
        return (vx, ix, T_NTC, T_diode)

class NTC_characterization:
    def __init__(self, ntc):
        self.ntc = ntc
        self.I_max = 1000
        self.N_sim = 1001
        self.i = np.logspace(-6, np.log10(self.I_max), self.N_sim, endpoint=True)
    def simulate(self, T_meas):
        eq = lambda Tx, ix: self.ntc.thermal_model(ix, "current", T_meas, Tx, enable_selfheating=True)
        T_end = np.vectorize(
            lambda ix: spo.fsolve(eq, T_meas, args=(ix,)))(self.i)
        R_out = self.ntc.r_value(T_end)
        v_out = self.ntc.r_model(self.i, T_end)
        return (v_out, self.i, R_out, T_end)
    def analyze(self, T_meas):
        v_out = self.ntc.r_model(self.i, T_meas)
        R_m = self.ntc.r_value(T_meas)
        I_m = np.sqrt(T_meas**2/(self.ntc.par['Beta'] * self.ntc.par['Rth'] * R_m))
        V_m = T_meas/2 * np.sqrt(R_m/(self.ntc.par['Beta'] * self.ntc.par['Rth']))
        return (v_out, self.i, R_m, I_m, V_m)
    def attempt(self, T_meas):
        # Opening the loop works worse than linearizing the NTC model
        T_end = T_meas + self.ntc.par['Rth'] * self.i**2 * self.ntc.r_value(T_meas)
        R_out = self.ntc.r_value(T_end)
        v_out = self.ntc.r_model(self.i, T_end)
        return (v_out, self.i, R_out, T_end)

# Simulation
diode_1N4148 = Diode(Is =  2.7e-9, Eta = 1.8, Rth = 350, Model_type="negative_saturation")
diode_1N4007 = Diode(Is = 500e-12, Eta = 1.5, Rth =  93, Model_type="negative_saturation")
diode_BC817  = Diode(Is =  20e-15, Eta = 1.0, Rth = 160, Model_type="negative_saturation")
ntc_B57703M_10k = NTC()
mysim = NTC_simulation(3.3, 1, diode_BC817, ntc_B57703M_10k, name="BC817")
mychar = NTC_characterization(ntc_B57703M_10k)
Tx, Tx_deg = mysim.get_temperatures()
v_out = np.empty((mysim.N_sim, 4))
i_ntc = np.empty_like(v_out)
T_ntc = np.empty_like(v_out)
T_diode = np.empty_like(v_out)
v_out[:,0], i_ntc[:,0] = mysim.sim_ideal_diode()
v_out[:,1], i_ntc[:,1], T_ntc[:,1], T_diode[:,1] = mysim.sim_divider(NTC_selfheat=True, Diode_selfheat=True)
v_out[:,2], i_ntc[:,2] = mysim.analysis_ideal_diode()
v_out[:,3], i_ntc[:,3] = mysim.analysis_divider()
v_n, i_n, R_n, T_n = mychar.simulate(400)
v_m, i_m, R_m, I_m, V_m = mychar.analyze(400)
v_a, i_a, R_a, T_a = mychar.attempt(400)

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
plt.show(block=False)

plt.figure(4)
# plt.plot(v_out[0:-1,:], fake_curvature(v_out, 1/Tx[:,np.newaxis]), label=['ideal diode (sim)', 'divider (sim)', 'ideal diode (model)', 'divider (model)'])
plt.plot(v_out, multi_curvature(v_out, 1/Tx), label=['ideal diode (sim)', 'divider (sim)', 'ideal diode (model)', 'divider (model)'])
plt.xlabel('Output voltage (V)')
plt.ylabel('Curvature')
plt.legend()
plt.grid()
plt.show(block=False)

plt.figure(5)
plt.subplot(3, 1, 1)
plt.loglog(i_n * 1e3, v_n)
plt.loglog(i_m * 1e3, v_m)
plt.loglog(I_m * 1e3, V_m, 'x')
plt.loglog(i_a * 1e3, v_a)
plt.xlabel('Test current (mA)')
plt.ylabel('NTC voltage (V)')
plt.grid()
plt.subplot(3, 1, 2)
plt.loglog(i_n * 1e3, T_n)
plt.xlabel('Test current (mA)')
plt.ylabel('NTC temperature (K)')
plt.grid()
plt.subplot(3, 1, 3)
plt.loglog(i_n * 1e3, R_n)
plt.loglog(i_n * 1e3, ntc_B57703M_10k.r_value(400) * np.ones_like(i_n))
plt.loglog(i_n * 1e3, ntc_B57703M_10k.r_value(1e6) * np.ones_like(i_n))
plt.xlabel('Test current (mA)')
plt.ylabel('NTC resistance (ohm)')
plt.grid()
plt.show(block=False)

x = np.log(i_n)
y = np.log(v_n)
y = np.diff(y) / np.diff(x)
x = x[0:-1]
plt.figure(6)
plt.plot(x, y)
plt.show(block=True)

print('Ciao')