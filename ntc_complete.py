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
    def thermal_model(self, v_or_i, input, Tstart, Tend):
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
        return Tstart + self.par['Rth'] * p_loss - Tend

class Diode(Bipole):
    def __init__(self, Is=1e-15, Eta=1, Rth=100, Model_type="pure_exp"):
        super().__init__()
        self.par['Is'] = Is
        self.par['Eta'] = Eta
        self.par['Rth'] = Rth
        self.par['Model_type'] = Model_type
        self.par['Eg'] = 1.12 * spc.eV # TODO: make it a parameter to account for non-silicon devices
    def __update_TSEP(self, Tx):
        # Compute temperature sensitive parameters according to the specified temperature
        # If temperature not specified, use the one stored in the object
        self.Vt = spc.Boltzmann * Tx/spc.e
        self.Is = self.par['Is'] * np.exp(-self.par['Eg']/spc.Boltzmann * (1/Tx - 1/300))
    def r_model(self, ix, Tx, model="default"):
        self.__update_TSEP(Tx)
        # Implement various models
        # default can be overriden by the user to reuse expressions
        # Otherwise set by Model_type
        if model == "default":
            model = self.par['Model_type']
        match model:
            case "pure_exp":
                return self.par['Eta'] * self.Vt * np.log(ix/self.Is)
            case "negative_saturation":
                return self.par['Eta'] * self.Vt * np.log(1.0 + ix/self.Is)
            case _:
                raise ValueError("Unimplemented diode model requested")
    def g_model(self, vx, Tx=None, model="default"):
        self.__update_TSEP(Tx)
        if model == "default":
            model = self.par['Model_type']
        match model:
            case "pure_exp":
                return self.Is * np.exp(vx/self.par['Eta']/self.Vt)
            case "negative_saturation":
                return self.Is * np.exp(vx/self.par['Eta']/self.Vt - 1.0)
            case _:
                raise ValueError("Unimplemented diode model requested")

class NTC(Bipole):
    def __init__(self, R0=10e3, Beta=3780, T0=spc.convert_temperature(25, 'Celsius', 'Kelvin'), Rth= 100, Model_type='beta_value'):
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
        self.Tmin_deg = -50
        self.Tmax_deg = 150
        self.Tbase_deg = 25
        self.Nsim = 1001
        self.Vbias = 3.3
        self.Nj = 4
        self.diode0 = Diode(Is = 25e-9, Eta = 1.6)
        self.ntc0 = NTC()
        self.Teval = np.linspace(self.Tmin_deg, self.Tmax_deg, self.Nsim, endpoint=True)
        self.Teval = spc.convert_temperature(self.Teval, 'Celsius', 'Kelvin')
        self.Tbase = spc.convert_temperature(self.Tbase_deg, 'Celsius', 'Kelvin')
    def get_temperatures(self):
        return self.Teval
    def analysis_ideal_diode(self):
        vx = self.Nj * (self.diode0.r_model(self.Vbias/self.ntc0.par['R0'], self.Tbase) - 
                        self.diode0.par['Eta'] * self.diode0.Vt * np.log(self.ntc0.r_value(self.Teval)/self.ntc0.par['R0']))
        ix = self.diode0.g_model(vx/self.Nj, self.Tbase)
        return (vx, ix)
    def analysis_divider(self):
        vx = self.Vbias - self.Nj * self.diode0.par['Eta'] * self.diode0.Vt * sps.lambertw(
            self.ntc0.r_value(self.Teval)/(self.Nj * self.diode0.par['Eta'] * self.diode0.Vt) * 
            self.diode0.g_model(self.Vbias/self.Nj, self.Tbase)
        )
        ix = self.diode0.g_model(vx/self.Nj, self.Tbase)
        return (vx, ix)
    def sim_ideal_diode(self):
        # Ideal diode (diode voltage not affecting NTC voltage)
        ix = self.ntc0.g_model(self.Vbias, self.Teval)
        vx = self.Nj * self.diode0.r_model(ix, self.Tbase)
        return (vx, ix) # output voltage and diodes current
    def sim_divider(self, NTC_selfheat=False, Diode_selfheat=False):
        # Asymmetrical voltage divider
        # cir = lambda ix: Vbias - ntc0.r_model(ix, Tx) - Nj*diode0.r_model(ix) # KVL
        eq_circuit  = lambda vx, T_NTC, T_diode: self.ntc0.g_model(self.Vbias - vx, T_NTC) - self.diode0.g_model(vx/self.Nj, T_diode) # KCL
        eq_ntc_sh   = lambda vx, T_NTC, T_NTC0: self.ntc0.thermal_model(self.Vbias - vx, "voltage", T_NTC0, T_NTC)
        eq_diode_sh = lambda vx, T_diode, T_diode0: self.diode0.thermal_model(vx/self.Nj, "voltage", T_diode0, T_diode)
        eqs_system  = lambda vx, T_NTC, T_diode, T_NTC0, T_diode0: np.array([
            eq_circuit(vx, T_NTC, T_diode), 
            eq_ntc_sh(vx, T_NTC, T_NTC0), 
            eq_diode_sh(vx, T_diode, T_diode0)
            ])
        if (not NTC_selfheat) and (not Diode_selfheat):
            # Electric only (x: diode voltage)
            x = np.vectorize(lambda T_meas: spo.fsolve(lambda x: eqs_system(x[0], T_meas, self.Tbase, T_meas, self.Tbase)[0], 1.0))(self.Teval)
            vx = x
            T_NTC = self.Teval * np.ones_like(vx)
            T_diode = self.Tbase * np.ones_like(vx)
        elif NTC_selfheat and (not Diode_selfheat):
            # NTC self-heating only (x: [diode voltage, NTC temperature])
            x = np.vectorize(lambda T_meas: spo.fsolve(lambda x: eqs_system(x[0], x[1], self.Tbase, T_meas, self.Tbase)[[0, 1]], [1.0, T_meas]), signature='()->(2)')(self.Teval)
            vx = x[:,0]
            T_NTC = x[:,1]
            T_diode = self.Tbase * np.ones_like(vx)
        elif (not NTC_selfheat) and Diode_selfheat:
            # Diode self-heating only (x: [diode voltage, diode temperature])
            x = np.vectorize(lambda T_meas: spo.fsolve(lambda x: eqs_system(x[0], T_meas, x[1], T_meas, self.Tbase)[[0, 2]], [1.0, self.Tbase]), signature='()->(2)')(self.Teval)
            vx = x[:,0]
            T_NTC = self.Teval * np.ones_like(vx)
            T_diode = x[:,1]
        else:
            # Self-heating on both diode and NTC
            x = np.vectorize(lambda T_meas: spo.fsolve(lambda x: eqs_system(x[0], x[1], x[2], T_meas, self.Tbase), [1.0, T_meas, self.Tbase]), signature='()->(3)')(self.Teval)
            vx = x[:,0]
            T_NTC = x[:,1]
            T_diode = x[:,2]
        ix = self.diode0.g_model(vx/self.Nj, T_diode)
        return (vx, ix, T_NTC, T_diode)

# Simulation
mysim = NTC_simulation()
Tx = mysim.get_temperatures()
v_out = np.empty((mysim.Nsim, 4))
i_ntc = np.empty_like(v_out)
T_ntc = np.empty_like(v_out)
T_diode = np.empty_like(v_out)
v_out[:,0], i_ntc[:,0] = mysim.sim_ideal_diode()
v_out[:,1], i_ntc[:,1], T_ntc[:,1], T_diode[:,1] = mysim.sim_divider(NTC_selfheat=True, Diode_selfheat=True)
v_out[:,2], i_ntc[:,2] = mysim.analysis_ideal_diode()
v_out[:,3], i_ntc[:,3] = mysim.analysis_divider()

# Results
plt.figure(1)
plt.plot(v_out, 1/Tx)
plt.xlabel('Output voltage (V)')
plt.ylabel('Temperature reciprocal (1/K)')
plt.grid()
plt.show(block=False)

plt.figure(2)
plt.plot(v_out, Tx)
plt.xlabel('Output voltage (V)')
plt.ylabel('Temperature (K)')
plt.grid()
plt.show(block=False)

plt.figure(3)
plt.plot(Tx, T_ntc[:,1], label='NTC')
plt.plot(Tx, T_diode[:,1], label='diode')
plt.plot(Tx, Tx, label='reference')
plt.xlabel('Measured temperature (K)')
plt.ylabel('Component temperature (K)')
plt.grid()
plt.show(block=True)

print('Ciao')