# NTC overall simulation and study

import numpy as np
import numpy.polynomial.polynomial as npp
import pandas as pd
import scipy.constants as spc
import scipy.optimize as spo
import scipy.special as sps
import scipy.io as spio
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
    def thermal_model(self, v_or_i, input, Tstart, Tend, selfheating=True):
        if selfheating:
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

class Resistor(Bipole):
    def __init__(self, R0, T0=spc.convert_temperature(25, 'Celsius', 'Kelvin'), Rth=0, TCR=300e-6):
        super().__init__()
        self.par['R0'] = R0
        self.par['T0'] = T0
        self.par['Rth'] = Rth
        self.par['TCR'] = TCR
        # "model" parameter is completely neglected in this case
    def r_model(self, ix, Tx, model="default"):
        return self.r_value(Tx, model) * ix
    def g_model(self, vx, Tx, model="default"):
        return 1/self.r_value(Tx, model) * vx
    def r_value(self, Tx, model="default"):
        return self.par['R0'] * (1 + self.par['TCR'] * (Tx - self.par['T0']))
    
class Diode(Bipole):
    def __init__(self, Is=1e-15, Eta=1, Rth=100):
        super().__init__()
        self.par['Is'] = Is
        self.par['Eta'] = Eta
        self.par['Rth'] = Rth
        self.par['Eg'] = 1.12 * spc.eV # TODO: make it a parameter to account for non-silicon devices
        self.par['T0'] = 300 # TODO: make it a parameter
    def model_default(self, model):
        # Default model is negative saturation
        if model == "default":
            model = "negative_saturation"
        return model
    # Compute temperature sensitive parameters according to the specified temperature
    def Vt(self, Tx):
        return spc.Boltzmann * Tx/spc.e
    def Is(self, Tx):
        return self.par['Is'] * np.exp(-self.par['Eg']/spc.Boltzmann * (1/Tx - 1/self.par['T0'])) # TODO parameterize reference temperature for given saturation current
    def r_model(self, ix, Tx, model="default"):
        # Implement various models
        model = self.model_default(model)
        match model:
            case "pure_exp":
                return self.par['Eta'] * self.Vt(Tx) * np.log(ix/self.Is(Tx))
            case "negative_saturation":
                return self.par['Eta'] * self.Vt(Tx) * np.log(1.0 + ix/self.Is(Tx))
            case _:
                raise ValueError("Unimplemented diode model requested")
    def g_model(self, vx, Tx, model="default"):
        model = self.model_default(model)
        match model:
            case "pure_exp":
                return self.Is(Tx) * np.exp(vx/self.par['Eta']/self.Vt(Tx))
            case "negative_saturation":
                return self.Is(Tx) * np.exp(vx/self.par['Eta']/self.Vt(Tx) - 1.0)
            case _:
                raise ValueError("Unimplemented diode model requested")

class NTC(Resistor):
    def __init__(self, R0, T0, Rth, Beta=None, Poly=None, Table=None, Range=None):
        super().__init__(R0=R0, T0=T0, Rth=Rth, TCR=0)
        if Beta is not None:
            self.par['Beta'] = Beta
        elif Poly is not None:
            self.par['DCBA'] = Poly
        elif Table is not None:
            data = pd.read_csv(Table, delimiter=';', names=("T_deg", "r"))
            data['T'] = spc.convert_temperature(data['T_deg'], 'Celsius', 'Kelvin')
            data['recT'] = 1/data['T']
            data['g'] = np.log(data['r'])
            # TODO consider the range of the table on which to compute the coefficients
            # Determine Beta model coefficients
            par = npp.polyfit(data['recT'] - 1/self.par['T0'], data['g'], 1)
            self.par['Beta'] = par[1]
            asd = par[0]
            # TODO find a robust way to determine Steinhart-Hart coefficients

            # Determine the coefficients of the polynomial model
            self.par['DCBA'] = npp.polyfit(data['recT'], data['g'], 3)

            plt.figure()
            plt.plot(data['T_deg'], data['g'], label="Datasheet")
            plt.plot(data['T_deg'], np.log(self.r_value(data['T'], model="beta_value")/self.par['R0']), label="Beta model")
            plt.plot(data['T_deg'], np.log(self.r_value(data['T'], model="polynomial")/self.par['R0']), label="Polynomial model")
            plt.xlabel('Temperature ($^{\circ}C$)')
            plt.ylabel('Normalized log resistance (1)')
            plt.legend()
            plt.grid()
            plt.show()

            print(data)
        else:
            raise TypeError("At least one of Beta, Poly or Table must be specified")
    def model_default(self, model):
        # Default model is negative saturation
        if model == "default":
            model = "polynomial"
        return model
    def r_value(self, Tx, model="default"):
        model = self.model_default(model)
        match model:
            case "beta_value":
                return self.par['R0'] * np.exp(self.par['Beta'] * (1/Tx - 1/self.par['T0']))
            case "steinhart_hart":
                x = 1/self.par['C'] * (self.par['A'] - 1/Tx)
                y = np.sqrt((self.par['B']/(3*self.par['C']))**3 + x**2/4)
                return np.exp((y - x/2)**(1/3) - (y + x/2)**(1/3))
            case "polynomial":
                return self.par['R0'] * np.exp(npp.polyval(1/Tx, self.par['DCBA']))
            case _:
                raise ValueError("Unimplemented NTC model requested")

# Simulation parameters
class NTC_simulation:
    def __init__(self, V_bias, N_j, diode, ntc, R_cal=np.array([1e3, 10e3]), name="NTC_simulation"):
        self.T_min_deg = -50
        self.T_max_deg = 150
        self.T_base_deg = 25
        self.N_sim = 1001
        self.V_bias = V_bias
        self.N_j = N_j
        self.diode = diode
        self.ntc = ntc
        self.R_cal = R_cal
        self.g_cal = np.log(R_cal/self.ntc.par['R0'])
        self.v_cal = np.empty_like(self.R_cal)
        self.name = name
        self.T_eval = np.linspace(self.T_min_deg, self.T_max_deg, self.N_sim, endpoint=True)
        self.T_eval = spc.convert_temperature(self.T_eval, 'Celsius', 'Kelvin')
        self.T_base = spc.convert_temperature(self.T_base_deg, 'Celsius', 'Kelvin')
        self.calibrate()
    def calibrate(self):
        for ii_rcal in range(len(self.R_cal)):
            self.v_cal[ii_rcal], *_ = self.sim_divider(Resistor_selfheat=True, Diode_selfheat=True, res=Resistor(R0=self.R_cal[ii_rcal]), temp=self.T_base)
    def get_temperatures(self):
        return (self.T_eval, spc.convert_temperature(self.T_eval, 'Kelvin', 'Celsius'))
    def analysis_ideal_diode(self):
        vx = self.N_j * (self.diode.r_model(self.V_bias/self.ntc.par['R0'], self.T_base, model="pure_exp") - 
                        self.diode.par['Eta'] * self.diode.Vt(self.T_base) * np.log(self.ntc.r_value(self.T_eval, model="beta_value")/self.ntc.par['R0']))
        ix = self.diode.g_model(vx/self.N_j, self.T_base, model="pure_exp")
        Rx = self.ntc.r_value(self.T_eval, model="beta_value")
        g = np.log(Rx/self.ntc.par['R0'])
        return (vx, ix, Rx, g)
    def analysis_divider(self):
        vx = self.V_bias - self.N_j * self.diode.par['Eta'] * self.diode.Vt(self.T_base) * sps.lambertw(
            self.ntc.r_value(self.T_eval, model="beta_value")/(self.N_j * self.diode.par['Eta'] * self.diode.Vt(self.T_base)) * 
            self.diode.g_model(self.V_bias/self.N_j, self.T_base, model="pure_exp")
        )
        ix = self.diode.g_model(vx/self.N_j, self.T_base, model="pure_exp")
        Rx = self.ntc.r_value(self.T_eval, model="beta_value")
        g = np.log(Rx/self.ntc.par['R0'])
        return (vx, ix, Rx, g)
    def sim_ideal_diode(self):
        # Ideal diode (diode voltage not affecting NTC voltage)
        ix = self.ntc.g_model(self.V_bias, self.T_eval)
        vx = self.N_j * self.diode.r_model(ix, self.T_base) # diode at room temperature
        Rx = self.ntc.r_value(self.T_eval)
        g = np.log(Rx/self.ntc.par['R0'])
        return (vx, ix, Rx, g)
    def sim_divider(self, Resistor_selfheat=False, Diode_selfheat=False, res=None, temp=None):
        # Asymmetrical voltage divider
        # cir = lambda ix: Vbias - ntc0.r_model(ix, Tx) - Nj*diode0.r_model(ix) # KVL
        if res is None:
            res = self.ntc
        if temp is None:
            temp = self.T_eval
        eq_circuit  = lambda vx, T_res, T_diode: res.g_model(self.V_bias - vx, T_res) - self.diode.g_model(vx/self.N_j, T_diode) # KCL
        eq_res_sh   = lambda vx, T_res, T_res0: res.thermal_model(self.V_bias - vx, "voltage", T_res0, T_res, Resistor_selfheat)
        eq_diode_sh = lambda vx, T_diode, T_diode0: self.diode.thermal_model(vx/self.N_j, "voltage", T_diode0, T_diode, Diode_selfheat)
        eqs_system  = lambda vx, T_res, T_diode, T_res0, T_diode0: np.array([
            eq_circuit(vx, T_res, T_diode), 
            eq_res_sh(vx, T_res, T_res0), 
            eq_diode_sh(vx, T_diode, T_diode0)
            ])
        x = np.vectorize(
            lambda T_meas: spo.fsolve(
                lambda x: eqs_system(
                    x[0], x[1], x[2], T_meas, self.T_base), 
                [self.N_j, T_meas, self.T_base], 
                diag=[self.N_j, T_meas, self.T_base]), 
            signature='()->(3)')(temp)
        # Transform array of dimension 3 in bidimentional to avoid errors during calibration
        if len(x.shape) == 1:
            x = x[np.newaxis,:]
        vx = x[:,0]
        T_res = x[:,1]
        T_diode = x[:,2]
        ix = self.diode.g_model(vx/self.N_j, T_diode)
        Rx = res.r_value(T_res)
        g = np.log(Rx/res.par['R0'])
        return (vx, ix, T_res, T_diode, Rx, g)

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
diode_1N4148 = Diode(Is =  2.7e-9, Eta = 1.8, Rth = 350)
diode_1N4007 = Diode(Is = 500e-12, Eta = 1.5, Rth =  93)
diode_BC817  = Diode(Is =  20e-15, Eta = 1.0, Rth = 160)
# ntc_B57703M_10k = NTC(R0=10e3, T0=spc.convert_temperature(25, 'Celsius', 'Kelvin'), Rth=333, Beta=3988)
ntc_B57703M_10k = NTC(R0=10e3, T0=spc.convert_temperature(25, 'Celsius', 'Kelvin'), Rth=333, Table="./Dati_NTC/temperature_data.csv")
mysim = NTC_simulation(3.3, 4, diode_BC817, ntc_B57703M_10k, name="BC817")
mychar = NTC_characterization(ntc_B57703M_10k)
Tx, Tx_deg = mysim.get_temperatures()
v_out = np.empty((mysim.N_sim, 4))
i_ntc = np.empty_like(v_out)
T_ntc = np.empty_like(v_out)
T_diode = np.empty_like(v_out)
Rx = np.empty_like(v_out)
g = np.empty_like(v_out)
v_out[:,0], i_ntc[:,0], Rx[:,0], g[:,0] = mysim.sim_ideal_diode()
v_out[:,1], i_ntc[:,1], T_ntc[:,1], T_diode[:,1], Rx[:,1], g[:,1] = mysim.sim_divider(Resistor_selfheat=True, Diode_selfheat=True)
v_out[:,2], i_ntc[:,2], Rx[:,2], g[:,2] = mysim.analysis_ideal_diode()
v_out[:,3], i_ntc[:,3], Rx[:,3], g[:,3] = mysim.analysis_divider()
asd = mysim.g_cal[0] + (mysim.g_cal[1] - mysim.g_cal[0]) * (mysim.v_cal[0] - v_out[:,1])/(mysim.v_cal[0] - mysim.v_cal[1])
qwe = np.polyfit(asd, g[:,1], 3)
# v_n, i_n, R_n, T_n = mychar.simulate(400)
# v_m, i_m, R_m, I_m, V_m = mychar.analyze(400)
# v_a, i_a, R_a, T_a = mychar.attempt(400)

data = spio.loadmat('Dati_NTC/asdf.mat')
Ron_min = max(min(data['Ron'][:,0]), min(data['Ron'][:,3]))
Ron_max = min(max(data['Ron'][:,0]), max(data['Ron'][:,3]))
rons = np.linspace(Ron_min, Ron_max, 1001)
Ts_ref = np.interp(rons, data['Ron'][:,3], data['T'][:,3])
Ts_out = np.interp(rons, data['Ron'][:,0], data['T'][:,0])
T_rms_error = np.sqrt(np.mean((Ts_ref - Ts_out)**2))
print('RMS error: ', T_rms_error)


plt.figure(8)
plt.gcf().set_size_inches(4, 3)
plt.plot(data['T'][:,(0, 3)], 1e3*data['Ron'][:,(0, 3)], label=['proposed sensing', 'thermal chamber'])
plt.xlabel('Junction temperature ($^{\circ}C$)')
plt.ylabel('On-state resistance ($m\Omega$)')
plt.legend()
plt.grid()
plt.savefig('ron.pdf', bbox_inches='tight')
plt.show(block=False)

plt.figure(9)
plt.gcf().set_size_inches(4, 3)
plt.plot(Ts_ref, Ts_out, '-', label='experimental data')
plt.plot(Ts_ref, Ts_ref, '--', label='ideal reference')
plt.xlabel('Ambient temperature ($^{\circ}C$)')
plt.ylabel('Sensed temperature ($^{\circ}C$)')
plt.legend()
plt.grid()
plt.savefig('temp_comparison.pdf', bbox_inches='tight')
plt.show(block=False)

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
plt.gcf().set_size_inches(4, 3)
plt.plot(Tx_deg, 1e3*(T_ntc[:,1] - Tx), label='NTC')
plt.plot(Tx_deg, 1e3*(T_diode[:,1] - mysim.T_base), label='BJT')
plt.xlabel('Measured temperature (°C)')
plt.ylabel('Component overtemperature (mK)')
plt.legend()
plt.grid()
plt.savefig('overtemp_N6.pdf', bbox_inches='tight')
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
plt.plot(v_out, g, label=['ideal diode (sim)', 'divider (sim)', 'ideal diode (model)', 'divider (model)'])
plt.plot(v_out[:,1], asd, label='simplified')
plt.xlabel('Output voltage (V)')
plt.ylabel('Real normalized resistance g (1)')
plt.legend()
plt.grid()
plt.show(block=False)

plt.figure(6)
plt.plot(asd, g[:,1], asd, np.polyval(qwe, asd))
plt.xlabel('Estimated normalized resistance g (1)')
plt.ylabel('Real normalized resistance g (1)')
plt.legend()
plt.grid()
plt.show(block=False)

plt.figure(7)
plt.gcf().set_size_inches(4, 3)
plt.plot(v_out[:,1], g[:,1], label='simulation')
plt.plot(v_out[:,1], asd, label='simplified')
plt.xlabel('Output voltage $v_x$ (V)')
plt.ylabel('Normalized log resistance $g$ (1)')
plt.legend()
plt.grid()
plt.savefig('characteristics_N1.pdf', bbox_inches='tight')
plt.show(block=True)

'''
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
'''

print('Ciao')
print(qwe)
