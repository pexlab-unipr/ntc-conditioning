# Classes to simulate circuit components in Python

import numpy as np
import numpy.polynomial.polynomial as npp
import pandas as pd
import scipy.constants as spc
import matplotlib.pyplot as plt

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
            # For now, just save the temperature limits, for later inversion
            self.par['Tmin'] = np.min(data['T'])
            self.par['Tmax'] = np.max(data['T'])
            # Determine Beta model coefficients
            par = npp.polyfit(data['recT'] - 1/self.par['T0'], data['g'], 1)
            self.par['Beta'] = par[1]
            asd = par[0]
            # TODO find a robust way to determine Steinhart-Hart coefficients

            # Determine the coefficients of the polynomial model
            self.par['DCBA'] = npp.polyfit(data['recT'], data['g'], 3)

            # plt.figure()
            # plt.plot(data['T_deg'], data['g'], label="Datasheet")
            # plt.plot(data['T_deg'], np.log(self.r_value(data['T'], model="beta_value")/self.par['R0']), label="Beta model")
            # plt.plot(data['T_deg'], np.log(self.r_value(data['T'], model="polynomial")/self.par['R0']), label="Polynomial model")
            # plt.xlabel('Temperature (°C)')
            # plt.ylabel('Normalized log resistance (1)')
            # plt.legend()
            # plt.grid()
            # plt.show(block=False)
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
    def T_from_logR(self, logR, model="default"):
        # Compute temperature from g = logR = log(R/R0)
        model = self.model_default(model)
        match model:
            case "beta_value":
                return 1/(1/self.par['T0'] + logR/self.par['Beta'])
            case "steinhart_hart":
                raise NotImplementedError("Inverse Steinhart-Hart not implemented")
            case "polynomial":
                T_est = np.empty_like(logR)
                for ii in range(logR.size):
                    # Solve 1/T from polynomial equation
                    # logR = D + C*(1/T) + B*(1/T)^2 + A*(1/T)^3
                    # A*(1/T)^3 + B*(1/T)^2 + C*(1/T) + (D - logR) = 0
                    # Coefficients of the polynomial in 1/T
                    p = self.par['DCBA'].copy()
                    jj = ii if logR.ndim == 1 else np.unravel_index(ii, logR.shape, order='C')
                    p[0] -= logR[jj]
                    roots = npp.Polynomial(p).roots()
                    # Find the only suitable solution
                    Ts = 1/roots # solving for 1/T
                    Ts = Ts[np.isreal(Ts)].real # only real temperatures
                    Ts = Ts[(Ts > self.par['Tmin']-10) & (Ts < self.par['Tmax']+10)] # only temperature in model range
                    if len(Ts) != 1:
                        T_est[jj] = np.nan
                        #raise ValueError("Multiple or no real roots found for polynomial inversion")
                    else:
                        T_est[jj] = Ts
                return T_est
            case _:
                raise ValueError("Unimplemented NTC model requested")

class ADC:
    def __init__(self, bits=12, Vref=3.3, enob=12):
        self.par = {}
        self.par['bits'] = bits
        self.par['Vref'] = Vref
        self.par['LSB'] = Vref/(2**bits - 1)
        self.par['enob'] = enob
        if enob > bits:
            raise ValueError("ENOB cannot be greater than ADC resolution")
        elif enob < bits:
            self.par['noise_rms'] = Vref/np.sqrt(10**((enob*6.02 + 1.76)/10) - 1) # assuming Gaussian distribution of noise
        else:
            self.par['noise_rms'] = 0.0
        # TODO: why not use Vref/2**enob ?
        # Initialize random number generator for noise
        self.rng = np.random.default_rng(210789) # fixed seed for reproducibility
    def adc_model(self, vx):
        # Ideal ADC model: quantization only
        vx = vx + self.rng.normal(0, self.par['noise_rms'], size=vx.shape) # add noise
        adc_code = np.round(vx/self.par['LSB'])
        adc_code = np.clip(adc_code, 0, 2**self.par['bits'] - 1)
        return adc_code
    def dac_model(self, code):
        # Ideal DAC model: quantization only
        code = np.clip(code, 0, 2**self.par['bits'] - 1)
        vx = code * self.par['LSB']
        return vx
    def allpass(self, vx):
        vout = self.dac_model(self.adc_model(vx))
        return vout