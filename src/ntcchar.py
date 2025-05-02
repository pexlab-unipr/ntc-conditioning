# Class for thermal characterization of NTC thermistors
import numpy as np
import scipy.optimize as spo

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
