# NTC conditioning circuits simulation

from components import Diode, NTC, Resistor

import numpy as np
import scipy.constants as spc
import scipy.optimize as spo
import scipy.special as sps

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
