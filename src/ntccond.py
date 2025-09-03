# NTC conditioning circuits simulation

from components import Diode, NTC, Resistor

import numpy as np
import numpy.polynomial.polynomial as npp
import pandas as pd
import scipy.constants as spc
import scipy.optimize as spo
import scipy.special as sps

import matplotlib.pyplot as plt

class NTC_conditioning:
    def __init__(self, conf):
        self.name = conf['name']
        self.sim_type = conf['sim_type']
        self.T_min_deg = conf['Tm_min']
        self.T_max_deg = conf['Tm_max']
        self.N_pts = conf['N_pts']
        self.V_bias = conf['V_bias']
        self.N_j = conf['N_diodes']
        self.diode = conf['diode']
        self.ntc = conf['ntc']
        self.ntc_selfheat = conf['ntc_selfheat']
        self.diode_selfheat = conf['diode_selfheat']
        self.do_compensate = conf['do_compensate']
        self.T_cal_A = spc.convert_temperature(conf['T_A'], 'Celsius', 'Kelvin')
        self.T_cal_B = spc.convert_temperature(conf['T_B'], 'Celsius', 'Kelvin')
        self.T_base = spc.convert_temperature(conf['Tamb_diode'], 'Celsius', 'Kelvin')
        self.T_comp = spc.convert_temperature(conf['T_comp'], 'Celsius', 'Kelvin')
    def simulate(self):
        data = pd.DataFrame()
        Tm = np.linspace(self.T_min_deg, self.T_max_deg, self.N_pts, endpoint=True)
        Tm = spc.convert_temperature(Tm, 'Celsius', 'Kelvin')
        data['Tm'] = Tm
        data['Tm_deg'] = spc.convert_temperature(Tm, 'Kelvin', 'Celsius')
        match self.sim_type:
            case "diode_divider_sim":
                self.calibrate()
                if self.do_compensate:
                    self.update_compensation(Tm)
                data['Vx'], data['Ix'], data['T_ntc'], data['T_diode'], data['Rx'], data['g'], data['T_diode0'] = self.sim_diode_divider(Tm)
                data['g_est'] = self.approx_diode_divider(data['Vx'])
                data['g_est'] = self.compensate_logr(data['g_est'])
                data['Tm_est_deg'] = spc.convert_temperature(self.ntc.T_from_logR(data['g_est'], model="polynomial"), 'Kelvin', 'Celsius')
            case "resistive_divider_sim":
                data['Vx'], data['Ix'], data['T_ntc'], data['Rx'], data['g'] = self.sim_resistive_divider(Tm)
            case _:
                raise ValueError("Unimplemented NTC conditioning simulation type requested")
        return data
    # Calibrate diode parameters at two temperature points
    def calibrate(self):
        T_cal = np.array([self.T_cal_A, self.T_cal_B])
        v_cal, _, _, _, R_cal, g_cal, _ = self.sim_diode_divider(T_cal)
        self.v_A = v_cal[0]
        self.v_B = v_cal[1]
        self.g_A = g_cal[0]
        self.g_B = g_cal[1]
        self.R_A = R_cal[0]
        self.R_B = R_cal[1]
    # Compensate log resistance estimation for self-heating and conditioning non-idealities
    def update_compensation(self, Tm):
        # Temporarily change diode temperature for compensation of the log resistance
        T_base = self.T_base
        self.T_base = self.T_comp
        Vx, _, _, _, _, g, _ = self.sim_diode_divider(Tm)
        g_est = self.approx_diode_divider(Vx)
        self.T_base = T_base
        #
        plt.figure(15)
        plt.plot(Tm - 273.15, Vx)
        plt.grid()
        plt.show(block=True)
        # Fit a polynomial to the compensation error
        par = npp.polyfit(g_est, g, 3)
        self.comp_par = par
    def analysis_ideal_diode(self):
        vx = self.N_j * (self.diode.r_model(self.V_bias/self.ntc.par['R0'], self.T_base, model="pure_exp") - 
                        self.diode.par['Eta'] * self.diode.Vt(self.T_base) * np.log(self.ntc.r_value(self.Tm, model="beta_value")/self.ntc.par['R0']))
        ix = self.diode.g_model(vx/self.N_j, self.T_base, model="pure_exp")
        Rx = self.ntc.r_value(self.Tm, model="beta_value")
        g = np.log(Rx/self.ntc.par['R0'])
        return (vx, ix, Rx, g)
    def analysis_divider(self):
        vx = self.V_bias - self.N_j * self.diode.par['Eta'] * self.diode.Vt(self.T_base) * sps.lambertw(
            self.ntc.r_value(self.Tm, model="beta_value")/(self.N_j * self.diode.par['Eta'] * self.diode.Vt(self.T_base)) * 
            self.diode.g_model(self.V_bias/self.N_j, self.T_base, model="pure_exp")
        )
        ix = self.diode.g_model(vx/self.N_j, self.T_base, model="pure_exp")
        Rx = self.ntc.r_value(self.Tm, model="beta_value")
        g = np.log(Rx/self.ntc.par['R0'])
        return (vx, ix, Rx, g)
    def sim_ideal_diode(self):
        # Ideal diode (diode voltage not affecting NTC voltage)
        ix = self.ntc.g_model(self.V_bias, self.Tm)
        vx = self.N_j * self.diode.r_model(ix, self.T_base) # diode at room temperature
        Rx = self.ntc.r_value(self.Tm)
        g = np.log(Rx/self.ntc.par['R0'])
        return (vx, ix, Rx, g)
    def sim_diode_divider(self, Tm):
        # Diode voltage divider
        # eq_circuit = lambda ix: Vbias - ntc0.r_model(ix, Tx) - Nj*diode0.r_model(ix) # KVL
        eq_circuit  = lambda vx, T_res, T_diode: self.ntc.g_model(self.V_bias - vx, T_res) - self.diode.g_model(vx/self.N_j, T_diode) # KCL
        eq_res_sh   = lambda vx, T_res, T_res0: self.ntc.thermal_model(self.V_bias - vx, "voltage", T_res0, T_res, self.ntc_selfheat)
        eq_diode_sh = lambda vx, T_diode, T_diode0: self.diode.thermal_model(vx/self.N_j, "voltage", T_diode0, T_diode, self.diode_selfheat)
        eqs_system  = lambda vx, T_res, T_diode, T_res0, T_diode0: np.array([
            eq_circuit(vx, T_res, T_diode), 
            eq_res_sh(vx, T_res, T_res0), 
            eq_diode_sh(vx, T_diode, T_diode0)])
        y = np.empty((len(Tm), 3))
        for ii in range(len(Tm)):
            # y[ii,:] = spo.fsolve(
            #     lambda x: eqs_system(
            #         x[0], x[1], x[2], Tm[ii], self.T_base), 
            #     [self.N_j, Tm[ii], self.T_base], 
            #     diag=[self.N_j, Tm[ii], self.T_base])
            res = spo.root(
                lambda x: eqs_system(
                    x[0], x[1], x[2], Tm[ii], self.T_base), 
                [self.N_j, Tm[ii], self.T_base], 
                method='lm')
            y[ii,:] = res.x
        vx = y[:,0]
        T_res = y[:,1]
        T_diode = y[:,2]
        ix = self.diode.g_model(vx/self.N_j, T_diode)
        Rx = self.ntc.r_value(T_res)
        g = np.log(Rx/self.ntc.par['R0'])
        T_diode0 = self.T_base * np.ones_like(T_res)
        return (vx, ix, T_res, T_diode, Rx, g, T_diode0)
    def sim_resistive_divider(self, Tm):
        # Resistive voltage divider
        vx = self.V_bias * self.ntc.par['R0']/(self.ntc.r_value(Tm) + self.ntc.par['R0'])
        ix = self.ntc.g_model(vx, Tm)
        T_res = Tm
        Rx = self.ntc.r_value(Tm)
        g = np.log(Rx/self.ntc.par['R0'])
        return (vx, ix, T_res, Rx, g)
    def approx_diode_divider(self, vx):
        g_est = self.g_A + (self.g_B - self.g_A) * (self.v_A - vx)/(self.v_A - self.v_B)
        return g_est
    def compensate_logr(self, g_est):
        if self.do_compensate:
            g_est = npp.polyval(g_est, self.comp_par)
        return g_est
