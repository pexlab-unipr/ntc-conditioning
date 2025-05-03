# NTC overall simulation and study

from components import Diode, NTC
from ntccond import NTC_conditioning
from utils import curvature, curvature2, fake_curvature, multi_curvature

import numpy as np
import pandas as pd
import scipy.constants as spc
import matplotlib.pyplot as plt

# Simulation parameters
Tm_min_deg = -50
Tm_max_deg = 150
T_base_deg = 25
N_pts = 101
diode_1N4148 = Diode(Is =  2.7e-9, Eta = 1.8, Rth = 350)
diode_1N4007 = Diode(Is = 500e-12, Eta = 1.5, Rth =  93)
diode_BC817  = Diode(Is =  20e-15, Eta = 1.0, Rth = 160)
ntc_B57703M_10k = NTC(R0=10e3, T0=spc.convert_temperature(25, 'Celsius', 'Kelvin'), Rth=333, Table="../data/temperature_data.csv")

meta = pd.DataFrame(
    columns=["name", "sim_type", "ntc", "diode", "Tm_min", "Tm_max", "Tamb_diode", "V_bias", "N_diodes", "ntc_selfheat", "diode_selfheat", "T_A", "T_B", "N_pts"],
    data = [
        ["Diode divider, no self-heating, 4 diodes", "diode_divider_sim", ntc_B57703M_10k, diode_BC817, Tm_min_deg, Tm_max_deg, T_base_deg, 3.3, 4, False, False, 15, 140, N_pts],
        ["Diode divider, self-heating, 4 diodes", "diode_divider_sim", ntc_B57703M_10k, diode_BC817, Tm_min_deg, Tm_max_deg, T_base_deg, 3.3, 4, True, True, 15, 140, N_pts],
        ["Diode divider, no self-heating, 1 diode", "diode_divider_sim", ntc_B57703M_10k, diode_BC817, Tm_min_deg, Tm_max_deg, T_base_deg, 3.3, 1, False, False, 15, 140, N_pts],
        ["Diode divider, self-heating, 1 diode", "diode_divider_sim", ntc_B57703M_10k, diode_BC817, Tm_min_deg, Tm_max_deg, T_base_deg, 3.3, 1, True, True, 15, 140, N_pts],
        ["Diode divider, self-heating, 4 diodes, Td = 20°C", "diode_divider_sim", ntc_B57703M_10k, diode_BC817, Tm_min_deg, Tm_max_deg, 20, 3.3, 4, True, True, 15, 140, N_pts],
        ["Diode divider, self-heating, 4 diodes, Td = 60°C", "diode_divider_sim", ntc_B57703M_10k, diode_BC817, Tm_min_deg, Tm_max_deg, 60, 3.3, 4, True, True, 15, 140, N_pts],
        ["Diode divider, self-heating, 4 diodes, Td = 100°C", "diode_divider_sim", ntc_B57703M_10k, diode_BC817, Tm_min_deg, Tm_max_deg, 100, 3.3, 4, True, True, 15, 140, N_pts]
        ])

N_sim = len(meta)

# Batch simulation
res = []
for index, row in meta.iterrows():
    print(f"Simulation {index+1:2d}/{N_sim:2d}: {row['name']}")
    mysim = NTC_conditioning(row)
    data = mysim.simulate()
    res.append(data)

# Results
plt.figure(2)
plt.plot(res[0]['Tm_deg'], res[0]['Vx'], label=meta['name'][0])
plt.plot(res[1]['Tm_deg'], res[1]['Vx'], label=meta['name'][1])
plt.plot(res[2]['Tm_deg'], res[2]['Vx'], label=meta['name'][2])
plt.plot(res[3]['Tm_deg'], res[3]['Vx'], label=meta['name'][3])
plt.xlabel('Measured temperature (°C)')
plt.ylabel('Output voltage (V)')
plt.legend()
plt.grid()
plt.show(block=False)

plt.figure(3)
plt.plot(res[0]['g_est'], res[0]['g'], label=meta['name'][0])
plt.plot(res[1]['g_est'], res[1]['g'], label=meta['name'][1])
plt.plot(res[2]['g_est'], res[2]['g'], label=meta['name'][2])
plt.plot(res[3]['g_est'], res[3]['g'], label=meta['name'][3])
plt.xlabel('Estimated log resistance (1)')
plt.ylabel('Real log resistance (1)')
plt.legend()
plt.grid()
plt.show(block=False)

plt.figure(4)
plt.plot(res[4]['g_est'], res[4]['g'], label=meta['name'][4])
plt.plot(res[5]['g_est'], res[5]['g'], label=meta['name'][5])
plt.plot(res[6]['g_est'], res[6]['g'], label=meta['name'][6])
plt.xlabel('Estimated log resistance (1)')
plt.ylabel('Real log resistance (1)')
plt.legend()
plt.grid()
plt.show(block=False)

Tx, Tx_deg = mysim.get_temperatures()
v_out = np.empty((mysim.N_pts, 4))
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
plt.plot(Tx_deg, 1e3*(T_ntc[:,1] - Tx), label='NTC')
plt.plot(Tx_deg, 1e3*(T_diode[:,1] - mysim.T_base), label='BJT')
plt.xlabel('Measured temperature (°C)')
plt.ylabel('Component overtemperature (mK)')
plt.legend()
plt.grid()
plt.show(block=False)

plt.figure(4)
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
plt.plot(v_out[:,1], g[:,1], label='simulation')
plt.plot(v_out[:,1], asd, label='simplified')
plt.xlabel('Output voltage $v_x$ (V)')
plt.ylabel('Normalized log resistance $g$ (1)')
plt.legend()
plt.grid()
plt.show(block=True)

print('Ciao')
