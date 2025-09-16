# NTC overall simulation and study

from components import Diode, NTC, ADC
from ntccond import NTC_conditioning
from utils import curvature, curvature2, fake_curvature, multi_curvature

import numpy as np
import pandas as pd
import scipy.constants as spc
import matplotlib.pyplot as plt
import os

# Simulation parameters
Tm_min_deg = -50
Tm_max_deg = 150
T_base_deg = 25
N_pts = 101
diode_1N4148 = Diode(Is =  2.7e-9, Eta = 1.8, Rth = 350)
diode_1N4007 = Diode(Is = 500e-12, Eta = 1.5, Rth =  93)
diode_BC817  = Diode(Is =  20e-15, Eta = 1.0, Rth = 160)
ntc_B57703M_10k = NTC(R0=10e3, T0=spc.convert_temperature(25, 'Celsius', 'Kelvin'), Rth=333, Table="../data/temperature_data.csv")
adc_12bit = ADC(bits=12, Vref=3.3, enob=9)
Out_folder = "./results/"

asd = ADC(bits=12, Vref=3.3, enob=9)
vx = 1.65*np.ones(1001)
vout = asd.allpass(vx)
plt.figure(1)
plt.hist(vout - vx, bins=50)
plt.xlabel('Error (V)')
plt.ylabel('Counts (1)')
plt.grid()
plt.show(block=False)

# Simulation configuration
meta = pd.DataFrame(
    columns=["name", "sim_type", "ntc", "diode", "adc", "Tm_min", "Tm_max", "Tamb_diode", "V_bias", "N_diodes", "ntc_selfheat", "diode_selfheat", "T_calib", "do_compensate", "T_comp", "T_comp_calib", "do_optimize", "do_noise_analysis", "noise_runs", "N_pts"],
    data = [
        [                       "Diode divider, no self-heating, 1 diode",     "diode_divider_sim", ntc_B57703M_10k, diode_BC817, adc_12bit, Tm_min_deg, Tm_max_deg, T_base_deg, 3.3, 1, False, False, (15, 140), False, 35, (-20, 30, 80, 130), False, False, 0, N_pts],
        [                          "Diode divider, self-heating, 1 diode",     "diode_divider_sim", ntc_B57703M_10k, diode_BC817, adc_12bit, Tm_min_deg, Tm_max_deg, T_base_deg, 3.3, 1,  True,  True, (15, 140), False, 35, (-20, 30, 80, 130), False, False, 0, N_pts],
        [                      "Diode divider, no self-heating, 4 diodes",     "diode_divider_sim", ntc_B57703M_10k, diode_BC817, adc_12bit, Tm_min_deg, Tm_max_deg, T_base_deg, 3.3, 4, False, False, (15, 140), False, 35, (-20, 30, 80, 130), False, False, 0, N_pts],
        [                         "Diode divider, self-heating, 4 diodes",     "diode_divider_sim", ntc_B57703M_10k, diode_BC817, adc_12bit, Tm_min_deg, Tm_max_deg, T_base_deg, 3.3, 4,  True,  True, (15, 140), False, 35, (-20, 30, 80, 130), False, False, 0, N_pts],
        [                      "Diode divider, no self-heating, 6 diodes",     "diode_divider_sim", ntc_B57703M_10k, diode_BC817, adc_12bit, Tm_min_deg, Tm_max_deg, T_base_deg, 3.3, 6, False, False, (15, 140), False, 35, (-20, 30, 80, 130), False, False, 0, N_pts],
        [                         "Diode divider, self-heating, 6 diodes",     "diode_divider_sim", ntc_B57703M_10k, diode_BC817, adc_12bit, Tm_min_deg, Tm_max_deg, T_base_deg, 3.3, 6,  True,  True, (15, 140), False, 35, (-20, 30, 80, 130), False, False, 0, N_pts],
        [             "Diode divider, self-heating, 4 diodes, Td = -40°C",     "diode_divider_sim", ntc_B57703M_10k, diode_BC817, adc_12bit, Tm_min_deg, Tm_max_deg,        -40, 3.3, 4,  True,  True, (15, 140), False, 35, (-20, 30, 80, 130), False, False, 0, N_pts],
        [              "Diode divider, self-heating, 4 diodes, Td = 25°C",     "diode_divider_sim", ntc_B57703M_10k, diode_BC817, adc_12bit, Tm_min_deg, Tm_max_deg,         25, 3.3, 4,  True,  True, (15, 140), False, 35, (-20, 30, 80, 130), False, False, 0, N_pts],
        [             "Diode divider, self-heating, 4 diodes, Td = 105°C",     "diode_divider_sim", ntc_B57703M_10k, diode_BC817, adc_12bit, Tm_min_deg, Tm_max_deg,        105, 3.3, 4,  True,  True, (15, 140), False, 35, (-20, 30, 80, 130), False, False, 0, N_pts],
        [                            "Resistive divider, no self-heating", "resistive_divider_sim", ntc_B57703M_10k, diode_BC817, adc_12bit, Tm_min_deg, Tm_max_deg, T_base_deg, 3.3, 6,  True,  True, (15, 140), False, 35, (-20, 30, 80, 130), False, False, 0, N_pts],
        [ "Diode divider, self-heating, 4 diodes, Td = 25°C, compensated",     "diode_divider_sim", ntc_B57703M_10k, diode_BC817, adc_12bit, Tm_min_deg, Tm_max_deg,         25, 3.3, 4,  True,  True, (15, 140),  True, 25, (-20, 30, 80, 130), False, False, 0, N_pts],
        [   "Diode divider, self-heating, 4 diodes, Td = 25°C, optimized",     "diode_divider_sim", ntc_B57703M_10k, diode_BC817, adc_12bit, Tm_min_deg, Tm_max_deg,         25, 3.3, 4,  True,  True, (15, 140), False, 25, (-20, 30, 80, 130),  True, False, 0, N_pts]
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
plt.plot(res[4]['Tm_deg'], res[4]['Vx'], label=meta['name'][4])
plt.plot(res[5]['Tm_deg'], res[5]['Vx'], label=meta['name'][5])
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
plt.plot(res[4]['g_est'], res[4]['g'], label=meta['name'][4])
plt.plot(res[5]['g_est'], res[5]['g'], label=meta['name'][5])
plt.xlabel('Estimated log resistance (1)')
plt.ylabel('Real log resistance (1)')
plt.legend()
plt.grid()
plt.show(block=False)

plt.figure(4)
plt.plot(res[6]['g_est'], res[6]['g'], label=meta['name'][6])
plt.plot(res[7]['g_est'], res[7]['g'], label=meta['name'][7])
plt.plot(res[8]['g_est'], res[8]['g'], label=meta['name'][8])
plt.xlabel('Estimated log resistance (1)')
plt.ylabel('Real log resistance (1)')
plt.legend()
plt.grid()
plt.show(block=False)

plt.figure(5)
plt.plot(res[1]['Tm_deg'], res[1]['Vx'], label=meta['name'][1])
plt.plot(res[3]['Tm_deg'], res[3]['Vx'], label=meta['name'][3])
plt.plot(res[5]['Tm_deg'], res[5]['Vx'], label=meta['name'][5])
plt.plot(res[9]['Tm_deg'], res[9]['Vx'], label=meta['name'][9])
plt.xlabel('Measured temperature (°C)')
plt.ylabel('Output voltage (V)')
plt.legend()
plt.grid()
plt.show(block=False)

plt.figure(6)
plt.gcf().set_size_inches(4, 3)
plt.plot(res[1]['Tm_deg'], res[1]['T_ntc'] - res[1]['Tm'], label="N = 1")
plt.plot(res[3]['Tm_deg'], res[3]['T_ntc'] - res[3]['Tm'], label="N = 4")
plt.plot(res[5]['Tm_deg'], res[5]['T_ntc'] - res[5]['Tm'], label="N = 6")
plt.xlabel('Measured temperature (°C)')
plt.ylabel('NTC overtemperature (K)')
plt.legend()
plt.grid()
plt.savefig(os.path.join(Out_folder, 'NTC_overtemperature.pdf'), bbox_inches='tight')
plt.show(block=False)

plt.figure(7)
plt.gcf().set_size_inches(4, 3)
plt.plot(res[1]['Tm_deg'], res[1]['T_diode'] - res[1]['T_diode0'], label="N = 1")
plt.plot(res[3]['Tm_deg'], res[3]['T_diode'] - res[3]['T_diode0'], label="N = 4")
plt.plot(res[5]['Tm_deg'], res[5]['T_diode'] - res[5]['T_diode0'], label="N = 6")
plt.xlabel('Measured temperature (°C)')
plt.ylabel('BJT overtemperature (K)')
plt.legend()
plt.grid()
plt.savefig(os.path.join(Out_folder, 'BJT_overtemperature.pdf'), bbox_inches='tight')
plt.show(block=False)

plt.figure(8)
plt.gcf().set_size_inches(4, 3)
plt.plot(res[1]['Vx'], res[1]['g'], label="N = 1")
plt.plot(res[3]['Vx'], res[3]['g'], label="N = 4")
plt.plot(res[5]['Vx'], res[5]['g'], label="N = 6")
plt.xlabel('Output voltage (V)')
plt.ylabel('Normalized log resistance g (1)')
plt.legend()
plt.grid()
plt.savefig(os.path.join(Out_folder, 'linearity.pdf'), bbox_inches='tight')
plt.show(block=False)

plt.figure(9)
plt.gcf().set_size_inches(4, 3)
plt.plot(res[6]['Tm_deg'], res[6]['Vx'], label="T_d = -40 °C")
plt.plot(res[7]['Tm_deg'], res[7]['Vx'], label="T_d = 25 °C")
plt.plot(res[8]['Tm_deg'], res[8]['Vx'], label="T_d = 105 °C")
plt.xlabel('Measured temperature (°C)')
plt.ylabel('Output voltage (V)')
plt.legend()
plt.grid()
plt.savefig(os.path.join(Out_folder, 'diode_temp_automotive.pdf'), bbox_inches='tight')
plt.show(block=False)

plt.figure(10)
plt.gcf().set_size_inches(4, 3)
plt.plot(res[6]['Tm_deg'], res[6]['g_est'], label="T_d = -40 °C")
plt.plot(res[7]['Tm_deg'], res[7]['g_est'], label="T_d = 25 °C")
plt.plot(res[8]['Tm_deg'], res[8]['g_est'], label="T_d = 105 °C")
plt.xlabel('Measured temperature (°C)')
plt.ylabel('Estimated log resistance (1)')
plt.legend()
plt.grid()
plt.savefig(os.path.join(Out_folder, 'diode_temp_automotive_calib.pdf'), bbox_inches='tight')
plt.show(block=False)

plt.figure(11)
plt.gcf().set_size_inches(4, 3)
plt.plot(res[6]['Tm_deg'], res[6]['Tm_est_deg'] - res[6]['Tm_deg'], label="T_d = -40 °C")
plt.plot(res[7]['Tm_deg'], res[7]['Tm_est_deg'] - res[7]['Tm_deg'], label="T_d = 25 °C")
plt.plot(res[8]['Tm_deg'], res[8]['Tm_est_deg'] - res[8]['Tm_deg'], label="T_d = 105 °C")
plt.xlabel('Measured temperature (°C)')
plt.ylabel('Conditioning error (°C)')
plt.legend()
plt.grid()
plt.savefig(os.path.join(Out_folder, 'error_automotive_range.pdf'), bbox_inches='tight')
plt.savefig(os.path.join(Out_folder, 'error_automotive_range.png'), bbox_inches='tight', dpi=600)
plt.show(block=False)

plt.figure(12)
plt.plot(res[7]['Tm_deg'], res[7]['Tm_est_deg'] - res[7]['Tm_deg'], label="Uncompensated")
plt.plot(res[10]['Tm_deg'], res[10]['Tm_est_deg'] - res[10]['Tm_deg'], label="Compensated")
plt.plot(res[11]['Tm_deg'], res[11]['Tm_est_deg'] - res[11]['Tm_deg'], label="Optimized")
plt.xlabel('Measured temperature (°C)')
plt.ylabel('Conditioning error (°C)')
plt.legend()
plt.grid()
plt.show(block=True)

print('Ciao')
