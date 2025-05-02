clear
clc
close all
format shorteng

K0 = 273.15;
data = readmatrix('temperature_data.csv');
ntc.T_deg = data(:,1);
ntc.r = data(:,2); % [1] normalized R, r = R/R0
ntc.g = log(ntc.r);

p_g_to_Tdeg = polyfit(ntc.g, ntc.T_deg, 3);
p_g_to_recT = polyfit(ntc.g, 1./(ntc.T_deg + K0), 3);

ntc.T_deg_polyTdeg = polyval(p_g_to_Tdeg, ntc.g);
ntc.T_deg_polyrecT = 1./polyval(p_g_to_recT, ntc.g) - K0;

err_g_to_Tdeg = rms(ntc.T_deg - ntc.T_deg_polyTdeg);
err_g_to_recT = rms(ntc.T_deg - ntc.T_deg_polyrecT);

p_g_to_Tdeg
err_g_to_Tdeg
err_g_to_recT

figure
hold on
plot(ntc.T_deg, ntc.g)
plot(ntc.T_deg_polyTdeg, ntc.g)
xlabel('Temperature (°C)')
ylabel('g (1)')
box on
grid on

figure
hold on
plot(1./(ntc.T_deg + K0), ntc.g)
plot(1./(ntc.T_deg_polyrecT + K0), ntc.g)
hold on
xlabel('Temperature reciprocal (1/K)')
ylabel('g (1)')
box on
grid on
