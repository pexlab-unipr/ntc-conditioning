function adc_dac_demo()
    % Parametri ADC
    bits = 12;
    Vref = 3.3;
    enob = 9;
    
    % Inizializzazione parametri
    LSB = Vref/(2^bits - 1);
    if enob > bits
        error('ENOB cannot be greater than ADC resolution');
    elseif enob < bits
        noise_rms = Vref / sqrt(10^((enob*6.02 + 1.76)/10) - 1);
    else
        noise_rms = 0.0;
    end
    
    % Generatore randomico a seme fisso
    rng(210789,'twister');
    
    % Segnale di ingresso
    vx = 1.65 * ones(1,1001);
    
    % Passaggio ADC -> DAC
    adc_code = adc_model(vx, bits, LSB, noise_rms);
    vout = dac_model(adc_code, bits, LSB);
    
    % Plot istogramma errore
    figure(1);
    histogram(vout - vx, 50);
    xlabel('Error (V)');
    ylabel('Counts (1)');
    grid on;
end

function adc_code = adc_model(vx, bits, LSB, noise_rms)
    % Aggiunge rumore gaussiano
    noise = randn(size(vx)) * noise_rms;
    vx_noisy = vx + noise;
    
    % Quantizzazione ADC ideale
    adc_code = round(vx_noisy / LSB);
    adc_code = min(max(adc_code, 0), 2^bits - 1);
end

function vx = dac_model(code, bits, LSB)
    % DAC ideale
    code = min(max(code, 0), 2^bits - 1);
    vx = code * LSB;
end
