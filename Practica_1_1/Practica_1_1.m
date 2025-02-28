%clear all;
%close all;
%% Información de las canciones seleccionadas
% Come_As_You_Are.wav Fs_rock= 48000Hz a 16 bits
% Nocturne_in_C.wav Fs_clasic = 48000Hz a 16 bits
% WILDFLOWER.wav Fs_pop = 48000Hz a 16 bits
%% Para saber frecuencia y bits por muestra
% [y, Fs_Come] = audioread('Come_As_You_Are.wav');
% info = audioinfo('WILDFLOWER.wav');
% disp(info.BitsPerSample);
% 
% [y, Fs_Nocturne] = audioread('Nocturne_No_2.wav');
% info = audioinfo('WILDFLOWER.wav');
% disp(info.BitsPerSample);
% 
% [y, Fs_Wilflower] = audioread('WILDFLOWER.wav');
% info = audioinfo('WILDFLOWER.wav');
% disp(info.BitsPerSample);
%% Inicio de práctica 1.1
% Lista de archivos de audio y configuración
audioFiles = {'Come_As_You_Are.wav', 'Nocturne_in_C.wav', 'WILDFLOWER.wav'};
nBits = 16;              % Usamos 16 bits
nLevels = 2^nBits;       % 65,536 niveles

for i = 1:length(audioFiles)
    filename = audioFiles{i};
    [x, fs] = audioread(filename);
    
    % Si el audio es estéreo, seleccionamos el primer canal
    if size(x, 2) > 1
        x = x(:, 1);
    end
    
    % Normalización: escalamos la señal para que el máximo absoluto sea 1
    x = x / max(abs(x));
    
    % Cuantización a 16 bits: mapeamos de [-1,1] a [0, 65535]
    x_quant = floor((x + 1) / 2 * (nLevels - 1));
    
    % Definir el alfabeto: niveles de 0 a 65535
    alfabeto = (0:nLevels-1)';  % vector columna
    
    % Contar las ocurrencias de cada nivel usando histcounts
    counts = histcounts(x_quant, -0.5:1:(nLevels-1)+0.5);
    
    % Calcular la probabilidad de cada símbolo (nivel)
    totalMuestras = length(x_quant);
    probabilities = counts / totalMuestras;
    
    % Crear una tabla completa con cada nivel y su probabilidad
    T = table(alfabeto, probabilities', 'VariableNames', {'Nivel', 'Probabilidad'});
    
    % Mostrar información básica en la consola
    fprintf('Archivo: %s, Frecuencia de muestreo: %d Hz\n', filename, fs);
    fprintf('Total de muestras: %d\n', totalMuestras);
    
    % Encontrar los 7 niveles con mayor probabilidad
    [sortedProbs, sortedIndices] = sort(probabilities, 'descend');
    top7Levels = alfabeto(sortedIndices(1:7));
    top7Probs = sortedProbs(1:7)';
    
    % Crear y mostrar la tabla para los 7 niveles más frecuentes
    T_top7 = table(top7Levels, top7Probs, 'VariableNames', {'Nivel', 'Probabilidad'});
    disp('Los 7 niveles con mayor probabilidad:');
    disp(T_top7);
    
    % (Opcional) Guardar la tabla completa en un archivo CSV para análisis posterior
    % csvFileName = strcat(filename(1:end-4), '_tabla_probabilidades.csv');
    % writetable(T, csvFileName);
    
    % Para evitar log2(0), solo se consideran los símbolos con probabilidad mayor a cero:
    H = -sum(probabilities(probabilities > 0) .* log2(probabilities(probabilities > 0)));
    fprintf('Entropía: %.4f bits/símbolo\n', H);

    fprintf('\n');
end
