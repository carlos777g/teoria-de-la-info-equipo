%clear all;
close all;
clc;
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

% Lista de archivos y sus géneros (puedes ajustarlos)
audioFiles = {'Come_As_You_Are.wav', 'Nocturne_in_C.wav', 'WILDFLOWER.wav'};
genres = {'Rock', 'Clásica', 'Pop'};

nBits = 16;              % Usamos 16 bits
nLevels = 2^nBits;       % 65,536 niveles

% Inicializamos un vector para guardar la entropía de cada archivo
entropies = zeros(length(audioFiles), 1);

for i = 1:length(audioFiles)
    filename = audioFiles{i};
    [x, fs] = audioread(filename);
    
    % Si el audio es estéreo, seleccionamos solo el primer canal
    if size(x, 2) > 1
        x = x(:, 1);
    end
    
    % Normalización: escalamos la señal para que el máximo valor absoluto sea 1
    x = x / max(abs(x));
    
    % Cuantización a 16 bits: mapeamos de [-1,1] a [0, 65535]
    x_quant = floor((x + 1) / 2 * (nLevels - 1));
    
    % Definir el alfabeto: niveles de 0 a 65535
    alfabeto = (0:nLevels-1)';  % vector columna
    
    % Contar ocurrencias de cada nivel
    counts = histcounts(x_quant, -0.5:1:(nLevels-1)+0.5);
    totalMuestras = length(x_quant);
    probabilities = counts / totalMuestras;
    
    % Calcular la entropía (solo se consideran los niveles que aparecen, es decir, P>0)
    nonzero = probabilities > 0;
    H = -sum(probabilities(nonzero) .* log2(probabilities(nonzero)));
    entropies(i) = H;
    
    fprintf('Archivo: %s (Género: %s)\n', filename, genres{i});
    fprintf('Frecuencia de muestreo: %d Hz, Total de muestras: %d\n', fs, totalMuestras);
    fprintf('Entropía: %.4f bits/símbolo\n\n', H);
    
    % Guardar la tabla completa de probabilidades
    T = table(alfabeto, probabilities', 'VariableNames', {'Nivel', 'Probabilidad'});
    csvFileName = strcat(filename(1:end-4), '_tabla_probabilidades.csv');
    writetable(T, csvFileName);
    
    % Extraer los 7 niveles con mayor probabilidad
    [sortedProbs, sortedIndices] = sort(probabilities, 'descend');
    top7Levels = alfabeto(sortedIndices(1:7));
    top7Probs = sortedProbs(1:7);
    % Calcular la información para esos niveles
    top7Info = log2(1 ./ top7Probs);
    
    % Extraer los 7 niveles con menor probabilidad (descartando los que no aparecen)
    nonzeroIndices = find(probabilities > 0); % índices de niveles presentes
    nonzeroProbs = probabilities(probabilities > 0);
    [sortedNonzeroProbs, sortedNonzeroOrder] = sort(nonzeroProbs, 'ascend');
    bottom7Levels = alfabeto(nonzeroIndices(sortedNonzeroOrder(1:7)));
    bottom7Probs = sortedNonzeroProbs(1:7);
    bottom7Info = log2(1 ./ bottom7Probs);
    
    % Mostrar en consola las tablas de top7 y bottom7
    T_top7 = table(top7Levels, top7Probs', top7Info', 'VariableNames', {'Nivel', 'Probabilidad', 'Informacion'});
    T_bottom7 = table(bottom7Levels, bottom7Probs', bottom7Info', 'VariableNames', {'Nivel', 'Probabilidad', 'Informacion'});
    disp(['Archivo: ', filename, ' - Top 7 niveles:']);
    disp(T_top7);
    disp(['Archivo: ', filename, ' - Bottom 7 niveles:']);
    disp(T_bottom7);
    
    % Graficar para este audio los 7 niveles con mayor y menor probabilidad
    figure;
    subplot(1,2,1)
    bar(top7Levels, top7Info, 'FaceColor', [0.2 0.6 0.5])
    xlabel('Nivel')
    ylabel('Información (bits)')
    title(sprintf('Top 7 niveles - %s', genres{i}))
    if(strcmp(filename, 'Come_As_You_Are.wav'))
        xlim([32760 32773])
    end
    grid on;
    
    subplot(1,2,2)
    bar(bottom7Levels, bottom7Info, 'FaceColor', [0.8 0.4 0.4])
    xlabel('Nivel')
    ylabel('Información (bits)')
    title(sprintf('Bottom 7 niveles - %s', genres{i}))
    grid on;
end

% Comparar las entropías de cada género en un gráfico de barras
figure;
bar(categorical(genres), entropies, 'FaceColor', [0.4 0.6 0.8])
xlabel('Género')
ylabel('Entropía (bits/símbolo)')
title('Comparación de Entropías entre Géneros')
grid on;
