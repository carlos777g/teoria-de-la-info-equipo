%% Configuración de archivos y parámetros
textFiles = {'espanol.txt', 'ingles.txt', 'portugues.txt'};
languages = {'Español', 'Inglés', 'Portugués'};

% Ordenes de extensión a analizar (agrupación de n palabras)
orders = [1, 2, 3, 4];

% Matriz para almacenar las entropías: filas para cada idioma, columnas para cada orden
entropyMatrix = zeros(length(textFiles), length(orders));

%% Procesamiento para cada archivo
for fileIdx = 1:length(textFiles)
    filename = textFiles{fileIdx};
    lang = languages{fileIdx};
    
    % Leer el contenido del archivo
    textContent = fileread(filename);
    
    % Convertir a minúsculas (para evitar duplicados por mayúsculas)
    textContent = lower(textContent);
    
    % Separar en palabras (suponiendo que solo hay espacios como separador)
    words = strsplit(textContent);
    % Eliminar celdas vacías (por si hay espacios extra)
    words = words(~cellfun('isempty', words));
    
    fprintf('=========================================\n');
    fprintf('Archivo: %s (Idioma: %s)\n', filename, lang);
    
    %% Para cada extensión de orden n (agrupación de n palabras)
    for orderIdx = 1:length(orders)
        n = orders(orderIdx);
        
        % Crear los n-gramas (agrupaciones consecutivas)
        nGrams = cell(1, length(words)-n+1);
        for i = 1:(length(words)-n+1)
            nGrams{i} = strjoin(words(i:i+n-1), ' ');
        end
        
        % Contar ocurrencias únicas usando unique y accumarray
        [uniqueNGrams, ~, idx] = unique(nGrams);
        counts = accumarray(idx, 1);
        totalNGrams = sum(counts);
        
        % Calcular las probabilidades de cada n-grama
        probabilities = counts / totalNGrams;
        
        % Calcular la información teórica para cada n-grama
        information = -log2(probabilities);
        
        % Calcular la entropía de la fuente extendida
        entropyVal = sum(probabilities .* information);
        entropyMatrix(fileIdx, orderIdx) = entropyVal;
        
        % Ordenar los n-gramas según su probabilidad (de mayor a menor)
        [sortedProbs, sortIdx] = sort(probabilities, 'descend');
        top5NGrams = uniqueNGrams(sortIdx(1:min(5,end)));
        top5Probs = sortedProbs(1:min(5,end));
        top5Info = -log2(top5Probs);  % Equivalente a information para esos
        
        % Mostrar resultados en consola
        fprintf('\nOrden (agrupación de %d palabras):\n', n);
        fprintf('Entropía: %.4f bits por n-grama\n', entropyVal);
        fprintf('Top 5 n-gramas con mayor probabilidad:\n');
        for j = 1:length(top5NGrams)
            fprintf('%d. "%s" -> Probabilidad: %.8f, Información: %.4f bits\n', ...
                j, top5NGrams{j}, top5Probs(j), top5Info(j));
        end
        fprintf('-----------------------------------------\n');
    end
end

%% Comparación gráfica de entropías
% Cada fila corresponde a un idioma y cada columna a un orden (n)
figure;
bar(categorical(languages), entropyMatrix)
xlabel('Idioma')
ylabel('Entropía (bits/n-grama)')
title('Comparación de Entropía de fuentes extendidas')
legend("Orden 1", "Orden 2", "Orden 3", "Orden 4", 'Location', 'Best')
grid on;
