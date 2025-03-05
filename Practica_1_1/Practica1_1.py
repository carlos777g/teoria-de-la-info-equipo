import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import pandas as pd
import os

# Configuración inicial
audio_files = ['Come_As_You_Are.wav', 'Nocturne_in_C.wav', 'WILDFLOWER.wav']
genres = ['Rock', 'Clásica', 'Pop']
n_bits = 16
n_levels = 2 ** n_bits  # 65536 niveles
entropies = []

for idx, filename in enumerate(audio_files):
    # Leer archivo de audio
    fs, data = wavfile.read(filename)
    
    # Convertir a mono si es estéreo
    if len(data.shape) > 1:
        data = data[:, 0]
        
    # Normalización
    data_norm = data / np.max(np.abs(data))
    
    # Cuantización a 16 bits
    data_quant = np.floor((data_norm + 1) / 2 * (n_levels - 1)).astype(int)
    
    # Crear alfabeto
    alfabeto = np.arange(n_levels)
    
    # Calcular histograma
    counts, _ = np.histogram(data_quant, bins=np.arange(-0.5, n_levels + 0.5))
    total_muestras = len(data_quant)
    probabilities = counts / total_muestras
    
    # Calcular entropía
    non_zero = probabilities > 0
    H = -np.sum(probabilities[non_zero] * np.log2(probabilities[non_zero]))
    entropies.append(H)
    
    # Imprimir información
    print(f'Archivo: {filename} (Género: {genres[idx]})')
    print(f'Frecuencia de muestreo: {fs} Hz, Total de muestras: {total_muestras}')
    print(f'Entropía: {H:.4f} bits/símbolo\n')
    
    # Guardar tabla de probabilidades
    df = pd.DataFrame({'Nivel': alfabeto, 'Probabilidad': probabilities})
    csv_filename = os.path.splitext(filename)[0] + '_tabla_probabilidades.csv'
    df.to_csv(csv_filename, index=False)
    
    # Niveles con mayor probabilidad
    sorted_indices = np.argsort(probabilities)[::-1]
    top7_levels = alfabeto[sorted_indices[:7]]
    top7_probs = probabilities[sorted_indices[:7]]
    top7_info = np.log2(1 / top7_probs)
    
    # Niveles con menor probabilidad (no cero)
    non_zero_indices = np.where(probabilities > 0)[0]
    non_zero_probs = probabilities[non_zero_indices]
    sorted_non_zero = np.argsort(non_zero_probs)
    bottom7_levels = non_zero_indices[sorted_non_zero[:7]]
    bottom7_probs = non_zero_probs[sorted_non_zero[:7]]
    bottom7_info = np.log2(1 / bottom7_probs)
    
    # Mostrar tablas
    print(f'Archivo: {filename} - Top 7 niveles:')
    print(pd.DataFrame({'Nivel': top7_levels, 
                       'Probabilidad': top7_probs, 
                       'Informacion': top7_info}))
    print(f'\nArchivo: {filename} - Bottom 7 niveles:')
    print(pd.DataFrame({'Nivel': bottom7_levels, 
                       'Probabilidad': bottom7_probs, 
                       'Informacion': bottom7_info}))
    
    # Graficar
    plt.figure(figsize=(12, 6))
    
    # Subplot izquierdo (Top 7)
    plt.subplot(1, 2, 1)
    plt.bar(top7_levels, top7_info, color='#33CC99')
    plt.xlabel('Nivel')
    plt.ylabel('Información (bits)')
    plt.title(f'Top 7 niveles - {genres[idx]}')
    if filename == 'Come_As_You_Are.wav':
        plt.xlim(32760, 32773)
    plt.grid(True)
    
    # Subplot derecho (Bottom 7)
    plt.subplot(1, 2, 2)
    plt.bar(bottom7_levels, bottom7_info, color='#FF6666')
    plt.xlabel('Nivel')
    plt.ylabel('Información (bits)')
    plt.title(f'Bottom 7 niveles - {genres[idx]}')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# Gráfico comparativo de entropías
plt.figure()
plt.bar(genres, entropies, color='#66B2FF')
plt.xlabel('Género')
plt.ylabel('Entropía (bits/símbolo)')
plt.title('Comparación de Entropías entre Géneros')
plt.grid(True)
plt.show()