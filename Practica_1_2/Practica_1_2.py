import os
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import math

# Configuración de archivos y parámetros
text_files = ['espanol.txt', 'ingles.txt', 'portugues.txt']
languages = ['Español', 'Inglés', 'Portugués']
orders = [1, 2, 3, 4]  # Ordenes: 1 = palabras individuales, 2 = bigramas, etc.

# Matriz para almacenar las entropías: filas para cada idioma, columnas para cada orden
entropy_matrix = np.zeros((len(text_files), len(orders)))

for file_idx, filename in enumerate(text_files):
    lang = languages[file_idx]
    
    # Leer el contenido del archivo (se asume codificación UTF-8)
    with open(filename, 'r', encoding='utf-8') as f:
        text_content = f.read()
    
    # Convertir a minúsculas
    text_content = text_content.lower()
    
    # Separar en palabras (asumiendo espacios como separador)
    words = text_content.split()
    # Eliminar posibles entradas vacías
    words = [w for w in words if w]
    
    print("=========================================")
    print(f"Archivo: {filename} (Idioma: {lang})")
    
    # Procesamiento para cada orden de extensión
    for order_idx, n in enumerate(orders):
        # Crear los n-gramas (agrupaciones consecutivas de n palabras)
        ngrams = [" ".join(words[i:i+n]) for i in range(len(words)-n+1)]
        
        # Contar ocurrencias únicas de cada n-grama
        counts = Counter(ngrams)
        total_ngrams = sum(counts.values())
        
        # Calcular la entropía y guardar la información de cada n-grama
        entropy_val = 0.0
        ngram_info = {}
        for ngram, count in counts.items():
            p = count / total_ngrams
            info = -math.log2(p)
            ngram_info[ngram] = (p, info)
            entropy_val += p * info
        entropy_matrix[file_idx, order_idx] = entropy_val
        
        # Obtener los 5 n-gramas con mayor probabilidad (ordenados de mayor a menor)
        sorted_ngrams = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        top5 = sorted_ngrams[:5]
        
        print(f"\nOrden (agrupación de {n} palabra{'s' if n > 1 else ''}):")
        print(f"Entropía: {entropy_val:.4f} bits por n-grama")
        print("Top 5 n-gramas con mayor probabilidad:")
        for j, (ngram, count) in enumerate(top5, start=1):
            p = count / total_ngrams
            info = -math.log2(p)
            print(f"{j}. \"{ngram}\" -> Probabilidad: {p:.8f}, Información: {info:.4f} bits")
        print("-----------------------------------------")

# Comparación gráfica de entropías
x = np.arange(len(languages))
width = 0.2  # ancho de cada barra

fig, ax = plt.subplots()
for i, n in enumerate(orders):
    ax.bar(x + i * width, entropy_matrix[:, i], width, label=f"Orden {n}")

ax.set_xlabel("Idioma")
ax.set_ylabel("Entropía (bits/n-grama)")
ax.set_title("Comparación de Entropía de fuentes extendidas")
ax.set_xticks(x + width * (len(orders) - 1) / 2)
ax.set_xticklabels(languages)
ax.legend()
ax.grid(True)
plt.show()
