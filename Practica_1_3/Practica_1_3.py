import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

def compute_transition_matrix(img, d=1):
    """
    Calcula la matriz de transición para un desplazamiento d en la dirección horizontal.
    La imagen debe ser en escala de grises (valores 0-255).
    """
    height, width = img.shape
    T_counts = np.zeros((256, 256), dtype=np.float64)
    for row in range(height):
        # Para cada fila, recorremos desde la columna 0 hasta width-d-1
        for col in range(width - d):
            current_val = img[row, col]
            next_val = img[row, col+d]
            T_counts[current_val, next_val] += 1
    return T_counts

def compute_conditional_entropy(T_counts):
    """
    Para cada valor i (fila de la matriz de transición),
    calcula H(i) = - sum_j P(j|i) log2(P(j|i)).
    Retorna un vector de entropías condicionales (de largo 256).
    """
    cond_entropy = np.zeros(256)
    row_sums = T_counts.sum(axis=1)
    for i in range(256):
        if row_sums[i] > 0:
            p_row = T_counts[i, :] / row_sums[i]
            p_nonzero = p_row[p_row > 0]
            cond_entropy[i] = -np.sum(p_nonzero * np.log2(p_nonzero))
    return cond_entropy

def compute_stationary_distribution(img):
    """
    Calcula la distribución estacionaria (histograma normalizado) de la imagen en escala de grises.
    """
    hist, _ = np.histogram(img, bins=256, range=(0,256))
    total = hist.sum()
    return hist / total

# Lista de imágenes (cambia los nombres de archivo según corresponda)
image_files = ["imagen_suave.jpg", "imagen_media.jpg", "imagen_variada.jpg"]
descriptions = ["Imagen Suave", "Imagen Media", "Imagen Variada"]

# Displacement para transiciones que se desean analizar
displacements = [1, 2, 10, 20, 100]

# Para almacenar la entropía (tasa) calculada con memoria (d=1)
entropy_rates = []

for idx, file in enumerate(image_files):
    # Cargar la imagen (se asume que es RGB)
    img_rgb = cv2.imread(file)
    if img_rgb is None:
        print(f"No se pudo leer la imagen: {file}")
        continue

    # Convertir a escala de grises (8 bits)
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    
    # Asegurar resolución 720p (1280x720). Si no es ese tamaño, se redimensiona.
    img_gray = cv2.resize(img_gray, (1280, 720))
    
    print("=========================================")
    print(f"Procesando {descriptions[idx]}: {file}")
    print(f"Resolución: {img_gray.shape[1]}x{img_gray.shape[0]}")
    
    # Fuente sin memoria: histograma global y entropía
    p_stationary = compute_stationary_distribution(img_gray)
    entropy_memoryless = -np.sum(p_stationary[p_stationary > 0] * np.log2(p_stationary[p_stationary > 0]))
    print(f"Entropía de la fuente sin memoria (por pixel): {entropy_memoryless:.4f} bits")
    
    # Fuente con memoria de orden 1 (transición entre píxeles vecinos horizontales)
    T_counts = compute_transition_matrix(img_gray, d=1)
    cond_entropy = compute_conditional_entropy(T_counts)
    # La entropía de la fuente con memoria (tasa de entropía) se estima como:
    # H = sum_{i} p(i) * H(P(.|i))
    entropy_rate = np.sum(p_stationary * cond_entropy)
    print(f"Entropía de la fuente con memoria (d=1): {entropy_rate:.4f} bits por pixel")
    entropy_rates.append(entropy_rate)
    
    # Para cada desplazamiento adicional (d=2, 10, 20, 100) se calculan algunas transiciones.
    # Aquí se ilustra, por ejemplo, la probabilidad de que, dado un cierto tono, el píxel a distancia d tenga ese mismo tono.
    most_common_pixel = np.argmax(p_stationary)  # tono más frecuente en la imagen
    for d in displacements:
        if d == 1:
            continue  # ya se calculó d=1
        T_counts_d = compute_transition_matrix(img_gray, d=d)
        row_sums = T_counts_d.sum(axis=1)
        if row_sums[most_common_pixel] > 0:
            p_transition = T_counts_d[most_common_pixel, :] / row_sums[most_common_pixel]
            prob_same = p_transition[most_common_pixel]
            print(f"Para desplazamiento d={d}, P(tono {most_common_pixel} -> {most_common_pixel}) = {prob_same:.6f}")
        else:
            print(f"Para desplazamiento d={d}, no se encontraron transiciones para el tono {most_common_pixel}.")
    
    # El mínimo de bits por píxel (en codificación óptima) se aproxima a la entropía de la fuente con memoria.
    print(f"Mínimo de bits por pixel para codificar esta imagen: {entropy_rate:.4f} bits\n")

# Comparar las entropías (d=1) de las 3 imágenes con un gráfico de barras
plt.figure()
plt.bar(range(len(entropy_rates)), entropy_rates, tick_label=descriptions)
plt.xlabel("Tipo de Imagen")
plt.ylabel("Entropía (bits por pixel)")
plt.title("Comparación de Entropía (fuente con memoria, d=1)")
plt.grid(True)
plt.show()
