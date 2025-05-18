# Entropía del doc
import math
from collections import defaultdict
import cv2
import numpy as np
import heapq


def calcular_entropia(archivo):
    with open(archivo, 'rb') as f:
        data = f.read()
    
    # Calcular frecuencias
    frecuencias = defaultdict(int)
    total = len(data)
    for byte in data:
        frecuencias[byte] += 1
    
    # Calcular probabilidades y entropía
    entropia = 0.0
    for count in frecuencias.values():
        p = count / total
        entropia -= p * math.log2(p)
    
    return entropia, frecuencias, total


# Shannon-Fano
def construir_arbol_shannon_fano(frecuencias):
    simbolos = sorted(frecuencias.items(), key=lambda x: (-x[1], x[0]))
    
    def dividir(simbolos):
        if len(simbolos) == 1:
            return {'simbolo': simbolos[0][0], 'izq': None, 'der': None}
        
        total = sum(freq for _, freq in simbolos)
        acum = 0
        for i, (_, freq) in enumerate(simbolos):
            acum += freq
            if acum >= total / 2:
                izq = dividir(simbolos[:i+1])
                der = dividir(simbolos[i+1:])
                return {'izq': izq, 'der': der, 'simbolo': None}
    
    return dividir(simbolos)

def generar_codigos_shannon_fano(nodo, codigo_actual="", codigos={}):
    if nodo['simbolo'] is not None:
        codigos[nodo['simbolo']] = codigo_actual
    else:
        generar_codigos_shannon_fano(nodo['izq'], codigo_actual + "0", codigos)
        generar_codigos_shannon_fano(nodo['der'], codigo_actual + "1", codigos)
    return codigos

def comprimir_shannon_fano(data, codigos):
    bits = ''.join([codigos[byte] for byte in data])
    # Convertir a bytes
    padding = 8 - (len(bits) % 8)
    bits += '0' * padding
    bytes_comprimidos = bytes([int(bits[i:i+8], 2) for i in range(0, len(bits), 8)])
    return bytes_comprimidos, padding



# Huffman
import heapq

class NodoHuffman:
    def __init__(self, simbolo=None, freq=0):
        self.simbolo = simbolo
        self.freq = freq
        self.izq = None
        self.der = None
    
    def __lt__(self, otro):
        return self.freq < otro.freq

def construir_arbol_huffman(frecuencias):
    heap = [NodoHuffman(simbolo, freq) for simbolo, freq in frecuencias.items()]
    heapq.heapify(heap)
    
    while len(heap) > 1:
        izq = heapq.heappop(heap)
        der = heapq.heappop(heap)
        padre = NodoHuffman(freq=izq.freq + der.freq)
        padre.izq = izq
        padre.der = der
        heapq.heappush(heap, padre)
    
    return heap[0]

def generar_codigos_huffman(nodo, codigo_actual="", codigos={}):
    if nodo.simbolo is not None:
        codigos[nodo.simbolo] = codigo_actual
    else:
        generar_codigos_huffman(nodo.izq, codigo_actual + "0", codigos)
        generar_codigos_huffman(nodo.der, codigo_actual + "1", codigos)
    return codigos

def comprimir_huffman(data, codigos):
    bits = ''.join([codigos[byte] for byte in data])
    padding = 8 - (len(bits) % 8)
    bits += '0' * padding
    bytes_comprimidos = bytes([int(bits[i:i+8], 2) for i in range(0, len(bits), 8)])
    return bytes_comprimidos, padding


# LZW
def comprimir_lzw(data):
    diccionario = {bytes([i]): i for i in range(256)}
    proximo_indice = 256
    actual = bytes([data[0]])
    comprimido = []
    
    for byte in data[1:]:
        nuevo = actual + bytes([byte])
        if nuevo in diccionario:
            actual = nuevo
        else:
            comprimido.append(diccionario[actual])
            diccionario[nuevo] = proximo_indice
            proximo_indice += 1
            actual = bytes([byte])
    
    comprimido.append(diccionario[actual])
    # Convertir a bytes (asumiendo códigos de 12 bits)
    bits = []
    for codigo in comprimido:
        bits.extend([(codigo >> 4) & 0xFF, (codigo << 4) & 0xF0])
    bytes_comprimidos = bytes(bits)
    return bytes_comprimidos





# =====================================================
# Funciones para cálculo de entropía (similares a tu código original)
# =====================================================

def compute_stationary_distribution(img):
    hist, _ = np.histogram(img, bins=256, range=(0, 256))
    total = hist.sum()
    return hist / total

def entropy_memoryless(img):
    p = compute_stationary_distribution(img)
    return -np.sum(p[p > 0] * np.log2(p[p > 0]))

# =====================================================
# Funciones para compresión (adaptadas para imágenes)
# =====================================================

def get_pixel_data(img):
    """Convierte la imagen 2D en una lista plana de bytes (píxeles)."""
    return img.flatten().tolist()

# --------------------- Shannon-Fano ---------------------
def shannon_fano_compress(pixel_data):
    # Calcular frecuencias
    frecuencias = defaultdict(int)
    for pixel in pixel_data:
        frecuencias[pixel] += 1

    # Construir árbol y códigos (usando tus funciones)
    arbol = construir_arbol_shannon_fano(frecuencias)
    codigos = generar_codigos_shannon_fano(arbol)
    
    # Comprimir
    bits = ''.join([codigos[pixel] for pixel in pixel_data])
    padding = 8 - (len(bits) % 8)
    bits += '0' * padding
    bytes_comprimidos = bytes([int(bits[i:i+8], 2) for i in range(0, len(bits), 8)])
    return bytes_comprimidos, padding, codigos

# --------------------- Huffman ---------------------
def huffman_compress(pixel_data):
    # Calcular frecuencias
    frecuencias = defaultdict(int)
    for pixel in pixel_data:
        frecuencias[pixel] += 1

    # Construir árbol y códigos (usando tus funciones)
    arbol = construir_arbol_huffman(frecuencias)
    codigos = generar_codigos_huffman(arbol)
    
    # Comprimir
    bits = ''.join([codigos[pixel] for pixel in pixel_data])
    padding = 8 - (len(bits) % 8)
    bits += '0' * padding
    bytes_comprimidos = bytes([int(bits[i:i+8], 2) for i in range(0, len(bits), 8)])
    return bytes_comprimidos, padding, codigos

# --------------------- LZW ---------------------
def lzw_compress(pixel_data):
    diccionario = {bytes([i]): i for i in range(256)}
    proximo_indice = 256
    actual = bytes([pixel_data[0]])
    comprimido = []
    
    for pixel in pixel_data[1:]:
        nuevo = actual + bytes([pixel])
        if nuevo in diccionario:
            actual = nuevo
        else:
            comprimido.append(diccionario[actual])
            diccionario[nuevo] = proximo_indice
            proximo_indice += 1
            actual = bytes([pixel])
    
    comprimido.append(diccionario[actual])
    # Empaquetar códigos en bytes (simplificado para códigos de 12 bits)
    bits = []
    for codigo in comprimido:
        bits.extend([(codigo >> 4) & 0xFF, (codigo << 4) & 0xF0])
    return bytes(bits)

# =====================================================
# Procesamiento de imágenes y comparación
# =====================================================

def procesar_imagen(imagen_path):
    # Cargar imagen y convertir a escala de grises (720p)
    img = cv2.imread(imagen_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (1280, 720))
    
    # Obtener datos de píxeles
    pixel_data = get_pixel_data(img)
    tam_original = len(pixel_data)  # 1 byte por píxel
    
    # Calcular entropía sin memoria
    entropia = entropy_memoryless(img)
    
    # Comprimir con los tres métodos
    compressed_sf, padding_sf, _ = shannon_fano_compress(pixel_data)
    compressed_huf, padding_huf, _ = huffman_compress(pixel_data)
    compressed_lzw = lzw_compress(pixel_data)
    
    # Calcular tamaños
    tasas = {
        'Shannon-Fano': len(compressed_sf),
        'Huffman': len(compressed_huf),
        'LZW': len(compressed_lzw),
        'Original': tam_original
    }
    
    # Calcular eficiencia (Entropía / (tamaño_comprimido / tamaño_original))
    eficiencias = {}
    for metodo in ['Shannon-Fano', 'Huffman', 'LZW']:
        tasa_compresion = tam_original / tasas[metodo]
        eficiencia = entropia / (tasas[metodo] * 8 / tam_original)  # bits por píxel
        eficiencias[metodo] = eficiencia
    
    return tasas, eficiencias

# =====================================================
# Ejecución y resultados
# =====================================================

image_files = ["imagen_suave.jpg", "imagen_media.jpg", "imagen_variada.jpg"]

for imagen in image_files:
    print(f"\nProcesando {imagen}:")
    tasas, eficiencias = procesar_imagen(imagen)
    print("Tasas de compresión:", tasas)
    print("Eficiencias relativas a la entropía:", eficiencias)
    # Guardar archivos comprimidos (ejemplo para Huffman)
    #with open(f"{imagen}_huffman.bin", "wb") as f:
     #   f.write(compressed_huf)
