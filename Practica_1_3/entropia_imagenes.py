import cv2
import numpy as np
import math
import heapq
from collections import defaultdict

# =================================================================
# Funciones comunes
# =================================================================
def calcular_entropia(imagen_gris):
    hist, _ = np.histogram(imagen_gris, bins=256, range=(0, 256))
    probabilidades = hist / hist.sum()
    return -np.sum(probabilidades[probabilidades > 0] * np.log2(probabilidades[probabilidades > 0]))

def procesar_imagen(imagen_path):
    img = cv2.imread(imagen_path, cv2.IMREAD_COLOR)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.resize(img_gray, (1280, 720))
    pixel_data = img_gray.flatten().tolist()
    return img_gray, pixel_data

# =================================================================
# Algoritmo de Shannon-Fano
# =================================================================
def shannon_fano(frecuencias):
    simbolos = sorted(frecuencias.items(), key=lambda x: (-x[1], x[0]))
    def dividir(simbolos):
        if len(simbolos) == 1: return {'simbolo': simbolos[0][0], 'izq': None, 'der': None}
        total = sum(freq for _, freq in simbolos)
        acum, split_idx = 0, 0
        for i, (_, freq) in enumerate(simbolos):
            acum += freq
            if acum >= total / 2: split_idx = i + 1; break
        return {'izq': dividir(simbolos[:split_idx]), 'der': dividir(simbolos[split_idx:]), 'simbolo': None}
    arbol = dividir(simbolos)
    codigos = {}
    def generar_codigos(nodo, codigo=""):
        if nodo['simbolo'] is not None: codigos[nodo['simbolo']] = codigo
        else: generar_codigos(nodo['izq'], codigo + "0"); generar_codigos(nodo['der'], codigo + "1")
    generar_codigos(arbol)
    return codigos

# =================================================================
# Algoritmo de Huffman
# =================================================================
class NodoHuffman:
    def __init__(self, simbolo=None, freq=0):
        self.simbolo, self.freq = simbolo, freq
        self.izq, self.der = None, None
    def __lt__(self, otro): return self.freq < otro.freq

def huffman(frecuencias):
    heap = [NodoHuffman(simbolo, freq) for simbolo, freq in frecuencias.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        izq = heapq.heappop(heap); der = heapq.heappop(heap)
        padre = NodoHuffman(freq=izq.freq + der.freq)
        padre.izq, padre.der = izq, der
        heapq.heappush(heap, padre)
    codigos = {}
    def generar_codigos(nodo, codigo=""):
        if nodo.simbolo is not None: codigos[nodo.simbolo] = codigo
        else: generar_codigos(nodo.izq, codigo + "0"); generar_codigos(nodo.der, codigo + "1")
    generar_codigos(heap[0])
    return codigos

# =================================================================
# Algoritmo LZW
# =================================================================
def lzw_compress(pixel_data):
    diccionario = {bytes([i]): i for i in range(256)}
    proximo_codigo = 256
    entrada_actual = bytes([pixel_data[0]])
    comprimido = []
    
    for pixel in pixel_data[1:]:
        nuevo = entrada_actual + bytes([pixel])
        if nuevo in diccionario:
            entrada_actual = nuevo
        else:
            comprimido.append(diccionario[entrada_actual])
            if proximo_codigo < 4096:
                diccionario[nuevo] = proximo_codigo
                proximo_codigo += 1
            entrada_actual = bytes([pixel])
    
    comprimido.append(diccionario[entrada_actual])
    # Convertir códigos a bytes (12 bits por código)
    bytes_comprimidos = bytearray()
    for i in range(0, len(comprimido), 2):
        codigo1 = comprimido[i]
        codigo2 = comprimido[i+1] if i+1 < len(comprimido) else 0
        byte1 = (codigo1 >> 4) & 0xFF
        byte2 = ((codigo1 & 0x0F) << 4) | ((codigo2 >> 8) & 0x0F)
        byte3 = codigo2 & 0xFF
        bytes_comprimidos.extend([byte1, byte2, byte3])
    return bytes(bytes_comprimidos)

# =================================================================
# Cálculo de métricas
# =================================================================
def calcular_L(probabilidades, codigos):
    return sum(prob * len(codigos[simbolo]) for simbolo, prob in enumerate(probabilidades) if prob > 0)

def calcular_eficiencias(img_gray, pixel_data, H, compressed_lzw):
    # Tamaño original (8 bits por píxel)
    tam_original = len(pixel_data) * 8
    # Tamaño comprimido (bits)
    tam_comprimido = len(compressed_lzw) * 8
    # Tasa de compresión (bits por píxel)
    tasa_compresion = tam_comprimido / len(pixel_data)
    eficiencia = H / tasa_compresion if tasa_compresion != 0 else 0
    return tasa_compresion, eficiencia

# =================================================================
# Resultados
# =================================================================
image_files = ["imagen_suave.jpg", "imagen_media.jpg", "imagen_variada.jpg"]
descriptions = ["Imagen Suave", "Imagen Media", "Imagen Variada"]

for idx, file in enumerate(image_files):
    img_gray, pixel_data = procesar_imagen(file)
    H = calcular_entropia(img_gray)
    
    # Calcular frecuencias para Shannon-Fano y Huffman
    hist, _ = np.histogram(img_gray, bins=256, range=(0, 256))
    probabilidades = hist / hist.sum()
    frecuencias = {i: prob for i, prob in enumerate(probabilidades) if prob > 0}
    
    # Shannon-Fano
    codigos_sf = shannon_fano(frecuencias)
    L_sf = calcular_L(probabilidades, codigos_sf)
    eficiencia_sf = H / L_sf
    
    # Huffman
    codigos_hf = huffman(frecuencias)
    L_hf = calcular_L(probabilidades, codigos_hf)
    eficiencia_hf = H / L_hf
    
    # LZW
    compressed_lzw = lzw_compress(pixel_data)
    lzw_bpp, eficiencia_lzw = calcular_eficiencias(img_gray, pixel_data, H, compressed_lzw)
    
    print(f"\n{descriptions[idx]} ({file}):")
    print(f"  Entropía H(S): {H:.4f} bits/símbolo")
    print("  Shannon-Fano:")
    print(f"    L = {L_sf:.4f} bits/símbolo | Eficiencia = {eficiencia_sf:.4f}")
    print("  Huffman:")
    print(f"    L = {L_hf:.4f} bits/símbolo | Eficiencia = {eficiencia_hf:.4f}")
    print("  LZW:")
    print(f"    Tasa de compresión = {lzw_bpp:.4f} bits/píxel | Eficiencia = {eficiencia_lzw:.4f}")