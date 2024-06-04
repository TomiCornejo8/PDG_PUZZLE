import numpy as np
from itertools import combinations
import os
import glob
import time as T

def read_solutions(folder_path):
    # Obtén la lista de todos los archivos CSV en la carpeta
    csv_files = glob.glob(os.path.join(folder_path, 'Solution_*.csv'))

    # Inicializa una lista para guardar las matrices
    matrices = []

    # Lee cada archivo CSV y convierte su contenido a una matriz de NumPy
    for file in csv_files:
        matrix = np.loadtxt(file, delimiter=',')
        matrices.append(matrix)
    
    return matrices

def matriz_a_secuencia(matriz):
    # Convertir la matriz a una secuencia (lista) de sus elementos
    return matriz.flatten()

def distancia_edicion(seq1, seq2):
    # Calcular la distancia de Levenshtein entre dos secuencias
    m, n = len(seq1), len(seq2)
    dp = np.zeros((m+1, n+1), dtype=int)
    
    for i in range(m+1):
        dp[i][0] = i
    for j in range(n+1):
        dp[0][j] = j
    
    for i in range(1, m+1):
        for j in range(1, n+1):
            if seq1[i-1] == seq2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    
    return dp[m][n]

def porcentaje_similitud(distancia_edicion, max_distancia):
    return (1 - distancia_edicion / max_distancia) * 100

def promedio_distancia_edicion(matrices):
    # Generar todas las combinaciones de matrices
    combinaciones = list(combinations(matrices, 2))
    porcentajes = []

    for (mat1, mat2) in combinaciones:
        seq1 = matriz_a_secuencia(mat1)
        seq2 = matriz_a_secuencia(mat2)
        distancia = distancia_edicion(seq1, seq2)
        
        max_distancia = mat1.size  # La distancia máxima es el número total de elementos
        similitud = porcentaje_similitud(distancia, max_distancia)
        porcentajes.append(similitud)
    
    # Calcular el promedio de los porcentajes de similitud
    promedio_porcentaje = np.mean(porcentajes)
    return promedio_porcentaje

folder_path = 'results/Experiment 27-05/SolutionsCsv'
matrices = read_solutions(folder_path)

start = T.time()
print(f"Comienzo del calculo de autosimilitud entre {len(matrices)} mapas")

promedio = promedio_distancia_edicion(matrices)
print(f"La autosimilitud de los mapas es: {promedio}")

print(f"Se termino de ejecutar en {T.time() - start} segundos")