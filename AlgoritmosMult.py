import random
import multiprocessing
import numpy as np

# 1. Naive on Array
def naive_on_array(A, B):
    n = len(A)
    result = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            for k in range(n):
                result[i][j] += A[i][k] * B[k][j]
    return result

# 2. Naive Loop Unrolling (Two)
def naive_loop_unrolling_two(A, B):
    n = len(A)
    result = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            k = 0
            while k < n - 1:
                result[i][j] += A[i][k] * B[k][j] + A[i][k+1] * B[k+1][j]
                k += 2
            if k < n:
                result[i][j] += A[i][k] * B[k][j]
    return result

# 3. Naive Loop Unrolling (Four)
def naive_loop_unrolling_four(A, B):
    n = len(A)
    result = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            k = 0
            while k < n - 3:
                result[i][j] += (A[i][k] * B[k][j] +
                                 A[i][k+1] * B[k+1][j] +
                                 A[i][k+2] * B[k+2][j] +
                                 A[i][k+3] * B[k+3][j])
                k += 4
            while k < n:
                result[i][j] += A[i][k] * B[k][j]
                k += 1
    return result

# 4. Winograd Original
def winograd_original(A, B):
    n = len(A)
    m1 = [sum(A[i][k] * A[i][k+1] for k in range(0, n-1, 2)) for i in range(n)]
    m2 = [sum(B[k][j] * B[k+1][j] for k in range(0, n-1, 2)) for j in range(n)]

    result = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            result[i][j] = -m1[i] - m2[j]
            for k in range(0, n-1, 2):
                result[i][j] += (A[i][k] + B[k+1][j]) * (A[i][k+1] + B[k][j])
            if n % 2:
                result[i][j] += A[i][n-1] * B[n-1][j]
    return result

# 5. Winograd Scaled
def winograd_scaled(A, B):
    n = len(A)

    # Paso 1: Calcular factores de escala
    scale_A = [max(abs(A[i][k]) for k in range(n)) or 1 for i in range(n)]  # Máximo absoluto de cada fila
    scale_B = [max(abs(B[k][j]) for k in range(n)) or 1 for j in range(n)]  # Máximo absoluto de cada columna

    # Escalar las matrices
    scaled_A = [[A[i][k] / scale_A[i] for k in range(n)] for i in range(n)]
    scaled_B = [[B[k][j] / scale_B[j] for j in range(n)] for k in range(n)]

    # Paso 2: Calcular sumas intermedias
    m1 = [sum(scaled_A[i][k] * scaled_A[i][k+1] for k in range(0, n-1, 2)) for i in range(n)]
    m2 = [sum(scaled_B[k][j] * scaled_B[k+1][j] for k in range(0, n-1, 2)) for j in range(n)]

    # Paso 3: Calcular el producto escalar
    result = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            result[i][j] = -m1[i] - m2[j]
            for k in range(0, n-1, 2):
                result[i][j] += (scaled_A[i][k] + scaled_B[k+1][j]) * (scaled_A[i][k+1] + scaled_B[k][j])
            if n % 2:  # Si la matriz tiene un tamaño impar, procesar el último elemento
                result[i][j] += scaled_A[i][n-1] * scaled_B[n-1][j]

    # Paso 4: Reescalar el resultado
    for i in range(n):
        for j in range(n):
            result[i][j] *= scale_A[i] * scale_B[j]

    return result

# 6. Strassen Naive
def strassen(A, B):
    n = len(A)
    if n == 1:
        return [[A[0][0] * B[0][0]]]
    
    mid = n // 2
    A11, A12, A21, A22 = [sub_matrix(A, mid, i, j) for i, j in [(0, 0), (0, mid), (mid, 0), (mid, mid)]]
    B11, B12, B21, B22 = [sub_matrix(B, mid, i, j) for i, j in [(0, 0), (0, mid), (mid, 0), (mid, mid)]]

    M1 = strassen(add(A11, A22), add(B11, B22))
    M2 = strassen(add(A21, A22), B11)
    M3 = strassen(A11, subtract(B12, B22))
    M4 = strassen(A22, subtract(B21, B11))
    M5 = strassen(add(A11, A12), B22)
    M6 = strassen(subtract(A21, A11), add(B11, B12))
    M7 = strassen(subtract(A12, A22), add(B21, B22))

    C11 = add(subtract(add(M1, M4), M5), M7)
    C12 = add(M3, M5)
    C21 = add(M2, M4)
    C22 = add(subtract(add(M1, M3), M2), M6)

    return combine(C11, C12, C21, C22)

# Funciones auxiliares para Strassen
def sub_matrix(M, mid, i, j):
    return [row[j:j+mid] for row in M[i:i+mid]]

def add(M1, M2):
    return [[M1[i][j] + M2[i][j] for j in range(len(M1))] for i in range(len(M1))]

def subtract(M1, M2):
    return [[M1[i][j] - M2[i][j] for j in range(len(M1))] for i in range(len(M1))]

def combine(C11, C12, C21, C22):
    n = len(C11)
    result = [[0] * (2 * n) for _ in range(2 * n)]
    for i in range(n):
        for j in range(n):
            result[i][j] = C11[i][j]
            result[i][j+n] = C12[i][j]
            result[i+n][j] = C21[i][j]
            result[i+n][j+n] = C22[i][j]
    return result

# 7. Strassen Winograd
def strassen_winograd(A, B):
    n = len(A)

    # Caso base: Si la matriz es 1x1, simplemente multiplicar
    if n == 1:
        return [[A[0][0] * B[0][0]]]

    # Dividir las matrices en submatrices
    mid = n // 2
    A11, A12, A21, A22 = [sub_matrix(A, mid, i, j) for i, j in [(0, 0), (0, mid), (mid, 0), (mid, mid)]]
    B11, B12, B21, B22 = [sub_matrix(B, mid, i, j) for i, j in [(0, 0), (0, mid), (mid, 0), (mid, mid)]]

    # Calcular productos intermedios con optimizaciones de Winograd
    M1 = strassen_winograd(add(A11, A22), add(B11, B22))  # (A11 + A22) * (B11 + B22)
    M2 = strassen_winograd(add(A21, A22), B11)            # (A21 + A22) * B11
    M3 = strassen_winograd(A11, subtract(B12, B22))       # A11 * (B12 - B22)
    M4 = strassen_winograd(A22, subtract(B21, B11))       # A22 * (B21 - B11)
    M5 = strassen_winograd(add(A11, A12), B22)            # (A11 + A12) * B22
    M6 = strassen_winograd(subtract(A21, A11), add(B11, B12))  # (A21 - A11) * (B11 + B12)
    M7 = strassen_winograd(subtract(A12, A22), add(B21, B22))  # (A12 - A22) * (B21 + B22)

    # Combinación de submatrices
    C11 = add(subtract(add(M1, M4), M5), M7)  # C11 = M1 + M4 - M5 + M7
    C12 = add(M3, M5)                         # C12 = M3 + M5
    C21 = add(M2, M4)                         # C21 = M2 + M4
    C22 = add(subtract(add(M1, M3), M2), M6)  # C22 = M1 + M3 - M2 + M6

    # Combinar las submatrices en una sola matriz
    return combine(C11, C12, C21, C22)

# 8. Sequential Block
def sequential_block(A, B):
    n = len(A)
    block_size = n
    result = [[0] * n for _ in range(n)]
    for i in range(0, n, block_size):
        for j in range(0, n, block_size):
            for k in range(0, n, block_size):
                for bi in range(i, min(i + block_size, n)):
                    for bj in range(j, min(j + block_size, n)):
                        for bk in range(k, min(k + block_size, n)):
                            result[bi][bj] += A[bi][bk] * B[bk][bj]
    return result

# 9. Parallel Block
def parallel_block_multiplication(A, B):
    n = len(A)
    block_size = n
    result = [[0] * n for _ in range(n)]

    # Crear tareas paralelas para bloques
    processes = []
    step = block_size
    for i in range(0, n, step):
        for j in range(0, n, step):
            # Determina los límites de cada bloque
            row_start = i
            row_end = min(i + block_size, len(A))
            col_start = j
            col_end = min(j + block_size, len(B[0]))

            # Crear un proceso para cada bloque
            p = multiprocessing.Process(target=compute_block, args=(result, A, B, row_start, row_end, col_start, col_end))
            processes.append(p)
            p.start()

    # Esperar a que todos los procesos terminen
    for process in processes:
        process.join()

    return result

# Función para calcular un bloque de la matriz resultante
def compute_block(result, A, B, i_start, i_end, j_start, j_end):
     for i in range(i_start, i_end):
        for j in range(j_start, j_end):
            result[i][j] = sum(A[i][k] * B[k][j] for k in range(len(B)))

# 10. Enhanced Parallel Block
def enhanced_parallel_block_multiplication(A, B):
    n = len(A)
    block_size = n

    # Crear matriz de resultado compartida usando multiprocessing.Array
    result = multiprocessing.Array('d', n * n)  # Memoria compartida (double precision)

    # Crear procesos para los bloques
    processes = []
    step = block_size
    for i in range(0, n, step):
        for j in range(0, n, step):
            process = multiprocessing.Process(
                target=compute_block2,
                args=(A, B, result, n, i, min(i + step, n), j, min(j + step, n), block_size)
            )
            processes.append(process)
            process.start()

    # Esperar a que todos los procesos terminen
    for process in processes:
        process.join()

    # Convertir la matriz compartida a una matriz regular
    return [[result[i * n + j] for j in range(n)] for i in range(n)]

# Definir la función compute_block fuera de enhanced_parallel_block_multiplication
def compute_block2(A, B, result, n, i_start, i_end, j_start, j_end, block_size):
    local_result = np.zeros((block_size, block_size))
    for i in range(i_start, i_end):
        for j in range(j_start, j_end):
            for k in range(n):
                local_result[i - i_start][j - j_start] += A[i][k] * B[k][j]
    
    # Escribir el bloque en la matriz compartida
    for i in range(i_start, i_end):
        for j in range(j_start, j_end):
            result[i * n + j] = local_result[i - i_start][j - j_start]



