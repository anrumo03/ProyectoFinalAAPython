from AlgoritmosMult import *
import time
import csv

def read_matrices_from_csv(filename):
    sizes = [4, 8, 16, 32, 64, 128, 256, 512]
    matrices = {}
    with open(filename, 'r', encoding='utf-8-sig') as file:
        lines = file.readlines()
    
    current_line = 0
    for size in sizes:
        matrix_a = []
        matrix_b = []
        
        # Leer la primera matriz (A) de tamaño actual
        for i in range(size):
            row = lines[current_line].strip().split(',')
            # Manejar valores vacíos reemplazándolos por 0
            row = [int(value) if value.strip() else 0 for value in row]
            matrix_a.append(row)
            current_line += 1
        
        current_line += 1

        # Leer la segunda matriz (B) de tamaño actual
        for i in range(size):
            row = lines[current_line].strip().split(',')
            # Manejar valores vacíos reemplazándolos por 0
            row = [int(value) if value.strip() else 0 for value in row]
            matrix_b.append(row)
            current_line += 1

        current_line += 1
        
        # Guardar ambas matrices en el diccionario
        matrices[size] = {"A": matrix_a, "B": matrix_b}
    
    return matrices


def measure_execution_time(algorithms, matrices, output_file):
    results = {}

    with open(output_file, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        
        # Escribir encabezado del CSV
        header = ['Matrix Size'] + list(algorithms.keys())
        csv_writer.writerow(header)
        
        # Procesar cada tamaño de matriz
        for size, matrix_pair in matrices.items():
            print(f"Procesando matrices de tamaño {size}x{size}...")
            matrix_a = matrix_pair["A"]
            matrix_b = matrix_pair["B"]
            
            results[size] = {}
            row = [size]  # Fila inicial con el tamaño de la matriz

            # Probar cada algoritmo
            for name, algorithm in algorithms.items():
                start_time = time.time()  # Inicia el cronómetro
                result = algorithm(matrix_a, matrix_b)  # Ejecuta el algoritmo
                end_time = time.time()  # Detiene el cronómetro

                elapsed_time = end_time - start_time
                results[size][name] = elapsed_time  # Guarda el tiempo
                print(f"  {name}: {elapsed_time:.6f} segundos")
                row.append(elapsed_time)  # Agrega el tiempo a la fila
            
            # Escribir fila en el CSV
            csv_writer.writerow(row)

    print(f"\nResultados guardados en: {output_file}")
    return results


def main():

    matrices = read_matrices_from_csv("matrices.csv")

    algorimos = {
        "Naive Multiplication": naive_on_array,
        "Naive Loop Unrolling (Two)": naive_loop_unrolling_two,
        "Naive Loop Unrolling (Four)": naive_loop_unrolling_four,
        "Winograd Original": winograd_original,
        "Winograd Scaled": winograd_scaled,
        "Strassen Naive": strassen,
        "Strassen Winograd": strassen_winograd,
        "Sequential Block": sequential_block,
        "Paralel Block": parallel_block_multiplication,
        "Enhanced Paralel Block": enhanced_parallel_block_multiplication
    }

    results = measure_execution_time(algorimos, matrices, "tiempos.csv")

    """ # Muestra los resultados
    print("\nResultados de tiempos de ejecución:")
    for size, times in results.items():
        print(f"Tamaño {size}x{size}:")
        for algorithm, time_taken in times.items():
            print(f"  {algorithm}: {time_taken:.6f} segundos") """
    

if __name__=="__main__":
    main()