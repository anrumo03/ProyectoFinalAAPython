import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FixedLocator

# Archivos CSV
file_seconds = 'tiempos.csv'  # El archivo en segundos
file_milliseconds = r'C:\universidad\Uprograms\Java\8vo\analisis\ProyectoFinal\C#\ProyectoFinalAACSharp\PruebaGrafica\CSV\tiempos.csv'  # Reemplaza con tu segundo archivo

# Cargar datos
data_seconds = pd.read_csv(file_seconds)
data_milliseconds = pd.read_csv(file_milliseconds)

# Crear el primer gráfico (segundos)
plt.figure(figsize=(10, 6))
for column in data_seconds.columns[1:]:
    plt.plot(data_seconds["Matrix Size"], data_seconds[column], marker='o', label=column)

# Personalizar los valores del eje X
x_ticks = [4, 16, 32, 64, 128, 256, 512]  # Cambia estos valores según tu preferencia
plt.xticks(x_ticks, rotation=45)  # Rotar las etiquetas
plt.gca().xaxis.set_major_locator(FixedLocator(x_ticks))

plt.title("Tiempos de ejecución en Python")
plt.xlabel("Tamaño de la matriz")
plt.ylabel("Tiempo (segundos)")
plt.gca().yaxis.set_major_locator(MultipleLocator(20))  # Intervalos de 20 segundos
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1), title="Métodos")
plt.tight_layout()
plt.show()

# Crear el segundo gráfico (milisegundos)
plt.figure(figsize=(10, 6))
for column in data_milliseconds.columns[1:]:
    plt.plot(data_milliseconds["Matrix Size"], data_milliseconds[column], marker='o', label=column)

# Personalizar los valores del eje X
x_ticks = [4, 16, 32, 64, 128, 256, 512]  # Cambia estos valores según tu preferencia
plt.xticks(x_ticks, rotation=45)  # Rotar las etiquetas
plt.gca().xaxis.set_major_locator(FixedLocator(x_ticks))

plt.title("Tiempos de ejecución en C#")
plt.xlabel("Tamaño de la matriz")
plt.ylabel("Tiempo (milisegundos)")
plt.gca().yaxis.set_major_locator(MultipleLocator(1000))  # Intervalos de 1000 ms
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1), title="Métodos")
plt.tight_layout()
plt.show()

# Crear el tercer gráfico comparativo
# Convertir los datos en segundos a milisegundos
data_seconds_ms = data_seconds.copy()
for column in data_seconds.columns[1:]:
    data_seconds_ms[column] = data_seconds[column] * 1000  # Convertir segundos a milisegundos

# Calcular los promedios de cada columna
averages_seconds_ms = data_seconds_ms.mean(numeric_only=True)
averages_milliseconds = data_milliseconds.mean(numeric_only=True)

# Crear el gráfico comparativo
plt.figure(figsize=(12, 6))
x_labels = averages_seconds_ms.index  # Nombres de los algoritmos
x = range(len(x_labels))  # Posiciones en el eje X

# Graficar barras
plt.bar(x, averages_seconds_ms, width=0.4, label='Promedio (Python - ms)', align='center')
plt.bar([i + 0.4 for i in x], averages_milliseconds, width=0.4, label='Promedio (C# - ms)', align='center')

# Personalizar gráfico
plt.xticks([i + 0.2 for i in x], x_labels, rotation=45, ha='right')
plt.ylabel('Tiempo promedio (ms)')
plt.title('Comparación de tiempos promedio por algoritmo')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.6)

# Mostrar la gráfica
plt.tight_layout()
plt.show()
