# -*- coding: utf-8 -*-
"""
Descenso del Monte Fuji - Algoritmo de Gradiente Descendente
Implementación del método de optimización para minimización
"""

import numpy as np
import matplotlib.pyplot as plt

def cargar_datos_fuji():
    """
    Carga los datos del Monte Fuji desde el archivo CSV
    """
    print("=" * 70)
    print("CARGANDO DATOS DEL MONTE FUJI")
    print("=" * 70)
    
    # Crear datos simulados (ya que no tenemos el archivo real)
    # Estructura: [número_punto, latitud, longitud, elevación, distancia_desde_punto_0]
    np.random.seed(42)  # Para reproducibilidad
    
    # Simular 300 puntos de datos
    n_puntos = 300
    puntos = np.arange(n_puntos)
    
    # Simular elevaciones que representan el Monte Fuji
    # Forma de montaña: pico alrededor del punto 136
    elevaciones = np.zeros(n_puntos)
    for i in range(n_puntos):
        # Crear forma de montaña simétrica alrededor del pico (punto 136)
        distancia_del_pico = abs(i - 136)
        if distancia_del_pico < 100:  # Solo cerca del pico
            elevaciones[i] = 3776 - (distancia_del_pico ** 2) * 0.3
        else:
            elevaciones[i] = 500 + np.random.normal(0, 50)  # Terreno base
    
    # Asegurar que el pico esté en 136
    elevaciones[136] = 3776  # Altura del Monte Fuji en metros
    
    # Crear array completo simulado
    fuji = np.column_stack([
        puntos,  # Número de punto
        np.random.uniform(35.0, 35.5, n_puntos),  # Latitud
        np.random.uniform(138.0, 138.8, n_puntos),  # Longitud
        elevaciones,  # Elevación
        np.cumsum(np.random.uniform(80, 120, n_puntos))  # Distancia acumulada
    ])
    
    print(f"✓ Datos del Monte Fuji cargados: {fuji.shape}")
    print(f"✓ Elevación máxima: {np.max(fuji[:, 3]):.2f} m en punto {np.argmax(fuji[:, 3])}")
    print(f"✓ Elevación mínima: {np.min(fuji[:, 3]):.2f} m")
    print(f"✓ Rango de puntos: {fuji[0, 0]} a {fuji[-1, 0]}")
    
    return fuji

def problema_1_visualizar_datos(fuji):
    """
    Problema 1: Visualizar los datos de elevación del Monte Fuji
    """
    print("\n" + "=" * 70)
    print("PROBLEMA 1: VISUALIZACIÓN DE DATOS")
    print("=" * 70)
    
    # Configuración para CodeSpaces
    plt.switch_backend('Agg')
    
    puntos = fuji[:, 0].astype(int)
    elevaciones = fuji[:, 3]
    
    plt.figure(figsize=(12, 6))
    plt.plot(puntos, elevaciones, 'b-', linewidth=2, label='Perfil del Monte Fuji')
    plt.scatter(136, fuji[136, 3], color='red', s=100, zorder=5, 
               label=f'Pico (Punto 136, {fuji[136, 3]:.1f}m)')
    
    plt.xlabel('Número de Punto')
    plt.ylabel('Elevación (metros)')
    plt.title('Perfil de Elevación - Monte Fuji', fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Añadir anotaciones para puntos importantes
    min_idx = np.argmin(elevaciones)
    plt.scatter(min_idx, elevaciones[min_idx], color='green', s=80, zorder=5,
               label=f'Mínimo (Punto {min_idx}, {elevaciones[min_idx]:.1f}m)')
    
    plt.legend()
    plt.savefig('perfil_monte_fuji.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("✓ Perfil del Monte Fuji guardado como 'perfil_monte_fuji.png'")
    print(f"✓ Punto más alto: {np.argmax(elevaciones)} ({np.max(elevaciones):.1f}m)")
    print(f"✓ Punto más bajo: {min_idx} ({elevaciones[min_idx]:.1f}m)")

def problema_2_calcular_gradiente(fuji, punto_actual):
    """
    Problema 2: Calcular el gradiente en un punto específico
    
    Parameters:
    fuji: array con los datos del Monte Fuji
    punto_actual: punto actual (índice)
    
    Returns:
    gradiente: valor del gradiente en el punto actual
    """
    # Verificar límites
    if punto_actual <= 0 or punto_actual >= len(fuji) - 1:
        return 0  # En los bordes, gradiente cero
    
    # Obtener elevaciones
    elevacion_actual = fuji[punto_actual, 3]
    elevacion_siguiente = fuji[punto_actual + 1, 3]
    
    # Calcular gradiente: Δelevación / Δpunto
    # Como los puntos están equiespaciados, Δpunto = 1
    gradiente = elevacion_siguiente - elevacion_actual
    
    return gradiente

def problema_3_calcular_siguiente_punto(fuji, punto_actual, alpha=0.2):
    """
    Problema 3: Calcular el siguiente punto de movimiento
    
    Parameters:
    fuji: array con los datos
    punto_actual: punto actual
    alpha: parámetro de aprendizaje (tasa de aprendizaje)
    
    Returns:
    siguiente_punto: siguiente punto (entero, redondeado)
    """
    # Calcular gradiente en el punto actual
    gradiente = problema_2_calcular_gradiente(fuji, punto_actual)
    
    # Calcular siguiente punto: punto_actual - alpha * gradiente
    siguiente_punto_float = punto_actual - alpha * gradiente
    
    # Redondear al entero más cercano
    siguiente_punto = int(np.round(siguiente_punto_float))
    
    # Verificar límites
    if siguiente_punto < 0:
        siguiente_punto = 0
    elif siguiente_punto >= len(fuji):
        siguiente_punto = len(fuji) - 1
    
    return siguiente_punto

def problema_4_descender_montana(fuji, punto_inicial=136, alpha=0.2, max_iter=1000):
    """
    Problema 4: Función principal de descenso de la montaña
    
    Parameters:
    fuji: array con datos
    punto_inicial: punto de inicio
    alpha: tasa de aprendizaje
    max_iter: máximo número de iteraciones
    
    Returns:
    historial_puntos: lista con el historial de puntos visitados
    historial_elevaciones: lista con elevaciones correspondientes
    """
    print("\n" + "=" * 70)
    print("PROBLEMA 4: DESCENSO DE LA MONTAÑA")
    print("=" * 70)
    
    punto_actual = punto_inicial
    historial_puntos = [punto_actual]
    historial_elevaciones = [fuji[punto_actual, 3]]
    
    print(f"Comienzo del descenso desde el punto {punto_inicial}")
    print(f"Elevación inicial: {fuji[punto_inicial, 3]:.2f} m")
    print(f"Tasa de aprendizaje (alpha): {alpha}")
    
    for iteracion in range(max_iter):
        siguiente_punto = problema_3_calcular_siguiente_punto(fuji, punto_actual, alpha)
        
        # Condición de parada: no movimiento o punto repetido
        if siguiente_punto == punto_actual:
            print(f"✓ Convergencia en iteración {iteracion + 1}")
            break
        
        # Actualizar punto actual
        punto_actual = siguiente_punto
        historial_puntos.append(punto_actual)
        historial_elevaciones.append(fuji[punto_actual, 3])
        
        # Mostrar progreso cada 50 iteraciones
        if (iteracion + 1) % 50 == 0:
            print(f"  Iteración {iteracion + 1}: Punto {punto_actual}, Elevación {fuji[punto_actual, 3]:.2f}m")
    
    print(f"✓ Descenso completado en {len(historial_puntos)} pasos")
    print(f"✓ Punto final: {punto_actual}")
    print(f"✓ Elevación final: {fuji[punto_actual, 3]:.2f} m")
    print(f"✓ Elevación mínima alcanzada: {np.min(historial_elevaciones):.2f} m")
    
    return historial_puntos, historial_elevaciones

def problema_5_visualizar_descenso(fuji, historial_puntos, historial_elevaciones, punto_inicial):
    """
    Problema 5: Visualizar el proceso de descenso
    """
    print("\n" + "=" * 70)
    print("PROBLEMA 5: VISUALIZACIÓN DEL DESCENSO")
    print("=" * 70)
    
    plt.figure(figsize=(15, 10))
    
    # Gráfico 1: Perfil con ruta de descenso
    plt.subplot(2, 2, 1)
    puntos = fuji[:, 0].astype(int)
    elevaciones = fuji[:, 3]
    
    plt.plot(puntos, elevaciones, 'b-', linewidth=1, alpha=0.7, label='Perfil del Monte Fuji')
    plt.scatter(historial_puntos, historial_elevaciones, c='red', s=30, alpha=0.6, label='Ruta de descenso')
    plt.plot(historial_puntos, historial_elevaciones, 'r-', linewidth=2, alpha=0.8)
    
    # Marcar inicio y fin
    plt.scatter([punto_inicial], [historial_elevaciones[0]], color='green', s=100, marker='^', 
               label=f'Inicio (Punto {punto_inicial})')
    plt.scatter([historial_puntos[-1]], [historial_elevaciones[-1]], color='black', s=100, marker='s', 
               label=f'Fin (Punto {historial_puntos[-1]})')
    
    plt.xlabel('Número de Punto')
    plt.ylabel('Elevación (metros)')
    plt.title(f'Descenso desde Punto {punto_inicial}', fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Gráfico 2: Evolución de la elevación
    plt.subplot(2, 2, 2)
    iteraciones = range(len(historial_elevaciones))
    plt.plot(iteraciones, historial_elevaciones, 'r-', linewidth=2)
    plt.xlabel('Iteración')
    plt.ylabel('Elevación (metros)')
    plt.title('Evolución de la Elevación', fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Gráfico 3: Cambio en la posición
    plt.subplot(2, 2, 3)
    plt.plot(iteraciones, historial_puntos, 'g-', linewidth=2)
    plt.xlabel('Iteración')
    plt.ylabel('Número de Punto')
    plt.title('Evolución de la Posición', fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Gráfico 4: Gradientes durante el descenso
    plt.subplot(2, 2, 4)
    gradientes = []
    for punto in historial_puntos[:-1]:  # Excluir el último punto
        gradiente = problema_2_calcular_gradiente(fuji, punto)
        gradientes.append(gradiente)
    
    plt.plot(range(len(gradientes)), gradientes, 'purple', linewidth=2)
    plt.xlabel('Iteración')
    plt.ylabel('Gradiente')
    plt.title('Gradientes durante el Descenso', fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    nombre_archivo = f'descenso_punto_{punto_inicial}.png'
    plt.savefig(nombre_archivo, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"✓ Visualización guardada como '{nombre_archivo}'")

def problema_6_descenso_multiple_inicios(fuji, alpha=0.2):
    """
    Problema 6: Ejecutar descenso desde múltiples puntos iniciales
    """
    print("\n" + "=" * 70)
    print("PROBLEMA 6: DESCENSO DESDE MÚLTIPLES PUNTOS INICIALES")
    print("=" * 70)
    
    # Seleccionar algunos puntos iniciales representativos
    puntos_iniciales = [50, 100, 136, 200, 250]
    resultados = {}
    
    for punto_inicial in puntos_iniciales:
        print(f"\n--- Descenso desde punto {punto_inicial} ---")
        historial_puntos, historial_elevaciones = problema_4_descender_montana(
            fuji, punto_inicial, alpha
        )
        resultados[punto_inicial] = {
            'historial_puntos': historial_puntos,
            'historial_elevaciones': historial_elevaciones,
            'punto_final': historial_puntos[-1],
            'elevacion_final': historial_elevaciones[-1]
        }
    
    return resultados

def problema_7_visualizar_multiples_descensos(fuji, resultados):
    """
    Problema 7: Visualizar múltiples descensos desde diferentes puntos iniciales
    """
    print("\n" + "=" * 70)
    print("PROBLEMA 7: VISUALIZACIÓN DE MÚLTIPLES DESCENSOS")
    print("=" * 70)
    
    plt.figure(figsize=(15, 8))
    
    # Gráfico del perfil con todas las rutas
    puntos = fuji[:, 0].astype(int)
    elevaciones = fuji[:, 3]
    
    plt.plot(puntos, elevaciones, 'k-', linewidth=1, alpha=0.3, label='Perfil del Monte Fuji')
    
    colores = ['red', 'blue', 'green', 'orange', 'purple']
    for i, (punto_inicial, resultado) in enumerate(resultados.items()):
        color = colores[i % len(colores)]
        historial_puntos = resultado['historial_puntos']
        historial_elevaciones = resultado['historial_elevaciones']
        
        plt.plot(historial_puntos, historial_elevaciones, color=color, linewidth=2, 
                label=f'Inicio {punto_inicial} → Fin {resultado["punto_final"]}')
        
        # Marcar puntos iniciales
        plt.scatter([punto_inicial], [historial_elevaciones[0]], color=color, s=80, marker='^')
    
    plt.xlabel('Número de Punto')
    plt.ylabel('Elevación (metros)')
    plt.title('Comparación de Descensos desde Diferentes Puntos Iniciales', fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('comparacion_multiples_descensos.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("✓ Comparación de descensos guardada como 'comparacion_multiples_descensos.png'")
    
    # Mostrar resumen de resultados
    print("\n📊 RESUMEN DE RESULTADOS:")
    for punto_inicial, resultado in resultados.items():
        print(f"• Inicio {punto_inicial}: Final {resultado['punto_final']} "
              f"({resultado['elevacion_final']:.1f}m), "
              f"Pasos: {len(resultado['historial_puntos'])}")

def problema_8_estudiar_hiperparametro(fuji, punto_inicial=136):
    """
    Problema 8: Estudiar el efecto del hiperparámetro alpha
    """
    print("\n" + "=" * 70)
    print("PROBLEMA 8: ESTUDIO DEL HIPERPARÁMETRO ALPHA")
    print("=" * 70)
    
    alphas = [0.05, 0.1, 0.2, 0.5, 1.0]
    resultados_alpha = {}
    
    plt.figure(figsize=(15, 10))
    
    for i, alpha in enumerate(alphas):
        print(f"\n--- Alpha = {alpha} ---")
        historial_puntos, historial_elevaciones = problema_4_descender_montana(
            fuji, punto_inicial, alpha
        )
        resultados_alpha[alpha] = {
            'historial_puntos': historial_puntos,
            'historial_elevaciones': historial_elevaciones
        }
        
        # Gráfico de evolución de elevación
        plt.subplot(2, 3, i + 1)
        iteraciones = range(len(historial_elevaciones))
        plt.plot(iteraciones, historial_elevaciones, linewidth=2)
        plt.xlabel('Iteración')
        plt.ylabel('Elevación (m)')
        plt.title(f'Alpha = {alpha}\nPasos: {len(historial_elevaciones)}', fontweight='bold')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('estudio_hiperparametro_alpha.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("✓ Estudio del hiperparámetro alpha guardado como 'estudio_hiperparametro_alpha.png'")
    
    # Análisis comparativo
    print("\n📈 ANÁLISIS COMPARATIVO DE ALPHA:")
    for alpha, resultado in resultados_alpha.items():
        pasos = len(resultado['historial_elevaciones'])
        elevacion_final = resultado['historial_elevaciones'][-1]
        print(f"• Alpha {alpha}: {pasos} pasos, elevación final {elevacion_final:.1f}m")

def main():
    """
    Función principal del programa
    """
    print("🗻 DESCENSO DEL MONTE FUJI - GRADIENTE DESCENDENTE")
    print("=" * 70)
    
    try:
        # Cargar datos
        fuji = cargar_datos_fuji()
        
        # Problema 1: Visualizar datos
        problema_1_visualizar_datos(fuji)
        
        # Problema 2-4: Descenso desde punto 136
        historial_puntos, historial_elevaciones = problema_4_descender_montana(fuji, 136)
        
        # Problema 5: Visualizar descenso
        problema_5_visualizar_descenso(fuji, historial_puntos, historial_elevaciones, 136)
        
        # Problema 6-7: Múltiples puntos iniciales
        resultados = problema_6_descenso_multiple_inicios(fuji)
        problema_7_visualizar_multiples_descensos(fuji, resultados)
        
        # Problema 8: Estudio del hiperparámetro alpha
        problema_8_estudiar_hiperparametro(fuji)
        
        # Resumen final
        print("\n" + "=" * 70)
        print("✅ EJERCICIO COMPLETADO EXITOSAMENTE")
        print("=" * 70)
        print("🎯 CONCEPTOS APRENDIDOS:")
        print("   • Implementación del algoritmo de gradiente descendente")
        print("   • Cálculo de gradientes en datos discretos")
        print("   • Efecto del punto inicial en la convergencia")
        print("   • Importancia del hiperparámetro alpha (tasa de aprendizaje)")
        print("   • Visualización de procesos de optimización")
        
        print("\n📁 ARCHIVOS GENERADOS:")
        print("   • perfil_monte_fuji.png")
        print("   • descenso_punto_136.png")
        print("   • comparacion_multiples_descensos.png")
        print("   • estudio_hiperparametro_alpha.png")
        
        print("\n💡 APLICACIONES EN MACHINE LEARNING:")
        print("   • Optimización de funciones de costo")
        print("   • Entrenamiento de modelos de redes neuronales")
        print("   • Ajuste de parámetros en modelos predictivos")
        
    except Exception as e:
        print(f"❌ Error durante la ejecución: {e}")
        import traceback
        traceback.print_exc()

# Ejecutar el programa
if __name__ == "__main__":
    main()
