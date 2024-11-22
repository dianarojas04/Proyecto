#Integrantes
#Juan Carlos Cañón Cárdenas 
#Ibsen Anneth Sánchez Hernández
#Diana Carolina Rojas Orjuela

from matplotlib import pyplot as plt
import cv2
import numpy as np
import math

# Redefinir variables y volver a cargar la imagen
image_path = "158.jpg"
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Volver a cargar el canal verde
r,g,b = cv2.split(image_rgb)

# Procesar la imagen con Sobel y aplicar el umbral de Otsu
sobel_x = cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=3)
sobel_combined = cv2.magnitude(sobel_x, sobel_y)
sobel_combined = np.uint8(np.absolute(sobel_combined))
_, otsu_thresholded = cv2.threshold(sobel_combined, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Aplicar filtro de mediana
median_filtered = cv2.medianBlur(otsu_thresholded, 5)

# Dilatar la máscara y mejorarla con operación de cierre
kernel = np.ones((5, 5), np.uint8)
dilated_mask = cv2.dilate(median_filtered, kernel, iterations=1)
mascara_mejorada = cv2.morphologyEx(dilated_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

# Invertir la máscara mejorada
mascara_invertida_mejorada = cv2.bitwise_not(mascara_mejorada)

# Aplicar la máscara a la imagen original
imagen_filtrada_mejorada = cv2.bitwise_and(image_rgb, image_rgb, mask=mascara_invertida_mejorada)

# Realizar interpolación para rellenar las áreas borradas
imagen_rellenada_mejorada = cv2.inpaint(image_rgb, mascara_mejorada, inpaintRadius=3, flags=cv2.INPAINT_TELEA)


# Aplicar filtro gaussiano al canal rojo para suavizar la imagen y reducir el ruido
canal_rojo_suavizado = cv2.GaussianBlur(r, (5, 5), 0)

# Mostrar el canal rojo suavizado
plt.imshow(canal_rojo_suavizado, cmap='gray')
plt.title('Canal Rojo Suavizado')
plt.axis('off')
plt.show()

# Calcular y mostrar el histograma del canal rojo suavizado
plt.figure(figsize=(8, 5))
plt.hist(canal_rojo_suavizado.ravel(), bins=256, range=(0, 256), color='red', alpha=0.7)
plt.title('Histograma del Canal Rojo Suavizado')
plt.xlabel('Intensidad de los píxeles')
plt.ylabel('Frecuencia')
plt.show()

# Fragmento de código para aplicar umbralización iterativa descendente y detectar contornos
umbral_inicial = 255
incremento_descendente = 2
contornos_detectados = []

# Iterar el proceso de umbralización desde 255 hasta 0 en decrementos de 10
for umbral in range(umbral_inicial, 0, -incremento_descendente):
    # Aplicar umbral binario
    _, imagen_umbralizada = cv2.threshold(canal_rojo_suavizado, umbral, 255, cv2.THRESH_BINARY)
    
    # Encontrar contornos en la imagen umbralizada
    contornos, _ = cv2.findContours(imagen_umbralizada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Verificar si se encontraron contornos y detener el bucle
    if len(contornos) > 0:
        contornos_detectados = contornos
        break

# Seleccionar el contorno con el área máxima
if contornos_detectados:
    contorno_maximo = max(contornos_detectados, key=cv2.contourArea)

    # Calcular el área del contorno máximo
    asc_disc = cv2.contourArea(contorno_maximo)

    # Dibujar el contorno con el área máxima en la imagen original
    cv2.drawContours(image_rgb, [contorno_maximo], -1, (0, 255, 0), 2)  # Contorno en verde

    # Mostrar la imagen con el contorno del disco óptico
    plt.imshow(image_rgb)
    plt.title('Contorno del Disco Óptico Detectado')
    plt.axis('off')
    plt.show()

    # Imprimir el área calculada (opcional)
    print(f"Área contorno disco: {asc_disc:.2f} píxeles cuadrados")
else:
    print("No se encontraron contornos en la imagen.")


# Calcular el círculo mínimo que encierra el contorno máximo
(x, y), radio = cv2.minEnclosingCircle(contorno_maximo)
centro = (int(x), int(y))
radio = int(radio)
# Calcular el área del círculo del disco optico
area_disc = math.pi * (radio ** 2)

# Dibujar el círculo que encierra el contorno en la imagen original
cv2.circle(image_rgb, centro, radio, (255, 0, 0), thickness=2)  # Círculo en azul

# Mostrar la imagen con el contorno encerrado como un círculo
plt.imshow(image_rgb)
plt.title('Contorno del Disco Óptico Encerrado en un Círculo')
plt.axis('off')
plt.show()

# Imprimir el área calculada (opcional)
print(f"Área del círculo disco óptico: {area_disc:.2f} píxeles cuadrados")

# Mostrar la máscara mejorada y la imagen sin áreas detectadas
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(mascara_mejorada, cmap='gray')
axes[0].set_title('Máscara Mejorada (Con Cierre)')
axes[0].axis('off')

axes[1].imshow(imagen_rellenada_mejorada)
axes[1].set_title('Imagen Rellenada con Máscara Mejorada')
axes[1].axis('off')

plt.show()

# Convertir la imagen rellenada a escala de grises y mostrar solo el canal azul
imagen_gris = cv2.cvtColor(imagen_rellenada_mejorada, cv2.COLOR_RGB2GRAY)

# Mostrar la imagen en escala de grises
plt.imshow(imagen_gris, cmap='gray')
plt.title('Imagen en Escala de Grises')
plt.axis('off')
plt.show()


# Calcular y mostrar el histograma de la imagen en escala de grises
plt.figure(figsize=(8, 5))
plt.hist(imagen_gris.ravel(), bins=256, range=(0, 256), color='blue', alpha=0.7)
plt.title('Histograma de la Imagen en Escala de Grises')
plt.xlabel('Intensidad de los píxeles')
plt.ylabel('Frecuencia')
plt.show()


# Aplicar umbral descendente para segmentar la imagen desde 215
umbral_min = 215
umbral_max = 255

# Crear una máscara binaria con los umbrales especificados
_, mascara_umbral = cv2.threshold(imagen_gris, umbral_min, umbral_max, cv2.THRESH_BINARY)

# Encontrar los contornos en la máscara binaria
contornos, _ = cv2.findContours(mascara_umbral, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Dibujar los contornos en una copia de la imagen original
imagen_contornos = image_rgb.copy()
cv2.drawContours(imagen_contornos, contornos, -1, (255, 0, 0), 1)  # Contornos en azul

# Mostrar la imagen con los contornos
plt.imshow(imagen_contornos)
plt.title('Contornos Segmentados desde Umbral 215')
plt.axis('off')
plt.show()

# Rellenar los contornos segmentados
imagen_rellena = np.zeros_like(image_rgb)  # Crear una imagen en negro del mismo tamaño

# Rellenar los contornos con color blanco
cv2.drawContours(imagen_rellena, contornos, -1, (255, 255, 255), thickness=cv2.FILLED)

# Aplicar un desenfoque gaussiano para dar apariencia de esfera
imagen_rellena_suavizada = cv2.GaussianBlur(imagen_rellena, (15, 15), 0)

# Mostrar la imagen con los contornos rellenos y suavizados
plt.imshow(imagen_rellena_suavizada)
plt.title('Contornos Segmentados Rellenos con Apariencia Esférica')
plt.axis('off')
plt.show()

# Encontrar el contorno con el área mayor y eliminar el resto
max_area = 0
max_contorno = None

for contorno in contornos:
    area = cv2.contourArea(contorno)
    if area > max_area:
        max_area = area
        max_contorno = contorno

# Crear una imagen en negro del mismo tamaño para dibujar solo el contorno mayor
imagen_mayor_contorno = np.zeros_like(image_rgb)

# Dibujar solo el contorno con el área mayor
if max_contorno is not None:
    cv2.drawContours(imagen_mayor_contorno, [max_contorno], -1, (255, 255, 255), thickness=cv2.FILLED)

# Aplicar un desenfoque para dar apariencia esférica
imagen_mayor_contorno_suavizada = cv2.GaussianBlur(imagen_mayor_contorno, (15, 15), 0)

# Mostrar la imagen con el contorno mayor relleno y suavizado
plt.imshow(imagen_mayor_contorno_suavizada)
plt.title('Contorno con Área Mayor Relleno y Suavizado')
plt.axis('off')
plt.show()

# Aplicar erosión iterativa hasta separar las áreas
imagen_erosionada = imagen_mayor_contorno.copy()
kernel = np.ones((3, 3), np.uint8)

# Realizar la erosión iterativa y verificar visualmente la separación
for i in range(1, 50):  # Hasta un máximo de 50 iteraciones
    imagen_erosionada = cv2.erode(imagen_erosionada, kernel, iterations=1)
    # Contar los componentes conectados
    num_labels, _, _, _ = cv2.connectedComponentsWithStats(cv2.cvtColor(imagen_erosionada, cv2.COLOR_RGB2GRAY))
    if num_labels > 2:  # Si hay más de dos áreas separadas (fondo y contorno separado)
        break

# Encontrar los contornos en la imagen erosionada
contornos_finales, _ = cv2.findContours(cv2.cvtColor(imagen_erosionada, cv2.COLOR_RGB2GRAY), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if contornos_finales:
    # Seleccionar el contorno mayor
    contorno_mayor = max(contornos_finales, key=cv2.contourArea)
    
    # Calcular el área del contorno mayor
    asc_cup = cv2.contourArea(contorno_mayor)
    
    # Imprimir el área calculada (opcional)
    print(f"Área cup_contorno: {asc_cup:.2f} píxeles cuadrados")
else:
    print("No se encontraron contornos en la imagen erosionada.")

# Mostrar la imagen erosionada que separa las áreas
plt.imshow(imagen_erosionada)
plt.title('Imagen Erosionada con Áreas Separadas')
plt.axis('off')
plt.show()


# Encerrar el contorno mayor en un círculo y completarlo
imagen_circulo = imagen_erosionada.copy()


# Encontrar el contorno mayor nuevamente
contornos_finales, _ = cv2.findContours(cv2.cvtColor(imagen_erosionada, cv2.COLOR_RGB2GRAY), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if contornos_finales:
    # Seleccionar el contorno mayor
    contorno_mayor = max(contornos_finales, key=cv2.contourArea)

    # Calcular el área del contorno mayor
    area_cup = cv2.contourArea(contorno_mayor)

    # Obtener el círculo mínimo que encierra el contorno
    (x, y), radio = cv2.minEnclosingCircle(contorno_mayor)
    centro = (int(x), int(y))
    radio = int(radio)

    # Dibujar el círculo que encierra el contorno
    cv2.circle(imagen_circulo, centro, radio, (255, 255, 255), thickness=cv2.FILLED)

    # Imprimir el área calculada (opcional)
    print(f"Área copa_circulo: {area_cup:.2f} píxeles cuadrados")
else:
    print("No se encontraron contornos en la imagen.")


# Mostrar la imagen con el contorno encerrado y completado como un círculo
plt.imshow(imagen_circulo)
plt.title('Contorno Encerrado y Completado como Círculo_')
plt.axis('off')
plt.show()

# Trazar el contorno del círculo en la imagen original
imagen_original_con_contorno = image_rgb.copy()

# Dibujar el círculo en la imagen original
cv2.circle(imagen_original_con_contorno, centro, radio, (255, 0, 0), thickness=2)  # Círculo en azul

# Mostrar la imagen original con el contorno trazado
plt.imshow(imagen_original_con_contorno)
plt.title('Contorno Trazado en la Imagen Original')
plt.axis('off')
plt.show()

# Calcular el CDR usando asc_cup y asc_disc
if asc_disc > 0:
    cdr_asc = asc_cup / asc_disc
    print(f"CDR (asc): {cdr_asc:.2f}")

    # Verificar si el CDR indica sospecha de glaucoma
    if cdr_asc > 0.6:
        print("Sospecha de glaucoma (asc).")
    else:
        print("No se sospecha de glaucoma (asc).")
else:
    print("El área del disco óptico (asc_disc) es 0 o menor, no se puede calcular el CDR.")

# Calcular el CDR usando area_cup y area_disc
if area_disc > 0:
    cdr_area = area_cup / area_disc
    print(f"CDR (area): {cdr_area:.2f}")

    # Verificar si el CDR indica sospecha de glaucoma
    if cdr_area > 0.6:
        print("Sospecha de glaucoma (area).")
    else:
        print("No se sospecha de glaucoma (area).")
else:
    print("El área del disco óptico (area_disc) es 0 o menor, no se puede calcular el CDR.")
