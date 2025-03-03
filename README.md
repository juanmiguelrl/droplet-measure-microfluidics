<!-- filepath: /G:/droplet notebook/droplet-measure-microfluidics/README.md -->
# Droplet (circle) Image Measurement Tool  

<p align="center">
  <a href="#english"><img src="https://img.shields.io/badge/Language-English-blue" alt="English"></a>
  <a href="#español"><img src="https://img.shields.io/badge/Language-Español-red" alt="Español"></a>
</p>

## English
This program allows the detection and measurement of circle diameters in images, returning the processed images with the detected circles drawn on them and .csv files containing a list of data for all detected circles.

GUI Parameters:
- "Save predefined settings": Allows saving the current parameters in a "settings.txt" file to use them as default whenever the program is opened
- "Delete all predefined settings": Deletes the saved default settings
- "Select image input directory": Folder where the images to be processed are located
- "Select image output directory": Folder where the processed images will be saved
- "Calculate Circles": To indicate that detected circles should be calculated
- "Use default scale": To indicate to use the scale indicated at "Select default scale image" as default scale. Alternatively you can add an scale in the directory of each sample along with the rest of the images of the sample.
- "Select default scale image": Allows you to specify which image to use as a reference for scaling
- "Apply to the specified sample": Allows you to apply circle detection to only the sample with the specified number
- "Join data": Allows you to specify if you want to obtain a .csv file containing the combined data from all processed samples
- "Measure the real distance": To indicate that the real distance of the detected circles should be calculated (otherwise the measurement is calculated only in pixels)
- "Initiate detection": Runs the circle detection process  
  
Input directory structure: A directory with one or multiple directories with their names starting with "Sample " or "sample ", containing each one the sample images of each sample and alternatively a scale image for each sample if not wanted to use the same default scale for all the samples.  


## Español  
Este programa permite detectar y medir el diámetro de círculos en imágenes, devolviendo las imágenes con los círculos detectados dibujados sobre ellas y archivos .csv con un listado de los datos de los círculos detectados.

Parámetros a ajustar en GUI:
- "Save predefined settings" : Permite guardar en un archivo "settings.txt" los parámetros actualies para utilizarlos de forma predeterminada siempre que se abra el programa
- "Delete all predefined settings" : Borra los ajustes preterminados guardados
- "Select image input directory" : Carpeta donde se encuentran las imágenes a procesar.
- "Select image output directory" : Carpeta donde se guardarán las imágenes procesadas.
- "Calculate Circles" : Para indicar que se detecten y calculen los circulos detectados
- "Use default scale" : Para indicar que se utilice la escala indicada en "Select default scale image" como escala predeterminada. Alternativamente, puedes agregar una escala en el directorio de cada muestra junto con el resto de imágenes de la muestra. 
- "Select default scale image" : Permite indicar qué imagen utilizar como referencia para tomar la escala
- "Apply to the specified sample" : Permite aplicar la detección de círculos a sólo la muestra con el número indicado
- "Join data" : Permite indicar si se quiere obtener un archivo .csv que contenga los datos de todas las muestras procesadas unidos
- "Measure the real distance" : Para indicar que se calcule la distancia real de los círculos detectados (de lo contrario la medida se calcula sólo en píxeles)
- "Initiate detection" : Ejecuta la detección de los círculos  
  
Estructura del directorio de entrada (input): Un directorio con uno o varios directorios con sus nombres comenzando con "Sample" o "sample", que contienen cada uno las imágenes de cada muestra y, alternativamente, una imagen de la escala de cada muestra si no se desea usar la misma escala predeterminada para todas las muestras.