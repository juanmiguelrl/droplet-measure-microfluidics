#prepare input and output directories
import os
import re
import time
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from skimage import data, color, filters,measure,draw
import matplotlib.image as mpimg
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.patches import Circle
import cv2
from tkinter import *

# Define the maximum number of processes to use
# Define el número máximo de procesos a utilizar
MAX_PROCESSES = 16

#default pixel size
DEFAULT_PIXED_SIZE = None

SIZE_REDUCTION_RESOLUTION = 4

ONLY_ONE = False

SAMPLE_NUM = 0

TIME_CONSUMED = 0


def processImages(inputDir,outputDir,measure_distance,get_measure=True,calcCircles=True,useDefaultScale=False,scale_path=None,join_data=False,onlyOne=False,sampleNum=None):
    global ONLY_ONE
    global SAMPLE_NUM
    ONLY_ONE = onlyOne
    SAMPLE_NUM = sampleNum

    #create output directory
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)
    if useDefaultScale:
        if not os.path.exists(scale_path):
            print("Scale path does not exist")
            return
        else:
            #gets the directory where the scale is located
            scaleDir = os.path.dirname(scale_path)
            relativeDir = os.path.relpath(inputDir, scaleDir)
            newOutputDir = os.path.join(outputDir, relativeDir)
            detectMeasure(scale_path, newOutputDir, outputDir,setDefaultScale=True)
    #create results csv (for the moment it is empty)
    with open(os.path.join(outputDir, 'globalResults.csv'), 'w') as f:
        pass
    navigateDir(inputDir, outputDir,get_measure,calcCircles,measure_distance,join_data)


#navigate through the input directory
def navigateDir(inputDir,outputDir,get_measure,calcCircles,measure_distance,join_data):
    # navigates through the input directory until if gets to a directory with the name "sample X"
    start_time = time.time()
    for root, dirs, files in os.walk(inputDir):
        for dir in dirs:
            if (re.match(r'sample \d+', os.path.basename(dir)) or re.match(r'Sample \d+', os.path.basename(dir))):
                print("Processing directory: " + dir)
                processDir(os.path.join(root, dir),inputDir, outputDir,calcCircles,measure_distance,join_data)
    end_time = time.time()
    global TIME_CONSUMED
    TIME_CONSUMED = end_time - start_time
    print(f"Time elapsed: {TIME_CONSUMED} seconds")

def detect_measure_contours(img,xdim=-250,ydim=-500,contour_level=0.8):
    #to gray scale
    gray_image = color.rgb2gray(img)

    # Otsu's thresholding
    thresh = filters.threshold_otsu(gray_image)
    binary_inv = gray_image > thresh

    # Blur img
    blur = filters.gaussian(binary_inv, sigma=1)

    # Otsu's thresholding
    thresh = filters.threshold_otsu(blur)
    binary = blur > thresh

    # Find contours
    contours = measure.find_contours(binary, level=contour_level)

    return contours,img

def draw_contours_in_empty(image, contours):
    # plot contours
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 4))
    ax.imshow(image, cmap='gray')
    for contour in contours:
        ax.plot(contour[:, 1], contour[:, 0], linewidth=2)
    ax.axis('image')
    ax.set_axis_off()
    ax.set_title('Contours')

    fig.canvas.draw()

    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    plt.close(fig)

    return data

def draw_main_rectangle_in_image_and_get_main_contour(image, contours):
    mask = np.ones(image.shape, dtype="uint8")
    definitive_contour = None
    # draws a rectangle in the image for each contour
    for contour in contours:
        #minr, minc, maxr, maxc = measure.regionprops(contour.astype(int))[0].bbox
        minr = np.min(contour[:, 0])  # minimo valor de y
        minc = np.min(contour[:, 1])  # minimo valor de x
        maxr = np.max(contour[:, 0])  # maximo valor de y
        maxc = np.max(contour[:, 1])  # maximo valor de x
        print( "minr: ", minr, " minc: ", minc, " maxr: ", maxr, " maxc: ", maxc)
        if (maxr - minr) * (maxc - minc) > 1000:
            if definitive_contour is None or (maxr - minr) * (maxc - minc) > (np.max(definitive_contour[:, 0]) - np.min(definitive_contour[:, 0])) * (np.max(definitive_contour[:, 1]) - np.min(definitive_contour[:, 1])):
                definitive_contour = contour

    if definitive_contour is not None:
        minr = np.min(definitive_contour[:, 0])  # minimum value for y
        minc = np.min(definitive_contour[:, 1])  # minimum value for x
        maxr = np.max(definitive_contour[:, 0])  # minimum value for y
        maxc = np.max(definitive_contour[:, 1])  # minimum value for x
        rr, cc = draw.rectangle_perimeter(start=(minr, minc), extent=(maxr-minr, maxc-minc), shape=image.shape)
        mask[rr, cc] = 0
        for r, c in zip(rr, cc):
            image[r, c] = [0, 255, 0]

        #res_final = imgNew * mask[:, :]
    return definitive_contour, image

def calculate_pixel_size(definitive_contour,distance):
    #minr = np.min(definitive_contour[:, 0])  # minimum value for y
    minc = np.min(definitive_contour[:, 1])  # minimum value for x
    #maxr = np.max(definitive_contour[:, 0])  # maximum value for y
    maxc = np.max(definitive_contour[:, 1])  # maximum value for x

    #calculate distancia de un pixel en metros sabiendo que la distancia en horizontal del rectangulo es la siguiente
    distance = 1000
    distance_pixels = maxc - minc
    print('each pixel is ', distance/distance_pixels, ' micrometers')
    return distance/distance_pixels

def detectMeasure(imagePath, newOutputDir, outputDir,setDefaultScale=False,defaultPixelSize=None):
    pixel_size = defaultPixelSize
    if (defaultPixelSize == None):
        print("imagePath is ", imagePath)
        image = mpimg.imread(imagePath)
        # Set to 0 all pixels that are not red
        # Pone a 0 todos los píxeles que no sean rojos
        red_pixels = image.copy()
        red_pixels[(red_pixels[:, :, 0] < 200) | (red_pixels[:, :, 1] >= 100) | (red_pixels[:, :, 2] >= 100)] = 0

        contours,reduced_img = detect_measure_contours(red_pixels)

        #image_with_distance = draw_contours_in_empty(reduced_img, contours)
        # gets the rectangle and the main contour
        #obtiene el rectangulo y el contorno principal
        main_contour, image_with_main_rectangle = draw_main_rectangle_in_image_and_get_main_contour(reduced_img, contours)
        print("------------------------------------")
        print("Adding scale: " + imagePath)
        image_base_name, image_extension = os.path.splitext(os.path.basename(imagePath))
        if image_extension.lower() == '.tif':
            image_extension = '.tiff'
        new_image_name = image_base_name + "_with_distance" + image_extension
        plt.imsave(os.path.join(newOutputDir, new_image_name), image_with_main_rectangle)
    
        pixel_size = calculate_pixel_size(main_contour, measure_distance)

    if setDefaultScale:
        global DEFAULT_PIXED_SIZE
        DEFAULT_PIXED_SIZE = pixel_size
        print("Default pixel size: ", DEFAULT_PIXED_SIZE)
        return

    # read the existing CSV file
    # Lee el archivo CSV existente
    df = pd.read_csv(os.path.join(newOutputDir, 'results.csv'))

    # adds a new column to the DataFrame with the real diameter
    df['realDiameter'] = df['diameter_pixel'] * pixel_size

    # Saves the DataFrame back to the CSV file
    # Guarda el DataFrame de nuevo en el archivo CSV
    df.to_csv(os.path.join(newOutputDir, 'results.csv'), index=False)
    if imagePath:
        print("Scale added: " + imagePath)
    else:
        print("Default scale added")

def joinGlobal(newOutputDir, outputDir):
    dfNew = pd.read_csv(os.path.join(newOutputDir, 'results.csv'))
    # get newOutputDir name
    sampleName = os.path.basename(newOutputDir)
    dfNew = dfNew.assign(sample=sampleName)
    global_results_path = os.path.join(outputDir, 'globalResults.csv')
    if os.path.exists(global_results_path) and os.path.getsize(global_results_path) > 0:
        dfGlobal = pd.read_csv(global_results_path)
        dfGlobal = pd.concat([dfGlobal, dfNew], axis=0)
    else:
        dfGlobal = dfNew

    # Saves the DataFrame back to the CSV file
    # Guarda el DataFrame de nuevo en el archivo CSV
    dfGlobal.to_csv(os.path.join(outputDir, 'globalResults.csv'), index=False)

def processDir(inputDir,originalInputDir, outputDir,calcCircles,measure_distance,join_data):
    if ((not ONLY_ONE) or (ONLY_ONE and SAMPLE_NUM == int(os.path.basename(inputDir).split()[1]))):

        # Creates a CSV file of results reflecting the structure of the input directory in the output directory
        # For this it takes the relative path of the input directory and creates the structure of the output directory

        #crea un csv de resultados reflejando la estructura del directorio de entrada en el directorio de salida
        #para ello toma la ruta relativa del directorio de entrada y crea la estructura del directorio de salida
        relativeDir = os.path.relpath(inputDir, originalInputDir)
        newOutputDir = os.path.join(outputDir, relativeDir)
        print(newOutputDir)
        if not os.path.exists(newOutputDir):
            os.makedirs(newOutputDir)

        if calcCircles:
            if ((not ONLY_ONE) or (ONLY_ONE and SAMPLE_NUM == int(os.path.basename(inputDir).split()[1]))):
                with open(os.path.join(newOutputDir, 'results.csv'), 'w') as f:
                    f.write("x,y,diameter_pixel,imagePath\n")
                        #pass
                    sampleName = os.path.basename(newOutputDir)
                    sampleNum = int(sampleName.split()[1])
                    # navigate through the files in the input directory
                    # navega a través de los archivos en el directorio de entrada
                with ProcessPoolExecutor(max_workers=MAX_PROCESSES) as executor:
                    for root, dirs, files in os.walk(inputDir):
                        # creates a list to hold the futures
                        # crea una lista para contener los futures
                        futures = []
                        scaleFile = None
                        for file in files:
                            # checks if the file is an image
                            #comprueba si el archivo es una imagen
                            if re.match(r'.*\.(jpg|jpeg|png|gif|bmp|tif)$', file):
                                # checks if get_measure is true and if the image contains the word "scale"
                                #comprueba si get_measure es verdadero y si la imagen contiene la palabra "scale"
                                if re.search(r'scale|escala', file, re.IGNORECASE):
                                    print("Scale detected: " + file)
                                    scaleFile = file
                                    pass
                                elif calcCircles:
                                    # to detect the circles in the image
                                    #detecta los círculos en la imagen
                                    print("Processing image: " + file)
                                    #creates a thread for each image
                                    #crea un hilo para cada imagen
                                    futures.append(executor.submit(detectCircles, os.path.join(root, file), newOutputDir, outputDir,sampleNum))
                                    #detectCircles(os.path.join(root, file), newOutputDir, outputDir,sampleNum)
                        # waits for all threads to complete
                        #espera a que todos los hilos se completen
                        for future in as_completed(futures):
                            pass
                    if get_measure:
                        if scaleFile:
                            # detects the measure in the image
                            #detecta la medida en la imagen
                            print("Processing scale: " + scaleFile)
                            print("------------------------------------")
                            # creates a thread for each image
                            #crea un hilo para cada imagen
                            executor.submit(detectMeasure, os.path.join(root, scaleFile), newOutputDir, outputDir)
                            #detectMeasure(os.path.join(root, scaleFile), newOutputDir, outputDir)
                        else:
                            # detects the measure in the image
                            # detecta la medida en la imagen
                            print("Processing scale using default scale: ")
                            print("------------------------------------")
                            # creates a thread for each image
                            # crea un hilo para cada imagen
                            executor.submit(detectMeasure, None, newOutputDir, outputDir,defaultPixelSize=DEFAULT_PIXED_SIZE)
                            #detectMeasure(None, newOutputDir, outputDir,defaultPixelSize=DEFAULT_PIXED_SIZE)
            if join_data:
                joinGlobal(newOutputDir, outputDir)

# in newOutputDir we will store the images with the detected circles and the results of the detection for the specific sample and in outputDir we will store the results of the detection to have all the detections in the same file
# en newOutputDir guardaremos las imágenes con los círculos detectados y los resultados de la detección de la muestra específica y en outputDir guardaremos los resultados de la detección para tener todas las detecciones en el mismo archivo
def detectCircles(imagePath, newOutputDir, outputDir,sampleNum):
    print("Detecting circles in image: " + imagePath)
    try:
        
        image = mpimg.imread(imagePath)
        # reduces the size of the image
        #reduce el tamaño de la imagen
        image = image[::SIZE_REDUCTION_RESOLUTION, ::SIZE_REDUCTION_RESOLUTION]

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT_ALT, 1.5, minDist=10, param1=300, param2=0.9, minRadius=10, maxRadius=500)

        circles = np.round(circles[0, :]).astype("int")
        cx, cy, radii = [], [], []

        for (x, y, r) in circles:
            cx.append(x)
            cy.append(y)
            radii.append(r)

        # draw the circles on the original image
        #dibuja los círculos en la imagen original
        image_with_circles = image.copy()
        fig, ax = plt.subplots()
        ax.imshow(image_with_circles)

        i = 0
        for center_y, center_x, radius in zip(cy, cx, radii):
            #print(f"Circle {i+1}: center: ({center_x}, {center_y}), radius: {radius*2*SIZE_REDUCTION_RESOLUTION}")
            i += 1
            circle = Circle((center_x, center_y), radius, fill=False, edgecolor='red')
            ax.add_patch(circle)

        # draw additional circles to create a thickness
        # dibuja circulos adicionales para crear un grosor
        thickness= 5
        for i in range(1, thickness):
            for center_y, center_x, radius in zip(cy, cx, radii):
                circle = Circle((center_x, center_y), radius+i, fill=False, edgecolor='red')
                ax.add_patch(circle)

            # Annotate the circle number at the center of the circle
            # un loop aparte para poner el numero de cada circulo evitando que se superpongan con los círculos
            i = 0
            for center_y, center_x, radius in zip(cy, cx, radii):
                #print(f"Circle {i+1}: center: ({center_x}, {center_y}), radius: {radius*2*SIZE_REDUCTION_RESOLUTION}")
                i += 1
                circle = Circle((center_x, center_y), radius, fill=False, edgecolor='red')
                ax.add_patch(circle)
                # Annotate the circle number at the center of the circle
                # Anota el número del círculo en el centro del círculo
                ax.annotate(str(i+1), (center_x, center_y), color='blue', fontsize=12)

        # Save the image with the circles
        #guarda la imagen con los círculos
        canvas = FigureCanvas(fig)
        canvas.draw()
        image_from_plot = canvas.buffer_rgba()
        image_with_circles = np.asarray(image_from_plot)



        image_base_name, image_extension = os.path.splitext(os.path.basename(imagePath))
        if image_extension.lower() == '.tif':
            image_extension = '.tiff'
        new_image_name = image_base_name + "_with_circles" + image_extension
        plt.imsave(os.path.join(newOutputDir, new_image_name), image_with_circles)


        print("************************************")
        print("Image processed: " + imagePath)

        with open(os.path.join(newOutputDir, 'results.csv'), 'a') as f:
            # add the columns to the csv (which already have the header)
            #añade las columnas al csv (que ya tienen el encabezado)
            for center_y, center_x, radius in zip(cy, cx, radii):
                f.write(f"{center_x},{center_y},{radius*2*SIZE_REDUCTION_RESOLUTION},{imagePath}\n")

        print("Image data stored: " + imagePath)

    except Exception as e:
        print(f"Error processing {imagePath}: {str(e)}")



# Main function that creates and executes the interface
# Función principal que crea y ejecuta la interfaz
def main():
    global win, input_path, output_path, calc_circles, get_measure, measure_distance
    global useDefaultScale, scale_path, joid_data, onlyOne, sampleNum
    global input_label, output_label, scale_label, quantity_entry
    


    from tkinter import ttk, filedialog
    import sv_ttk
    import os

    def save_settings(input_dir, output_dir, measure_distance):
        with open('settings.txt', 'w') as f:
            f.write(f"input={input_dir}\n")
            f.write(f"output={output_dir}\n")
            f.write(f"calc_circles={calc_circles.get()}\n")
            f.write(f"get_measure={get_measure.get()}\n")
            f.write(f"measure_distance={measure_distance.get()}\n")
            f.write(f"useDefaultScale={useDefaultScale.get()}\n")
            f.write(f"scale_path={scale_path.get()}\n")
            f.write(f"joid_data={joid_data.get()}\n")
            f.write(f"onlyOne={onlyOne.get()}\n")
            f.write(f"sampleNum={sampleNum.get()}\n")

    def load_settings():
        try:
            with open('settings.txt', 'r') as f:
                lines = f.readlines()
                settings = {}
                for line in lines:
                    key, value = line.strip().split('=')
                    settings[key] = value
                return settings.get('input'), settings.get('output'), settings.get('calc_circles'), settings.get('get_measure'), settings.get('measure_distance'), settings.get('useDefaultScale'), settings.get('scale_path'), settings.get('joid_data'), settings.get('onlyOne'), settings.get('sampleNum')
        except FileNotFoundError:
            return None, None, None, None, None, None, None, None, None, None, None, None

    def callback2():
        # access to the entered routes
        # Acceder a las rutas ingresadas
        input_dir = input_path.get()
        output_dir = output_path.get()
        print(f"Ruta de lectura: {input_dir}")
        print(f"Ruta de guardado: {output_dir}")
        save_settings(input_dir, output_dir, measure_distance)
        Label(win, text="Settings saved!", font=('Century 20 bold')).pack(pady=4)

    # create an instance of the Tkinter window
    # Crear una instancia de la ventana Tkinter
    win = Tk()

    # Set the size of the window
    # Configurar la geometría de la ventana
    win.geometry("1000x750")

    # Aplicar el tema "dark"
    #ttk.set_theme('dark')

    # Functions to open the file explorer and select a directory path
    # Funciones para abrir el explorador de archivos y seleccionar una ruta de directorio
    def select_input_path():
        folder_selected = filedialog.askdirectory()
        input_path.set(folder_selected)
        input_label.config(text="Input image directory: " + folder_selected)

    def select_output_path():
        folder_selected = filedialog.askdirectory()
        output_path.set(folder_selected)
        output_label.config(text="Output image directory: " + folder_selected)

    # Create the input widgets
    # Crear los widgets de entrada
    input_path = StringVar()
    input_button = Button(win, text="Select image input directory:", command=select_input_path)
    input_button.pack()

    input_label = Label(win, text="")
    input_label.pack()

    output_path = StringVar()
    output_button = Button(win, text="Select image output directory:", command=select_output_path)
    output_button.pack()

    output_label = Label(win, text="")
    output_label.pack()

    # create a button
    # Crear un botón
    btn = ttk.Button(win, text="Save predefined settings", command=callback2)
    btn.pack(ipadx=10, pady=10)

    def confirm_delete():
        confirm_win = Toplevel(win)
        confirm_win.geometry("200x100")
        Label(confirm_win, text="Are you sure?").pack()
        Button(confirm_win, text="Yes", command=delete_settings).pack()
        Button(confirm_win, text="No", command=confirm_win.destroy).pack()

    def delete_settings():
        try:
            os.remove('settings.txt')
            input_path.set("")
            output_path.set("")
            calc_circles.set(False)
            get_measure.set(False)
            measure_distance.set(0)
            input_label.config(text="")
            output_label.config(text="")
            scale_label.config(text="")
            useDefaultScale.set(False)
            joid_data.set(False)
            onlyOne.set(False)
            sampleNum.set(0)
        except FileNotFoundError:
            pass

    delete_button = ttk.Button(win, text="Delete all predefined settings", command=confirm_delete)
    delete_button.pack(pady=10)

    def callback():
        # access to the entered routes
        # Acceder a las rutas ingresadas
        input_dir = input_path.get()
        output_dir = output_path.get()
        print(f"Ruta de lectura: {input_dir}")
        print(f"Ruta de guardado: {output_dir}")
        Label(win, text="Calculating circles...", font=('Century 20 bold')).pack(pady=4)
        processImages(input_dir,output_dir,measure_distance,get_measure.get(),calc_circles.get(),useDefaultScale.get(),scale_path.get(),joid_data.get(),onlyOne.get(),sampleNum.get())
        time_str = f"Time consumed: {TIME_CONSUMED:.2f} seconds"
        Label(win, text=time_str, font=('Century 20 bold')).pack(pady=4)

    #tickbox for the user to select the option to calculate the circles
    # tickbox para que el usuario seleccione la opción de calcular los círculos

    # create the checkbox
    # Crear la casilla de verificación
    calc_circles = BooleanVar()
    check_button = ttk.Checkbutton(win, text="Calculate Circles", variable=calc_circles)
    check_button.pack()



    ####################################
    # Scale checkbox and scale image file selector
    # Scale checkbox y scale image file selector
    frameScale = ttk.Frame(win)
    frameScale.pack()

    useDefaultScale = BooleanVar()

    check_button = ttk.Checkbutton(frameScale, text="Use default scale", variable=useDefaultScale)
    check_button.pack(side=LEFT)

    def select_scale_path():
        folder_selected = filedialog.askopenfilename()
        scale_path.set(folder_selected)
        scale_label.config(text="Scale image file: " + folder_selected)

    scale_path = StringVar()
    scale_button = ttk.Button(win, text="Select default scale image:", command=select_scale_path)
    scale_button.pack()

    scale_label = Label(win, text="")
    scale_label.pack()

    ##########################################
    #tickbox to apply the measurement to a single sample
    # tickbox para aplicar la medida a una sola muestra
    frameScale = ttk.Frame(win)
    frameScale.pack()

    onlyOne = BooleanVar()

    check_button = ttk.Checkbutton(frameScale, text="Apply to the specified sample", variable=onlyOne)
    check_button.pack(side=LEFT)

    #quantity of the real distance
    sampleNum = IntVar()
    sampleEntry = ttk.Entry(frameScale, textvariable=sampleNum)
    sampleEntry.pack(side=LEFT)
    ##########################################
    # tickbox to join the data in a single file
    # tickbox para unir los datos en un solo fichero
    joid_data = BooleanVar()
    check_button = ttk.Checkbutton(win, text="Join data", variable=joid_data)
    check_button.pack()

    ####################################
    # tickbox to measure the real distance
    #tickbox para medir la distancia real
    get_measure = BooleanVar()
    get_measure.set(True) 

    # Creates a frame to contain the checkbox and the entries
    # Crea un marco para contener la casilla de verificación y las entradas
    frame = ttk.Frame(win)
    frame.pack()

    check_button = ttk.Checkbutton(frame, text="Measure the real distance", variable=get_measure)
    check_button.pack(side=LEFT)

    # Quantity of the real distance
    #cantidad de la distancia real
    measure_distance = IntVar()
    quantity_entry = ttk.Entry(frame, textvariable=measure_distance)
    quantity_entry.pack(side=LEFT)

    def check_get_measure(*args):
        if get_measure.get():
            quantity_entry.pack(side=LEFT)
        else:
            quantity_entry.pack_forget()

    # Adds a tracker to get_measure
    # Añade un rastreador a get_measure
    get_measure.trace_add('write', check_get_measure)


    btn = ttk.Button(win, text="Initiate detection", command=callback)
    btn.pack(ipadx=10)

    win.bind('<Return>', lambda event: callback())

    # Loads the routes from the configuration file when starting the program
    # Carga las rutas desde el archivo de configuración al iniciar el programa
    input_dir, output_dir, calc_circles_value, get_measure_value, measure_distance_value, useDefaultScale_value, scale_path_value, joid_data_value, onlyOne_value, sampleNum_value = load_settings()
    if input_dir:
        print(input_dir)
        input_path.set(input_dir)
        input_label.config(text="Input image directory: " + input_path.get())
    if output_dir:
        output_path.set(output_dir)
        output_label.config(text="Output image directory: " + output_path.get())
    if calc_circles_value is not None:
        calc_circles.set(calc_circles_value == 'True')
    if get_measure_value is not None:
        get_measure.set(get_measure_value == 'True')
    if measure_distance_value is not None:
        measure_distance.set(int(measure_distance_value))
    if useDefaultScale_value is not None:
        useDefaultScale.set(useDefaultScale_value == 'True')
    if scale_path_value is not None:
        scale_path.set(scale_path_value)
        scale_label.config(text="Scale image directory: " + scale_path.get())
    if joid_data_value is not None:
        joid_data.set(joid_data_value == 'True')
    if onlyOne_value is not None:
        onlyOne.set(onlyOne_value == 'True')
    if sampleNum_value is not None:
        sampleNum.set(int(sampleNum_value))

    sv_ttk.set_theme("dark")
    win.mainloop()

# Punto de entrada del programa
if __name__ == "__main__":
    main()