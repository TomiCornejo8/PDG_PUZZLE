from openpyxl import Workbook
from openpyxl.styles import PatternFill
from openpyxl.utils import get_column_letter
from datetime import datetime
import numpy as np
import csv
import os

from modules import solver as S

def csv_to_numpy_matrix(file):
    with open(f'{file}', 'r') as csvfile:
        reader = csv.reader(csvfile)
        data = list(reader)
    
    # Convertimos los datos leídos a un array de NumPy
    numpy_matrix = np.array(data, dtype=int)
    return numpy_matrix

def bestSolution(dungeon):
    solutions = S.solve(dungeon)
    if len(solutions) > 0:
        best = solutions[0]
        for sol in solutions:
            if len(sol) < len(best):
                best = sol
    return best
# Función para crear una carpeta si no existe
def crear_carpeta_si_no_existe(nombre_carpeta):
    if not os.path.exists(nombre_carpeta):
        os.makedirs(nombre_carpeta)

# Función para obtener el número siguiente para el nombre del archivo
def obtener_numero_siguiente_carpeta(nombre_carpeta):
    if not os.path.exists(nombre_carpeta):
        return 1
    else:
        return len(os.listdir(nombre_carpeta)) + 1

def createFoldersCsv(map):
    now = datetime.now()
    formato_fecha = now.strftime("%d-%m")

    ExperimentFolder = "Results/Experiment " + formato_fecha
    solutionsFolder = os.path.join(ExperimentFolder,"SolutionsCsv")

    crear_carpeta_si_no_existe(ExperimentFolder)
    crear_carpeta_si_no_existe(solutionsFolder)

    nFile = obtener_numero_siguiente_carpeta(solutionsFolder)
    csvName = f"Solution_{nFile}.csv"
    csvFileName = os.path.join(solutionsFolder, csvName)

    np.savetxt(f'{csvFileName}', map, delimiter=',', fmt='%d')
    return csvFileName

def createFoldersExcel(wb):
    now = datetime.now()
    formato_fecha = now.strftime("%d-%m")

    ExperimentFolder = "Results/Experiment " + formato_fecha
    dungeonsFolder = os.path.join(ExperimentFolder, "Dungeons")

    crear_carpeta_si_no_existe(ExperimentFolder)
    crear_carpeta_si_no_existe(dungeonsFolder)

    nFile= obtener_numero_siguiente_carpeta(dungeonsFolder)
    dungeonName = f"Dungeon_{nFile}.xlsx"
    excelFileName = os.path.join(dungeonsFolder,dungeonName)

    wb.save(f"{excelFileName}")

def ajustar_ancho_columnas(worksheet,middleColumn):
    i=1
    for column_cells in worksheet.columns:
        max_length = 0
        column = column_cells[0].column_letter  # Obtiene la letra de la columna
        for cell in column_cells:
            try:
                if len(str(cell.value)) > max_length:
                    max_length = len(cell.value)
            except:
                pass
        if middleColumn <= i:
            adjusted_width = (max_length + 2)
            worksheet.column_dimensions[column].width = adjusted_width
        i+=1
 
def renderMatrix(map, fitness, nSol, minMoves):
    wb = Workbook()
    hoja = wb.active
    for fila_idx, fila in enumerate(map, start=1):
        for celda_idx, valor_celda in enumerate(fila, start=1):
            hoja.cell(row=fila_idx, column=celda_idx, value=valor_celda)
            if valor_celda == 0:
                color = "F0F0F0"
            elif valor_celda == 1:
                color = "C0C0C0"
            elif valor_celda == 2:
                color = "D2B48C"
            elif valor_celda == 3:
                color = "FFB6C1"
            elif valor_celda == 4:
                color = "98FB98"
            elif valor_celda == 5:
                color = "ADD8E6"
            relleno = PatternFill(start_color=color, end_color=color, fill_type="solid")
            hoja.cell(row=fila_idx, column=celda_idx).fill = relleno


   
    lastRow = (map.shape[0] // 2) - 1
    middleColumn = map.shape[1] + 2

    hoja.cell(row=lastRow, column=middleColumn,value="Fitness")
    hoja.cell(row=lastRow , column=middleColumn +1,value="Number of Solutions")
    hoja.cell(row=lastRow , column=middleColumn +2,value="Minimum Moves")

    lastRow += 1
    hoja.cell(row=lastRow, column=middleColumn,value=fitness)
    hoja.cell(row=lastRow , column=middleColumn +1,value=nSol)
    hoja.cell(row=lastRow , column=middleColumn +2,value=minMoves)

    ajustar_ancho_columnas(hoja,middleColumn)
   

    createFoldersExcel(wb)
    nArchivo=createFoldersCsv(map)
    #renderMatrixQueue(nArchivo)
    renderMatrixQueueWMoves(map)





def renderMatrixQueue(file):
    dungeon = csv_to_numpy_matrix(file)
    matrix_queue = bestSolution(dungeon)
    wb = Workbook()
    hoja = wb.active

    columna_inicio = 1
    while matrix_queue:
        map = matrix_queue.pop()
        for fila_idx, fila in enumerate(map, start=1):
            for celda_idx, valor_celda in enumerate(fila, start=1):
                hoja.cell(row=fila_idx, column=celda_idx + columna_inicio - 1, value=valor_celda)
                if valor_celda == 0:
                    color = "F0F0F0"
                elif valor_celda == 1:
                    color = "C0C0C0"
                elif valor_celda == 2:
                    color = "D2B48C"
                elif valor_celda == 3:
                    color = "FFB6C1"
                elif valor_celda == 4:
                    color = "98FB98"
                elif valor_celda == 5:
                    color = "ADD8E6"
                relleno = PatternFill(start_color=color, end_color=color, fill_type="solid")
                hoja.cell(row=fila_idx, column=celda_idx + columna_inicio - 1).fill = relleno

        columna_inicio += len(map[0]) + 1

    now = datetime.now()
    formato_fecha = now.strftime("%d-%m")
    ExperimentFolder = "Results/Experiment " + formato_fecha
    solutionsFolder = os.path.join(ExperimentFolder,"Solutions")

    crear_carpeta_si_no_existe(solutionsFolder)

    nFile= obtener_numero_siguiente_carpeta(solutionsFolder)
    dungeonName = f"Solution_{nFile}.xlsx"
    excelFileName = os.path.join(solutionsFolder,dungeonName)

    wb.save(f"{excelFileName}")


    wb = Workbook()
    hoja = wb.active
    
    for fila_idx, fila in enumerate(map, start=1):
        for celda_idx, valor_celda in enumerate(fila, start=1):
            hoja.cell(row=fila_idx, column=celda_idx, value=valor_celda)
            if valor_celda == 0:
                color = "F0F0F0"
            elif valor_celda == 1:
                color = "C0C0C0"
            elif valor_celda == 2:
                color = "D2B48C"
            elif valor_celda == 3:
                color = "FFB6C1"
            elif valor_celda == 4:
                color = "98FB98"
            elif valor_celda == 5:
                color = "ADD8E6"
            relleno = PatternFill(start_color=color, end_color=color, fill_type="solid")
            hoja.cell(row=fila_idx, column=celda_idx).fill = relleno


   
    lastRow = (map.shape[0] // 2) - 1
    middleColumn = map.shape[1] + 2

    hoja.cell(row=lastRow, column=middleColumn,value="Fitness")
    hoja.cell(row=lastRow , column=middleColumn +1,value="Number of Solutions")
    hoja.cell(row=lastRow , column=middleColumn +2,value="Minimum Moves")

    lastRow += 1
    hoja.cell(row=lastRow, column=middleColumn,value=fitness)
    hoja.cell(row=lastRow , column=middleColumn +1,value=nSol)
    hoja.cell(row=lastRow , column=middleColumn +2,value=minMoves)

    ajustar_ancho_columnas(hoja,middleColumn)
   

    createFoldersExcel(wb)
    nArchivo=createFoldersCsv(map)
    renderMatrixQueueWMoves(nArchivo)

def bestSolutionWMoves(dungeon):
    best =dungeon.pop()
    while dungeon:
        dungeono=dungeon.pop()
        if len(dungeono) < len(best):
            best = dungeono
    return best


def renderMatrixQueueWMoves(dungeon):
    dungeonWMoves=S.solverWithMoves(dungeon)
    matrix_queue = bestSolutionWMoves(dungeonWMoves)
    wb = Workbook()
    hoja = wb.active

    columna_inicio = 1
    while matrix_queue:
        map = matrix_queue.pop()
        lastRow = map.dungeon.shape[1] + 2 
        middleColumn = (map.dungeon.shape[0] // 2) + 1
        for fila_idx, fila in enumerate(map.dungeon, start=1):
            for celda_idx, valor_celda in enumerate(fila, start=1):
                hoja.cell(row=fila_idx, column=celda_idx + columna_inicio - 1, value=valor_celda)
                if valor_celda == 0:
                    color = "F0F0F0"
                elif valor_celda == 1:
                    color = "C0C0C0"
                elif valor_celda == 2:
                    color = "D2B48C"
                elif valor_celda == 3:
                    color = "FFB6C1"
                elif valor_celda == 4:
                    color = "98FB98"
                elif valor_celda == 5:
                    color = "ADD8E6"
                relleno = PatternFill(start_color=color, end_color=color, fill_type="solid")
                hoja.cell(row=fila_idx, column=celda_idx + columna_inicio - 1).fill = relleno

        if columna_inicio == 1: 
            hoja.cell(row=lastRow, column=middleColumn,value=map.move)
        columna_inicio += len(map.dungeon[0]) + 1
        if columna_inicio != 1:
            hoja.cell(row=lastRow, column=columna_inicio - middleColumn,value=map.move)


    now = datetime.now()
    formato_fecha = now.strftime("%d-%m")
    ExperimentFolder = "Results/Experiment " + formato_fecha
    solutionsFolder = os.path.join(ExperimentFolder,"Solutions")

    crear_carpeta_si_no_existe(solutionsFolder)

    nFile= obtener_numero_siguiente_carpeta(solutionsFolder)
    dungeonName = f"Solution_{nFile}.xlsx"
    excelFileName = os.path.join(solutionsFolder,dungeonName)

    wb.save(f"{excelFileName}")