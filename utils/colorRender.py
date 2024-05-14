
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import PatternFill
from datetime import datetime


def html_to_argb(html_color):
    # Reorganizar el código de color HTML para que sea compatible con openpyxl (RRGGBB a AABBGGRR)
    hex_color = html_color[1:]
    argb_color = 'FF' + hex_color[4:] + hex_color[2:4] + hex_color[:2]
    return argb_color

def renderMatrix(map):
    wb = Workbook()
    hoja = wb.active
    for fila_idx, fila in enumerate(map, start=1):
        for celda_idx, valor_celda in enumerate(fila, start=1):
            hoja.cell(row=fila_idx, column=celda_idx, value=valor_celda)
            if valor_celda == 0:
                relleno = PatternFill(start_color=html_to_argb("#f0f0f0"), end_color=html_to_argb("#f0f0f0"), fill_type="solid")  
                hoja.cell(row=fila_idx, column=celda_idx).fill = relleno
            if valor_celda == 1:
                relleno = PatternFill(start_color=html_to_argb("#c0c0c0"), end_color=html_to_argb("#c0c0c0"), fill_type="solid")  
                hoja.cell(row=fila_idx, column=celda_idx).fill = relleno
            if valor_celda == 2:
                relleno = PatternFill(start_color=html_to_argb("#ffb6c1"), end_color=html_to_argb("#ffb6c1"), fill_type="solid")  
                hoja.cell(row=fila_idx, column=celda_idx).fill = relleno  
            if valor_celda == 3:
                relleno = PatternFill(start_color=html_to_argb("#d2b48c"), end_color=html_to_argb("#d2b48c"), fill_type="solid")  
                hoja.cell(row=fila_idx, column=celda_idx).fill = relleno
            if valor_celda == 4:
                relleno = PatternFill(start_color=html_to_argb("#98fb98"), end_color=html_to_argb("#98fb98"), fill_type="solid")  
                hoja.cell(row=fila_idx, column=celda_idx).fill = relleno
            if valor_celda == 5:
                relleno = PatternFill(start_color=html_to_argb("#add8e6"), end_color=html_to_argb("#add8e6"), fill_type="solid")  
                hoja.cell(row=fila_idx, column=celda_idx).fill = relleno

    # Guardar los cambios en el archivo Excel
    now = datetime.now()
    formato_fecha = now.strftime("%d-%m_%H-%M-%S")
    nArchivo = "Dungeon "+ formato_fecha + ".xlsx"
    wb.save(f"results/{nArchivo}")