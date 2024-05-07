# Imports externos
import numpy as np
import time as T

# Imports propios
from modules import scenarioFiller as SF
from modules import movements as M
from utils import colorRender as color

# MAIN
inicio = T.time()
print(f"Se inicia la generación del mapa")

enemyFactor =  0.085
blockFactor = 0.05
maxMoves = 1000
nHight = 10
mWidth = 10
expantionFactor = 0.1

map = SF.scenarioFiller(nHight,mWidth,expantionFactor)
map,player = M.getMap(map,maxMoves,blockFactor,enemyFactor)
map = np.array(map)

np.savetxt('results/result.csv', map, delimiter=',', fmt='%d')
color.renderMatrix(map)

print(f"Mapa <result.csv> generado en {T.time() - inicio} segundos")