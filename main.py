# Imports externos
import numpy as np
import time as T

# Imports propios
from modules import scenarioFiller as SF
from modules import movements as M
from modules import reverse as R
from utils import colorRender as color

# MAIN
inicio = T.time()
print(f"Se inicia la generación del mapa")

enemyFactor =  0.085
blockFactor = 0.05
maxMoves = 1000
nHight = 5
mWidth = 5
expantionFactor = 0.1

dungeon = SF.scenarioFiller(nHight,mWidth,expantionFactor)
print(dungeon)
#dungeon,player = M.getDungeon(dungeon,maxMoves,blockFactor,enemyFactor)
dungeon = np.array(dungeon)

#np.savetxt('results/result.csv', dungeon, delimiter=',', fmt='%d')
#color.renderMatrix(dungeon)

print(f"Mapa <result.csv> generado en {T.time() - inicio} segundos")