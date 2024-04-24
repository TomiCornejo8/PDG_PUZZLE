# Imports externos
import numpy as np
import time as T

# Imports propios
from modules import ScenarioFiller as SF
from auxiliar import colorRender as color

# MAIN
inicio = T.time()
print(f"Se inicia la generación del mapa")

map = SF.scenarioFiller(100,100,0.5)

voidSpace = np.argwhere(map == 0)
player = voidSpace[np.random.choice(len(voidSpace))]
map[player[0],player[1]] = 5

np.savetxt('results/result.csv', map, delimiter=',', fmt='%d')

print(f"Mapa <result.csv> generado en {T.time() - inicio} segundos")

color.renderMatrix(map)