# Imports externos
import numpy as np
import time as T

# Imports propios
from modules import scenarioFiller as SF
from modules import Fi2Pop
from modules import movements as M
from modules import reverse as R
from utils import colorRender as color

# CONFIG
nHight = 10
mWidth = 10
expantionFactor = 0.1
enemyFactor =  0.10
blockFactor = 0.12
maxIter = 20
nPop = 10
mutationFactor = 0.5

# MAIN
inicio = T.time()
print(f"Se inicia la generación del mapa")

dungeon = SF.scenarioFiller(nHight,mWidth,expantionFactor)

population = []

for _ in range(nPop*2):
    population.append(SF.getIntialSol(dungeon.copy(),enemyFactor,blockFactor))

dungeon, fitness = Fi2Pop.FI2Pop(population,maxIter,nPop,mutationFactor)


np.savetxt('results/result.csv', dungeon, delimiter=',', fmt='%d')
color.renderMatrix(dungeon)

print(f"Mapa <result.csv> generado en {T.time() - inicio} segundos")