# Imports externos
import numpy as np
import time as T

# Imports propios
from modules import scenarioFiller as SF
from modules import Fi2Pop,ILS,ILS2F,Fi2Pop2F
from modules import reverse as R
from utils import colorRender as color

# CONFIG
nHight = 10
mWidth = 10
expantionFactor = 0.1
enemyFactor =  0.09
blockFactor = 0.083
maxIter = 10
nPop = 10
mutationFactor = 0.5
experiments = 5

# MAIN
inicio = T.time()
print(f"Se inicia la generación del mapa")

borderDungeon = SF.scenarioFiller(nHight,mWidth,expantionFactor)
# initialSolution = SF.getIntialSol(dungeon.copy(),enemyFactor,blockFactor)
# dungeon,bestFitness = ILS2F.ils(initialSolution,maxIter)

"""
for i in range(1,experiments +1):
    dungeon,bestFitness = ILS.ils(initialSolution,maxIter)
    np.savetxt('results/result.csv', dungeon, delimiter=',', fmt='%d')
    color.renderMatrix(dungeon)
    print(f"Nuevo mapa {i}\n")

"""

for i in range(1,experiments+1):
    print(f"Experimento {i}")
    population = []
    for _ in range(nPop*2):
        population.append(SF.getIntialSol(borderDungeon.copy(),enemyFactor,blockFactor))
    dungeon, fitness = Fi2Pop2F.FI2Pop(population,maxIter,nPop,mutationFactor)
    
    color.renderMatrix(dungeon)


print(f"Mapa <result.csv> generado en {T.time() - inicio} segundos")