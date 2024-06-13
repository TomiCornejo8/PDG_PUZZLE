# Imports externos
import time as T
import random

# Imports propios
from modules import ScenarioFiller as SF
from modules import Fi2Pop
from helpers import colorRender as color
from modules import solver as sol
# CONFIG
nHight = 10
mWidth = 10

expantionFactor = 0.13
enemyFactor =  0.083
blockFactor = 0.1

nPop = 12
mutationFactor = 0.5

maxIter = 40
maxMoves = 25
experiments = 1000
newMapShape = 10

# MAIN
inicio = T.time()
print(f"Se inicia la generación del mapa")

borderDungeon = SF.scenarioFiller(nHight,mWidth,expantionFactor)

for i in range(1,experiments+1):
    print(f"Experimento {i}")
    population = []
    for _ in range(nPop*2):
        population.append(SF.getIntialSol(borderDungeon.copy(),enemyFactor,blockFactor))
    if i % newMapShape == 0:
            borderDungeon = SF.scenarioFiller(nHight,mWidth,expantionFactor)
    maxMoves = random.randint(20,25)
    dungeon, fitness , nSol, minMoves = Fi2Pop.FI2Pop(population,maxIter,nPop,mutationFactor,maxMoves)
    color.renderMatrix(dungeon,fitness, nSol, minMoves)


print(f"Mapa <result.csv> generado en {T.time() - inicio} segundos")