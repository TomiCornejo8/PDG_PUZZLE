# Imports externos
import time as T

# Imports propios
from modules import ScenarioFiller as SF
from modules import Fi2Pop
from utils import colorRender as color

# CONFIG
nHight = 10
mWidth = 10

expantionFactor = 0.1
enemyFactor =  0.08
blockFactor = 0.08

nPop = 20
mutationFactor = 0.5

maxIter = 100
maxMoves = 30
experiments = 1

# MAIN
inicio = T.time()
print(f"Se inicia la generación del mapa")

borderDungeon = SF.scenarioFiller(nHight,mWidth,expantionFactor)

for i in range(1,experiments+1):
    print(f"Experimento {i}")
    population = []
    for _ in range(nPop*2):
        population.append(SF.getIntialSol(borderDungeon.copy(),enemyFactor,blockFactor))
    dungeon, fitness , nSol, minMoves = Fi2Pop.FI2Pop(population,maxIter,nPop,mutationFactor,maxMoves)
    
    color.renderMatrix(dungeon,fitness, nSol, minMoves)


print(f"Mapa <result.csv> generado en {T.time() - inicio} segundos")