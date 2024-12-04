# Imports externos
import time as T
import random
import csv

# Imports propios
from searchBase import scenarioFiller as SF
from searchBase import Fi2Pop
from helpers import colorRender as color
from searchBase import solver as sol
# CONFIG
nHight = 10
mWidth = 10

expantionFactor = 0.13
enemyFactor =  0.083
blockFactor = 0.1

nPop = 12
mutationFactor = 0.5

maxIter = 20
maxMoves = 25
experiments = 200
newMapShape = 10

# MAIN
inicio = T.time()
print(f"Se inicia la generación de {experiments} mapas")

borderDungeon = SF.scenarioFiller(nHight,mWidth,expantionFactor)

meanStop = 0
stops = []
with open("result.csv", mode="w", newline="") as file:
    for i in range(1,experiments+1):
        print(f"Experimento {i}")
        population = []
        for _ in range(nPop*2):
            population.append(SF.getIntialSol(borderDungeon.copy(),enemyFactor,blockFactor))
        if i % newMapShape == 0:
                borderDungeon = SF.scenarioFiller(nHight,mWidth,expantionFactor)
        maxMoves = random.randint(20,25)
        dungeon, fitness , nSol, minMoves, stop = Fi2Pop.FI2Pop(population,maxIter,nPop,mutationFactor,maxMoves)
        writer = csv.writer(file)
        writer.writerow([fitness, nSol, minMoves, stop])
        meanStop += stop
        stops.append(stop)
        color.renderMatrix(dungeon,fitness, nSol, minMoves)

meanStop /= experiments

print(f"Mapas generados en {T.time() - inicio} segundos. Tiempo promedio {meanStop} por mapa.")