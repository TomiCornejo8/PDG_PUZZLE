# External
import numpy as np
from anytree import Node
from collections import deque
import copy 

# Internal
from modules import mechanics as M

def getNEnemys(dungeon):
    return np.count_nonzero(dungeon == M.ENEMY)

def getEnemys(dungeon):
    return np.where(dungeon == M.ENEMY)

def getPlayer(dungeon):
    return np.where(dungeon == M.PLAYER)

def inDoor(player,dungeon):
    for move in M.moves:
        tile = M.lookAhead(player,move)
        if dungeon[tile[0],tile[1]] == M.DOOR:
            return True
    return False

def getDungeonHash(dungeon):
    return hash(tuple(map(tuple, dungeon)))

def setRoutes(routes,memoryStates,newDungeon,currentNode):
    dungeonHash = getDungeonHash(newDungeon)
    if not dungeonHash in memoryStates:
        memoryStates.append(dungeonHash)
        routes.append(Node(newDungeon,parent=currentNode))  
    return routes,memoryStates

def getRoutes(nodo):
    ruta = deque()
    while nodo is not None:
        ruta.append(nodo.name)
        nodo = nodo.parent
    return ruta

def solve(dungeon):
    routes = deque()
    memoryStates = []

    root = Node(dungeon.copy())
    routes.append(root)
    memoryStates.append(getDungeonHash(dungeon))
    solutions = []
        
    while routes:
        currentNode = routes.popleft()
        currentDungeon = currentNode.name
        player = getPlayer(currentDungeon)
        
        moves = M.getAllowMoves(currentDungeon,player)
        enemys = M.getMeleeEnemys(currentDungeon,player)

        for enemy in enemys:
            newDungeon = currentDungeon.copy()
            
            newDungeon = M.killEnemy(newDungeon,enemy)

            if getNEnemys(newDungeon) == 0 and inDoor(player,newDungeon): 
                solutions.append(getRoutes(Node(newDungeon,parent=currentNode)))
            else:
                routes,memoryStates = setRoutes(routes,memoryStates,newDungeon,currentNode)

        for move in moves:
            newDungeon = currentDungeon.copy()

            newPlayer = M.iceSliding(newDungeon,copy.copy(player),move)
            newDungeon[player[0],player[1]] = M.EMPTY
            newDungeon[newPlayer[0],newPlayer[1]] = M.PLAYER
            
            if getNEnemys(newDungeon) == 0 and inDoor(newPlayer,newDungeon): 
                solutions.append(getRoutes(Node(newDungeon,parent=currentNode)))
            else:
                routes,memoryStates = setRoutes(routes,memoryStates,newDungeon,currentNode)
    return solutions,root