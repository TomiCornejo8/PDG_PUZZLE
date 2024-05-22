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
    for move in M.MOVES:
        tile = M.lookAhead(player,move)
        if dungeon[tile[0],tile[1]] == M.DOOR:
            return True
    return False

def getDungeonHash(dungeon):
    return hash(tuple(map(tuple, dungeon)))

def setRoutesTree(routes,memoryStates,newDungeon,currentNode):
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

def setRoutes(routes,memoryStates,newDungeon):
    dungeonHash = getDungeonHash(newDungeon)
    if not dungeonHash in memoryStates:
        memoryStates.append(dungeonHash)
        routes.append(newDungeon)  
    return routes,memoryStates

def playable(dungeon):
    routes = deque()
    memoryStates = []

    routes.append(dungeon)
    memoryStates.append(getDungeonHash(dungeon))
      
    while routes:
        currentDungeon = routes.popleft()
        player = getPlayer(currentDungeon)
        
        moves = M.getAllowMoves(currentDungeon,player)
        enemys = M.getMeleeEnemys(currentDungeon,player)
        
        for enemy in enemys:
            newDungeon = currentDungeon.copy()
            newDungeon = M.killEnemy(newDungeon,enemy)

            if getNEnemys(newDungeon) == 0 and inDoor(player,newDungeon): 
                return True
            else:
                routes,memoryStates = setRoutes(routes,memoryStates,newDungeon)

        for move in moves:
            newDungeon = currentDungeon.copy()

            newDungeon = M.iceSliding(newDungeon,player,move)

            newPlayer = getPlayer(newDungeon)
            
            if getNEnemys(newDungeon) == 0 and inDoor(newPlayer,newDungeon): 
                  return True
            else:
                routes,memoryStates = setRoutes(routes,memoryStates,newDungeon)
    return False

def nSolutions(dungeon):
    solutions = solve(dungeon)
    
    n = len(solutions)
    if n==0: return n,0
    
    minM = len(solutions[0])
    for sol in solutions:
        m = len(sol)
        if m < minM:
            minM = m
    return n,minM
    
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
                routes,memoryStates = setRoutesTree(routes,memoryStates,newDungeon,currentNode)

        for move in moves:
            newDungeon = currentDungeon.copy()

            newDungeon = M.iceSliding(newDungeon,player,move)
            newPlayer  = getPlayer(newDungeon)
            
            if getNEnemys(newDungeon) == 0 and inDoor(newPlayer,newDungeon): 
                solutions.append(getRoutes(Node(newDungeon,parent=currentNode)))
            else:
                routes,memoryStates = setRoutesTree(routes,memoryStates,newDungeon,currentNode)
    return solutions

class DungeonWMoves:
    def __init__(self,dungeon,movement):
        self.dungeon = dungeon
        self.move = movement

def setRoutesTreeWMoves(routes,memoryStates,newDungeon,currentNode):
    dungeonHash = getDungeonHash(newDungeon.dungeon)
    if not dungeonHash in memoryStates:
        memoryStates.append(dungeonHash)
        routes.append(Node(newDungeon,parent=currentNode))  
    return routes,memoryStates

def getMove(move):
    if move == M.RIGHT: return "RIGHT"
    elif move == M.LEFT: return "LEFT"
    elif move == M.UP: return "UP"
    else: return "DOWN"

def getKill(player,move):
    playerInt=[int(player[0]),int(player[1])]
    moveInt=[int(move[0]),int(move[1])]    
    movement = [moveInt[0] - playerInt[0],moveInt[1] - playerInt[1]]
    if movement == [0,1]: return "RIGHT"
    elif movement == [0,-1]: return "LEFT"
    elif movement == [-1,0]: return "UP"
    else: return "DOWN"

def solverWithMoves(dungeon):
    routes = deque()
    memoryStates = []
    dungeonWMoves = DungeonWMoves(dungeon,"")
    root = Node(dungeonWMoves)
    routes.append(root)
    memoryStates.append(getDungeonHash(dungeonWMoves.dungeon))

    solutions = []
        
    while routes:
        currentNode = routes.popleft()
        currentDungeon = currentNode.name
        player = getPlayer(currentDungeon.dungeon)
        
        moves = M.getAllowMoves(currentDungeon.dungeon,player)
        enemys = M.getMeleeEnemys(currentDungeon.dungeon,player)

        for enemy in enemys:
            newDungeon = copy.deepcopy(currentDungeon)
            newDungeon = DungeonWMoves(M.killEnemy(newDungeon.dungeon,enemy),"Kill " + getKill(player,enemy))

            if getNEnemys(newDungeon.dungeon) == 0 and inDoor(player,newDungeon.dungeon): 
                solutions.append(getRoutes(Node(newDungeon,parent=currentNode)))
            else:
                routes,memoryStates = setRoutesTreeWMoves(routes,memoryStates,newDungeon,currentNode)

        for move in moves:
            newDungeon = copy.deepcopy(currentDungeon)

            newDungeon = M.iceSliding(newDungeon.dungeon,player,move)
            newPlayer  = getPlayer(newDungeon)
            newDungeon =  DungeonWMoves(newDungeon,"Move "+ getMove(move))
            if getNEnemys(newDungeon.dungeon) == 0 and inDoor(newPlayer,newDungeon.dungeon): 
                solutions.append(getRoutes(Node(newDungeon,parent=currentNode)))
            else:
                routes,memoryStates = setRoutesTreeWMoves(routes,memoryStates,newDungeon,currentNode)
    return solutions  