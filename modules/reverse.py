from modules import mechanics as M
import numpy as np
import random

# Functions
def getAdjacentWalls(dungeon):
    # Obtener las posiciones de los 1s en la dungeon
    walls = np.argwhere(dungeon == M.WALL)
    adjacentWalls = []

    for i, j in walls:
        # Verificar si el 1 está adyacente a un 0 en forma de cruz
        if (i > 0 and dungeon[i-1][j] == M.EMPTY) or \
           (i < dungeon.shape[0]-1 and dungeon[i+1][j] == M.EMPTY) or \
           (j > 0 and dungeon[i][j-1] == 0) or \
           (j < dungeon.shape[1]-1 and dungeon[i][j+1] == M.EMPTY):
            adjacentWalls.append((i, j))

    return adjacentWalls

def setDoor(dungeon):
    allowDoorPosition = getAdjacentWalls(dungeon)
    door = random.choice(allowDoorPosition)
    dungeon[door[0],door[1]] = M.DOOR
    return dungeon,door

def iceSliding(dungeon,player,move,stopFactor):
    while dungeon[M.lookAhead(player,move)[0],M.lookAhead(player,move)[1]] == M.EMPTY: 
        dungeon[player[0],player[1]] = M.TRAIL
        player = M.lookAhead(player,move)
        r1 = random.uniform(0,1)
        if r1 <= stopFactor: break
    return dungeon,player

def getAllowAxis(dungeon,player,moves):
    notAllow = [M.WALL,M.BLOCK,M.ENEMY,M.DOOR]
    newMoves = []
    for move in moves:
        tile = M.lookAhead(player,move)
        if not dungeon[tile[0],tile[1]] in notAllow:
            newMoves.append(move)
    return newMoves

def backSlidding(dungeon,player,move,axi):
    move = [move[0]*-1 , move[1]*-1]
    while dungeon[M.lookAhead(player,move)[0],M.lookAhead(player,move)[1]] == M.TRAIL: 
        dungeon[player[0],player[1]] = M.EMPTY
        player = M.lookAhead(player,move)
        if axi == "X":
            if len(getAllowAxis(dungeon,player,M.Y)) > 0: break
        else:
            if len(getAllowAxis(dungeon,player,M.X)) > 0: break
    return dungeon,player

def getDungeon(dungeon,maxMoves):
    # Se posiciona la puerta
    dungeon,door= setDoor(dungeon)

    # Se posiciona al "jugador" en una de las casillas perpendiculares de la puerta
    moves = M.getAllowMoves(dungeon,door)
    move = random.choice(moves)
    player = M.lookAhead(door,move)

    # Se elige el movimiento inicial
    moves = M.getAllowMoves(dungeon,player)
    move = random.choice(moves)

    if move in M.X:
        axi = 'X'
    else:
        axi = 'Y'

    stopFactor = 0.1
    
    for _ in range(maxMoves-1):
        dungeon,player = iceSliding(dungeon,player,move,stopFactor)
        if axi == 'X':
            moves = getAllowAxis(dungeon,player,M.Y)
            if len(moves) == 0:
                dungeon, newPlayer = backSlidding(dungeon,player,move,axi)
                if newPlayer == player:
                    axi = 'Y'
                    moves = getAllowAxis(dungeon,player,M.X)
                    move = []
            axi = 'Y'
        else:
            moves = getAllowAxis(dungeon,player,M.X)
            axi = 'X'
        
    # if len(moves) == 0:
    return(getAdjacentWalls(dungeon))