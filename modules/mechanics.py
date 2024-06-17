import numpy as np

# Moves
RIGHT = (0,1)
LEFT = (0,-1)
UP = (-1,0)
DOWN = (1,0)
STAY = (0,0)

MOVES = [RIGHT,LEFT,UP,DOWN]

EMPTY = 0
WALL = 1
ENEMY = 3
BLOCK = 2
DOOR = 4
PLAYER = 5

# Functions

def getPlayer(dungeon):
    return np.where(dungeon == PLAYER)

def isDone(dungeon):
    enemys = np.where(dungeon == ENEMY)
    door = False
    for move in MOVES:
        tile = lookAhead(dungeon,move)
        if dungeon[tile[0],tile[1]] == DOOR:
            door = True
            break
    if len(enemys) == 0 and door:
        return True
    return False

def lookAhead(dungeon,move):
    player = getPlayer(dungeon)
    return [player[0]+move[0],player[1]+move[1]]

def iceSliding(dungeon,move):
    
    while dungeon[lookAhead(dungeon,move)[0],lookAhead(dungeon,move)[1]] == EMPTY:
        player0 = getPlayer(dungeon) 
        player1 = lookAhead(dungeon,move)
        dungeon[player0[0],player0[1]] = EMPTY
        dungeon[player1[0],player1[1]] = PLAYER
          
    return dungeon

def killEnemy(dungeon,enemy):
    dungeon[enemy[0],enemy[1]] = EMPTY
    return dungeon

def getAllowMoves(dungeon,player):
    notAllow = [WALL,BLOCK,ENEMY,DOOR]
    newMoves = []
    for move in MOVES:
        tile = lookAhead(dungeon,move)
        dungeonTile = dungeon[tile[0],tile[1]]
        if not dungeonTile in notAllow:
            newMoves.append(move)

    return newMoves

def getMeleeEnemys(dungeon,player):
    enemys = []
    for move in MOVES:
        tile = lookAhead(dungeon,move)
        if dungeon[tile[0],tile[1]] == ENEMY:
            enemys.append(tile)
    return enemys