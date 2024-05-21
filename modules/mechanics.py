import numpy as np

# Moves
RIGHT = (0,1)
LEFT = (0,-1)
UP = (-1,0)
DOWN = (1,0)
STAY = (0,0)

moves = [RIGHT,LEFT,UP,DOWN]
X = [LEFT,RIGHT]
Y = [UP,DOWN]

EMPTY = 0
WALL = 1
ENEMY = 3
BLOCK = 2
DOOR = 4
PLAYER = 5

TRAIL = 6

# Functions
def lookAhead(player,move):
    return [player[0]+move[0],player[1]+move[1]]

def iceSliding(dungeon,player,move):
    dungeon[player[0],player[1]] = EMPTY

    while dungeon[lookAhead(player,move)[0],lookAhead(player,move)[1]] == EMPTY: 
        player = lookAhead(player,move)
        
    dungeon[player[0],player[1]] = PLAYER

    return dungeon

def killEnemy(dungeon,enemy):
    dungeon[enemy[0],enemy[1]] = EMPTY
    return dungeon

def getAllowMoves(dungeon,player):
    notAllow = [WALL,BLOCK,ENEMY,DOOR]
    newMoves = []
    for move in moves:
        tile = lookAhead(player,move)
        dungeonTile = dungeon[tile[0],tile[1]]
        if not dungeonTile in notAllow:
            newMoves.append(move)

    return newMoves

def getMeleeEnemys(dungeon,player):
    enemys = []
    for move in moves:
        tile = lookAhead(player,move)
        if dungeon[tile[0],tile[1]] == ENEMY:
            enemys.append(tile)
    return enemys