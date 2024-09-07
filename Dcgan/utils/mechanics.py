import numpy as np

# Moves
RIGHT = (0,1)
LEFT = (0,-1)
UP = (-1,0)
DOWN = (1,0)
STAY = (0,0)

MOVES = [RIGHT,LEFT,UP,DOWN]

# Entitys
EMPTY = 0
WALL = 1
ENEMY = 3
BLOCK = 2
DOOR = 4
PLAYER = 5

# Functions
def getPlayer(dungeon):
    return np.where(dungeon == PLAYER)

def lookAhead(dungeon,move):
    player = getPlayer(dungeon)
    tile = [player[0]+move[0],player[1]+move[1]]
    entity = dungeon[tile[0],tile[1]]
    return tile,entity

def iceSliding(dungeon,move):
    while True: 
        tile,entity = lookAhead(dungeon,move)
        if entity == EMPTY:
            player = getPlayer(dungeon)
            dungeon[player[0],player[1]] = EMPTY
            dungeon[tile[0],tile[1]] = PLAYER
        else:
            break
    return dungeon

def killEnemy(dungeon,enemy):
    dungeon[enemy[0],enemy[1]] = EMPTY
    return dungeon

def step(dungeon,move):
    tile,entity = lookAhead(dungeon,move)
    if entity == EMPTY:
        dungeon = iceSliding(dungeon,move)
    elif entity == ENEMY:
        dungeon = killEnemy(dungeon,tile)
    return dungeon

def win(dungeon):
    lenEnemys = len(np.argwhere(dungeon == ENEMY))
    if lenEnemys > 0: return False
    
    for move in MOVES:
        _,entity = lookAhead(dungeon,move)
        if entity == DOOR: return True

    return False

def getMove(move):
    if move == RIGHT: return "RIGHT"
    elif move == LEFT: return "LEFT"
    elif move == UP: return "UP"
    else: return "DOWN"

def stepWithMoves(dungeon,move):
    text = ""
    tile,entity = lookAhead(dungeon,move)
    if entity == EMPTY:
        dungeon = iceSliding(dungeon,move)
        text = f"Move {getMove(move)}"
    elif entity == ENEMY:
        dungeon = killEnemy(dungeon,tile)
        text = f"Kill {getMove(move)}"
    return dungeon,text