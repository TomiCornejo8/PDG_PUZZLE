import numpy as np
import random
from modules import mechanics as M

def setRandomPlayer(dungeon):
    voidSpace = np.argwhere(dungeon == M.EMPTY)
    player = voidSpace[np.random.choice(len(voidSpace))]
    return player

def getEmpityTiles(dungeon,player,move):
    tile = player.copy()
    voidArray = []
    notAllow = [M.WALL,M.ENEMY]
    while not dungeon[M.lookAhead(tile,move)[0],M.lookAhead(tile,move)[1]] in notAllow:
        tile = M.lookAhead(tile,move)
    if dungeon[tile[0],tile[1]] == M.EMPTY: voidArray.append(tile.copy())
    return voidArray

def checkEdges(dungeon,player):
    moves = M.moves
    for move in moves:
        tile = M.lookAhead(player,move)
        if dungeon[tile[0],tile[1]] == M.WALL : return move
    return M.STAY

def iceSliding(dungeon,player,move):
    while dungeon[M.lookAhead(player,move)[0],M.lookAhead(player,move)[1]] == M.EMPTY:
        dungeon[player[0],player[1]] = M.TRAIL
        player = M.lookAhead(player,move)
    return player

"""
def removeOppositeMove(moves,move):
    if move == LEFT:
        if RIGHT in moves: moves.remove(RIGHT)
    elif move == RIGHT:
        if LEFT in moves: moves.remove(LEFT)
    elif move == UP:
        if  DOWN in moves: moves.remove(DOWN)
    elif move == DOWN:
        if UP in moves: moves.remove(UP)
    return moves
"""

def getDungeon(dungeon,maxMoves,blockFactor = 0.05,enemyFactor = 0.05):
    player = initialPlayer = setRandomPlayer(dungeon)    
    currentMove = M.STAY
    moves = M.moves

    enemys = []

    contMoves = 0
    print(f"MAPA INICIAL\n{dungeon}\n{player}\n")
    while contMoves < maxMoves or checkEdges(dungeon,player) == M.STAY:    
        moves = M.getAllowMoves(dungeon,player)
        print(f"ite {contMoves}\n moves {moves} player {player}\n {dungeon}")
        if len(moves) == 0: break
        currentMove = random.choice(moves)
         
        r1 = random.uniform(0,1)
        if r1 <= blockFactor:
            voidArray = getEmpityTiles(dungeon,player,currentMove)
            if len(voidArray) != 0: 
                block = random.choice(voidArray)
                dungeon[block[0],block[1]] = M.BLOCK
        
        player = iceSliding(dungeon,player,currentMove)
        contMoves += 1 
        if contMoves == maxMoves: break
    
    move = checkEdges(dungeon,player)
    door = M.lookAhead(player,move)
    dungeon[door[0],door[1]] = M.DOOR
    dungeon[initialPlayer[0],initialPlayer[1]] = M.PLAYER
    
    return dungeon,player