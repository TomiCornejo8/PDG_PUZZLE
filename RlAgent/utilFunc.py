from collections import deque
import numpy as np

# Moves
RIGHT = (0, 1)
LEFT = (0, -1)
UP = (-1, 0)
DOWN = (1, 0)
STAY = (0, 0)

MOVES = [RIGHT, LEFT, UP, DOWN]

# Entitys
EMPTY = 0
WALL = 1
ENEMY = 3
BLOCK = 2
DOOR = 4
PLAYER = 5

# Función para obtener la posición del jugador
def getPlayer(dungeon):
    player_position = np.column_stack(np.where(dungeon == PLAYER))
    if len(player_position) > 0:
        return player_position[0][0], player_position[0][1]
    return None

# Función para mirar el siguiente tile
def lookAhead(dungeon, move):
    player = getPlayer(dungeon)
    if player:
        tile = [player[0] + move[0], player[1] + move[1]]
        entity = dungeon[tile[0], tile[1]]
        return tile, entity
    return None, None

# Función para deslizarse sobre el hielo
def iceSliding(dungeon, move):
    while True:
        tile, entity = lookAhead(dungeon, move)
        if entity == EMPTY:
            player = getPlayer(dungeon)
            dungeon[player[0], player[1]] = EMPTY
            dungeon[tile[0], tile[1]] = PLAYER
        else:
            break
    return dungeon

# Función para eliminar un enemigo
def killEnemy(dungeon, enemy):
    dungeon[enemy[0], enemy[1]] = EMPTY
    return dungeon

# Implementación de BFS (búsqueda en anchura) para encontrar un camino
def bfs(dungeon, start, goal):
    queue = deque([start])
    visited = set()
    visited.add(tuple(start))
    
    while queue:
        current = queue.popleft()
        if current == tuple(goal):
            return True
        for move in MOVES:
            next_tile = (current[0] + move[0], current[1] + move[1])
            if (0 <= next_tile[0] < dungeon.shape[0] and
                0 <= next_tile[1] < dungeon.shape[1] and
                next_tile not in visited and
                dungeon[next_tile[0], next_tile[1]] != WALL and
                dungeon[next_tile[0], next_tile[1]] != BLOCK):
                queue.append(next_tile)
                visited.add(next_tile)
    
    return False

# Función que verifica si el enemigo está bloqueando el camino a la puerta
def is_enemy_blocking_path(dungeon, enemy_position):
    door_position = np.argwhere(dungeon == DOOR)
    if len(door_position) == 0:
        return False  # No hay puerta en el mapa
    player_pos = getPlayer(dungeon)
    
    # Si el enemigo está en el camino del jugador a la puerta, bloquea
    return bfs(dungeon, player_pos, door_position[0])

# Función que verifica si el jugador ha ganado
def win(dungeon):
    lenEnemys = len(np.argwhere(dungeon == ENEMY))
    if lenEnemys > 0: return False

    for move in MOVES:
        _, entity = lookAhead(dungeon, move)
        if entity == DOOR: return True

    return False

# Función auxiliar para contar enemigos
def getnEnemys(dungeon):
    return len(np.argwhere(dungeon == ENEMY))

# Función que verifica si la puerta está cerca
def isDoorNearBy(dungeon):
    for move in MOVES:
        _, entity = lookAhead(dungeon, move)
        if entity == DOOR: return True
    return False
