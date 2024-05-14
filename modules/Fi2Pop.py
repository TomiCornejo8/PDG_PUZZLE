import numpy as np
import random

from modules import mechanics as M
from modules import solver

class Individual:
    def __init__(self, solution, fitness):
        self.solution = solution
        self.fitness = fitness

"""
Función de Objetivo de Equilibrio de Obstáculos:
    Esta función mide el balance entre los diferentes tipos de obstáculos (bloques, enemigos, paredes)
    en el mapa.
"""
def fitness_obstacle_balance(dungeon):
    total_cells = dungeon.size
    num_blocks = np.sum(dungeon == M.BLOCK)
    num_enemies = np.sum(dungeon == M.ENEMY)
    return 1 - abs(num_blocks / total_cells - num_enemies / total_cells)

"""
Función de Objetivo de Distancia Promedio al Enemigo:
    Esta función mide la distancia promedio desde cada celda del jugador 
    a los enemigos. Un mapa donde los enemigos están demasiado lejos puede ser menos desafiante.
    Ahora modificada para aumentar la distancia.
"""
def fitness_enemy_distance(dungeon):
    player_pos = np.argwhere(dungeon == M.PLAYER)[0]
    enemy_positions = np.argwhere(dungeon == M.ENEMY)
    if enemy_positions.size == 0:
        return 0
    distances = [np.linalg.norm(player_pos - enemy) for enemy in enemy_positions]
    avg_distance = np.mean(distances)
    return avg_distance

"""
Función de Objetivo de Distancia a la Puerta:
    Esta función mide la distancia desde el jugador a la puerta. 
    Una mayor distancia puede hacer el juego más desafiante.
"""
def fitness_door_distance(dungeon):
    player_pos = np.argwhere(dungeon == M.PLAYER)[0]
    door_pos = np.argwhere(dungeon == M.DOOR)[0]
    distance = np.linalg.norm(player_pos - door_pos)
    return distance

"""
Función de Objetivo de Cercanía de los Bloques:
    Esta función mide la cercanía de los bloques al jugador.
    Una menor distancia puede hacer el juego más desafiante.
"""
def fitness_block_proximity(dungeon):
    player_pos = np.argwhere(dungeon == M.PLAYER)[0]
    block_positions = np.argwhere(dungeon == M.BLOCK)
    if block_positions.size == 0:
        return 0
    distances = [np.linalg.norm(player_pos - block) for block in block_positions]
    avg_distance = np.mean(distances)
    return 1 / avg_distance if avg_distance > 0 else 0

def fitness(dungeon, w1=1.0, w2=0.1, w3=0.8, w4=0.1):
    fitness_ob = fitness_obstacle_balance(dungeon)
    fitness_ed = fitness_enemy_distance(dungeon)
    fitness_dd = fitness_door_distance(dungeon)
    fitness_bp = fitness_block_proximity(dungeon)
    
    # Combining the fitness scores
    combined_score = w1 * fitness_ob + w2 * fitness_ed + w3 * fitness_dd + w4 * fitness_bp
    return combined_score

def repairDoors(child1,child2):
    doors1 = np.argwhere(child1 == M.DOOR)
    doors2 = np.argwhere(child2 == M.DOOR)
    if len(doors1) == 2:
        door2 = random.choice(doors1)
        child1[door2[0],door2[1]] = M.WALL
        child2[door2[0],door2[1]] = M.DOOR
    elif len(doors2) == 2:
        door2 = random.choice(doors2)
        child2[door2[0],door2[1]] = M.WALL
        child1[door2[0],door2[1]] = M.DOOR
    return child1,child2

def repairPlayer(child1,child2):
    players1 = np.argwhere(child1 == M.PLAYER)
    players2 = np.argwhere(child2 == M.PLAYER)
    if len(players1) == 2:
        player2 = random.choice(players1)
        child1[player2[0],player2[1]] = M.EMPTY
        child2[player2[0],player2[1]] = M.PLAYER
    elif len(players2) == 2:
        player2 = random.choice(players2)
        child2[player2[0],player2[1]] = M.EMPTY
        child1[player2[0],player2[1]] = M.PLAYER
    return child1,child2

def crossover(parent1, parent2):
    # Obtener las dimensiones de la matriz
    m, n = parent1.shape
    
    # Generar un número aleatorio entre 0 y 1 para determinar si se utiliza la fila o la columna
    if np.random.rand() < 0.5:  # 50% de probabilidad para filas
        # Seleccionar un índice de fila aleatorio
        row_idx = np.random.randint(m)
        # Cruzamiento por fila
        child1 = np.vstack((parent1[:row_idx], parent2[row_idx:]))
        child2 = np.vstack((parent2[:row_idx], parent1[row_idx:]))
    else:  # 50% de probabilidad para columnas
        # Seleccionar un índice de columna aleatorio
        col_idx = np.random.randint(n)
        # Cruzamiento por columna
        child1 = np.hstack((parent1[:, :col_idx], parent2[:, col_idx:]))
        child2 = np.hstack((parent2[:, :col_idx], parent1[:, col_idx:]))

    child1,child2 = repairDoors(child1,child2)
    child1,child2 = repairPlayer(child1,child2)
    
    return child1, child2

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

def mutate(dungeon):
    tiles = np.argwhere((dungeon != M.WALL) & (dungeon != M.EMPTY))
    tile = random.choice(tiles)

    mutateTile = dungeon[tile[0],tile[1]]

    if mutateTile == M.DOOR:
        walls = getAdjacentWalls(dungeon)
        newDoor = random.choice(walls)
        dungeon[tile[0],tile[1]] = M.WALL
        dungeon[newDoor[0],newDoor[1]] = M.DOOR
    elif np.random.rand() <= 0.5 and mutateTile in [M.ENEMY,M.BLOCK]:
        if mutateTile == M.ENEMY:
            dungeon[tile[0],tile[1]] = M.BLOCK
        else:
            dungeon[tile[0],tile[1]] = M.ENEMY
    else:
        emptys = np.argwhere(dungeon == M.EMPTY)
        newEntity = random.choice(emptys)
        dungeon[tile[0],tile[1]] = M.EMPTY
        if mutateTile == M.PLAYER:
            dungeon[newEntity[0],newEntity[1]] = M.PLAYER
        else:
            dungeon[newEntity[0],newEntity[1]] = random.choice([M.ENEMY,M.BLOCK])
    return dungeon

def tournamentSelection(feapop, infpop, mutation = False):
    def tournament(population):
        """Realiza la selección por torneo en una población dada."""
        k = 3  # Tamaño del torneo, puedes ajustarlo según necesites
        tournament_contestants = random.sample(population, k)
        best = max(tournament_contestants, key=lambda x: x.fitness)
        return best

    # Selecciona un padre de cada población usando la selección por torneo
    parent1 = tournament(feapop.tolist())

    if mutation: return parent1.solution
    
    # Remueve parent1 de la población para evitar seleccionar el mismo individuo
    feapop = np.delete(feapop, np.where(feapop == parent1))
    
    parent2 = tournament(np.concatenate((feapop, infpop), axis=0).tolist())
    
    # Vuelve a añadir parent1 a feapop
    feapop = np.append(feapop,parent1)
    
    return parent1.solution, parent2.solution

def isFeasible(dungeon):
    solutions = solver.solve(dungeon.copy())
    if len(solutions) == 0: return False
    return True

def initialPopulation(population):
    feapop = []
    infpop = []
    for dungeon in population:
        if isFeasible(dungeon):
            feapop.append(Individual(dungeon,fitness(dungeon)))
        else:
            infpop.append(Individual(dungeon,fitness(dungeon)))
    return np.array(feapop),np.array(infpop)

def elitism(population,nPop):
    population = sorted(population, key=lambda x: x.fitness, reverse=True)
    population = population[:nPop]
    return np.array(population)

def FI2Pop(population,maxIter, nPop,mutationFactor):
    feapop, infpop = initialPopulation(population)
    
    for it in range(maxIter):
        feaoffspring = []
        infoffspring = []
        
        while len(feaoffspring) < nPop or len(infoffspring) < nPop:
            offspring = []
            
            if np.random.rand()  < mutationFactor:
                dungeon = tournamentSelection(feapop, infpop,True)
                dungeon = mutate(dungeon.copy())
                offspring.append(dungeon)
            else:
                parent1,parent2 = tournamentSelection(feapop, infpop)
                child1,child2 = crossover(parent1,parent2)
                if np.random.rand()  < mutationFactor:
                    child1 = mutate(child1.copy())
                    child2 = mutate(child2.copy())
                offspring.append(child1)
                offspring.append(child2)
            
            for dungeon in offspring:
                if isFeasible(dungeon) and len(feaoffspring) < nPop:
                    feaoffspring.append(Individual(dungeon,fitness(dungeon)))
                else:
                    if len(infoffspring) < nPop:
                        infoffspring.append(Individual(dungeon,fitness(dungeon)))
        
        feapop = np.concatenate((feapop, np.array(feaoffspring)), axis=0)
        infpop = np.concatenate((infpop, np.array(infoffspring)), axis=0)

        feapop = elitism(feapop,nPop)
        infpop = elitism(infpop,nPop)

        print(f"iter = {it} BestFitness = {feapop[0].fitness}")
    
    return feapop[0].solution,feapop[0].fitness