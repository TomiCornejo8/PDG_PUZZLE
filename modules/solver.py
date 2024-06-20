# External
import numpy as np
from anytree import Node
from collections import deque
import copy

# Internal
from modules import mechanics as M

def hashDungeon(dungeon):
    return hash(tuple(map(tuple, dungeon)))

def getRoute(leaf,parent):
    node = Node(leaf,parent=parent)
    route = deque()
    while node is not None:
        route.append(node.name)
        node = node.parent
    return route

def setState(states,memory,dungeon,parent):
    hashD = hashDungeon(dungeon)
    if not hashD in memory:
        memory.append(hashD)
        states.append(Node(dungeon,parent=parent))  
    return states,memory

def solveGame(dungeon):
    states = deque()
    states.append(Node(dungeon.copy()))
    
    memory = []
    memory.append(hashDungeon(dungeon))
    
    solutions = []
        
    while states:
        currentState = states.popleft()
        currentDungeon = currentState.name

        for move in M.MOVES:
            newDungeon = currentDungeon.copy()
            newDungeon = M.step(newDungeon,move)
            if M.win(newDungeon):
                solutions.append(getRoute(newDungeon,currentState))
            else:
                states,memory = setState(states,memory,newDungeon,currentState)
    return solutions

def nSolutions(dungeon):
    solutions = solveGame(dungeon)
    
    n = len(solutions)
    if n==0: return n,0
    
    minM = len(solutions[0])
    for sol in solutions:
        m = len(sol)
        if m < minM:
            minM = m
    return n,minM

class DungeonWMoves:
    def __init__(self,dungeon,movement):
        self.dungeon = dungeon
        self.move = movement

def setStateWithMoves(states,memory,dungeon,parent):
    hashD = hashDungeon(dungeon.dungeon)
    if not hashD in memory:
        memory.append(hashD)
        states.append(Node(dungeon,parent=parent))  
    return states,memory

def solveGameWithMoves(dungeon):
    states = deque()
    dungeonWMoves = DungeonWMoves(dungeon.copy(),"")
    states.append(Node(dungeonWMoves))
    
    memory = []
    memory.append(hashDungeon(dungeon))
    
    solutions = []
        
    while states:
        currentState = states.popleft()
        currentDungeon = currentState.name

        for move in M.MOVES:
            newDungeon = copy.deepcopy(currentDungeon.dungeon)
            newDungeon,movement = M.stepWithMoves(newDungeon,move)
            newDungeonM = DungeonWMoves(newDungeon,movement)

            if M.win(newDungeon):
                solutions.append(getRoute(newDungeonM,currentState))
            else:
                states,memory = setStateWithMoves(states,memory,newDungeonM,currentState)
    return solutions