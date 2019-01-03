import numpy as np
from .game_map import GameMap
from hlt.positionals import Position
from .positionals import Direction
import logging
import random


def map_halite(dimension, game_map, percentile):
    a = np.zeros((dimension, dimension))
    for x in range(dimension):
        for y in range(dimension):
            a[x, y] = game_map[Position(x,y)].halite_amount
    # logging.info("median = {}".format(np.median(a)))
    # logging.info("90% = {}".format(np.percentile(a,90)))
    # logging.info("10% = {}".format(np.percentile(a,10)))
    b = np.piecewise(a, [a < np.percentile(a, percentile), a >= np.percentile(a, percentile)], [lambda a: 0, lambda a: a])
    # b = b / np.median(b)
    return(b)

def map_halite_sq(dimension, game_map, low_halite):
    a = np.zeros((dimension, dimension))
    for x in range(dimension):
        for y in range(dimension):
            a[x, y] = (game_map[Position(x,y)].halite_amount) ** 1.25
            if a[x, y] < low_halite:
                a[x, y] = 0
    return(a)

def dist_to_shipyard(dimension, shipyard, game_map):
    a = np.zeros((dimension, dimension))
    for x in range(dimension):
        for y in range(dimension):
            a[x, y] = (game_map.calculate_distance(Position(x,y), shipyard) + 4)
    return(a)

def dist_to_cells(dimension, ship, game_map):
    a = np.zeros((dimension, dimension))
    for x in range(dimension):
        for y in range(dimension):
            a[x, y] = (game_map.calculate_distance(ship, Position(x,y)) + 6)
    return(a)

def costly_navigate(ship, destination, game_map, prev_pos):
    """
    Returns a singular safe move towards the destination.

    :param ship: The ship to move.
    :param destination: Ending position
    :return: A direction.
    """
    # Still need to account for moving back to stuck position
    costly_dir = []

    for direction in game_map.get_unsafe_moves(ship.position, destination):
        target_pos = ship.position.directional_offset(direction)
        costly_dir.append(Direction.invert(direction))
        if not game_map[target_pos].is_occupied:
            game_map[target_pos].mark_unsafe(ship)
            return direction

    source = game_map.normalize(ship.position)
    destination = game_map.normalize(destination)
    distance = abs(destination - source)

    if distance.x == 0:
        costly_dir = []
        costly_dir.append(Direction.East)
        costly_dir.append(Direction.West)

    if distance.y == 0:
        costly_dir = []
        costly_dir.append(Direction.North)
        costly_dir.append(Direction.South)

    random.shuffle(costly_dir)
    for direction in (costly_dir):
        target_pos = ship.position.directional_offset(direction)
        if not game_map[target_pos].is_occupied and target_pos != prev_pos:
            game_map[target_pos].mark_unsafe(ship)
            return direction

    return Direction.Still


def navigate(ship, destination, game_map, wait, prev_pos):
    move = game_map.naive_navigate(ship, destination)
    if move != (0,0):
        wait = False
        return (move, wait, prev_pos)

    if wait == False:
        wait = True
        return (move, wait, prev_pos)
        # move = game_map.naive_navigate(ship, destination)
        # # logging.info("moving")
        # if move == (0,0):
        #     wait = True
        #     # logging.info("waiting")
        # return (move, wait, prev_pos)
    else:
        # logging.info("costly moving")
        move = costly_navigate(ship, destination, game_map, prev_pos)
        wait = False
        return (move, wait, ship.position.directional_offset(move))
