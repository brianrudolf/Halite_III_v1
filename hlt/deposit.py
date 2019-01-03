import numpy as np
from .game_map import GameMap
from hlt.positionals import Direction
import random

def side_or_side(ship, destination, game_map):
    possible_moves = Direction.get_all_cardinals()
    direct_move = game_map.get_unsafe_moves(ship.position, destination)
    possible_moves.remove(direct_move[0])
    possible_moves.remove(Direction.invert(direct_move[0]))
    next_step = random.choice(possible_moves)
    if game_map[ship.position.directional_offset(next_step)].is_occupied is True:
        possible_moves.remove(next_step)
        return possible_moves[0]
    else:
        return next_step


def safe_return():
    ships_return.append(ship.id)
    command_queue.append(
        ship.move(
            game_map.naive_navigate(ship,me.shipyard.position)))
