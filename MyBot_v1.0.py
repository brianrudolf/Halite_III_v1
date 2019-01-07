# Import the Halite SDK, which will let you interact with the game.
import hlt
from hlt import constants

# This library contains direction metadata to better interface with the game.
from hlt import positionals
from hlt.positionals import Direction, Position
from hlt.game_map import Player
from hlt.optimizers import map_halite, dist_to_cells, dist_to_shipyard, costly_navigate, navigate, map_halite_sq
from hlt.deposit import side_or_side

import random
import logging

from collections import OrderedDict
import operator

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

np.seterr(divide='ignore', invalid='ignore')    #ignore division errors and invalid operation errors
np.set_printoptions(threshold=np.nan, linewidth=np.nan)


""" <<<Game Begin>>> """

# This game object contains the initial game state.
game = hlt.Game()
# At this point "game" variable is populated with initial map data.
# This is a good place to do computationally expensive start-up pre-processing.
low_halite = constants.MAX_HALITE * 0.1
worthwhile_halite = constants.MAX_HALITE / 5
deposit_level = constants.MAX_HALITE * 0.95

mining = {}
waiting = {}
ship_goal = {}
returning = {}
costly_pos = {}

ships_return = []
mined_positions = []
ship_data = pd.DataFrame(columns = ["mining", "returning", "deploying"])

max_wait = 1

# As soon as you call "ready" function below, the 2 second per turn timer will start.
game.ready("The-Professor-v10b")
logging.info("Successfully created bot! My Player ID is {}.".format(game.my_id))

""" <<<Game Loop>>> """

num_players = max(game.players.keys()) + 1
logging.info("Total players = {}".format(num_players))

while True:
    game.update_frame()
    me = game.me
    game_map = game.game_map

    # hard coded values for total turns based on map size (which is available data)
    dimension = game_map.width
    if dimension == 32:
        total_turns = 400
    elif dimension == 40:
        total_turns = 425
    elif dimension == 48:
        total_turns = 450
    elif dimension == 56:
        total_turns = 475
    else:
        total_turns = 500

    command_queue = []

    dropoffs = me.get_dropoffs()


    dist_shipyard = dist_to_shipyard(dimension, me.shipyard.position, game_map)

    #total ships
    total_ships = 0
    for player in range(num_players):
        total_ships += len(game.players[player].get_ships())

    for ship in me.get_ships():

        if ship.id not in waiting:
            waiting[ship.id] = False
        if ship.id not in ship_goal:
            ship_goal[ship.id] = me.shipyard.position
        if ship.id not in returning:
            returning[ship.id] = False
        if ship.id not in costly_pos:
            costly_pos[ship.id] = me.shipyard.position

        if ship.halite_amount == 0 and ship.id in ships_return:
            ships_return.remove(ship.id)


        # Master branch of what the ship should do


        # Return to shipyard without collisions (follow up)
        if ship.id in ships_return:
            next_step, waiting[ship.id], costly_pos[ship.id] = navigate(ship, me.shipyard.position, game_map, waiting[ship.id], costly_pos[ship.id])
            # previous_pos[ship.id] = ship.position
            command_queue.append(
                ship.move(
                    next_step))



        # Decide whether to return to shipyard
        elif ship.halite_amount > deposit_level :#and ship.id not in ships_return: # or returning[ship.id] == True:
            if game_map[ship.position].halite_amount > low_halite and (ship.position.x, ship.position.y) in mined_positions:
                mined_positions.remove((ship.position.x, ship.position.y))
            # if game_map.calculate_distance(ship.position, me.shipyard.position) > (total_turns - game.turn_number - 5):
            #     attack[ship.id] = 1
            ships_return.append(ship.id)
            ship_goal[ship.id] = me.shipyard.position
            next_step, waiting[ship.id], costly_pos[ship.id] = navigate(ship, me.shipyard.position, game_map, waiting[ship.id], costly_pos[ship.id])
            # previous_pos[ship.id] = ship.position
            command_queue.append(
                ship.move(
                    next_step))

        # return near end of game and go away
        elif game_map.calculate_distance(ship.position, me.shipyard.position) > (total_turns - game.turn_number - 5) and returning[ship.id] == False and ship.halite_amount > deposit_level/3:
            if game_map[ship.position].halite_amount > low_halite and (ship.position.x, ship.position.y) in mined_positions:
                mined_positions.remove((ship.position.x, ship.position.y))
            ships_return.append(ship.id)
            returning[ship.id] = True
            ship_goal[ship.id] = me.shipyard.position
            next_step, waiting[ship.id], costly_pos[ship.id] = navigate(ship, me.shipyard.position, game_map, waiting[ship.id], costly_pos[ship.id])
            # previous_pos[ship.id] = ship.position
            command_queue.append(
                ship.move(
                    next_step))


        #if the ship is at its goal and is it's still worth mining, stay and mine
        elif ship.position == ship_goal[ship.id] and game_map[ship.position].halite_amount > low_halite:
            command_queue.append(ship.stay_still())

        # find the 'best' location to mine and travel to it, adding it as a goal
        # elif game_map[ship.position].halite_amount < low_halite:
        elif ship_goal[ship.id] == me.shipyard.position or (ship.position == ship_goal[ship.id] and game_map[ship.position].halite_amount < low_halite):

            cell_halite = map_halite(dimension, game_map, 10)
            if np.min(cell_halite) < low_halite:
                low_halite = 0.55*np.median(cell_halite)

            dist_halite = dist_to_cells(dimension, ship.position, game_map)

            cell_value = cell_halite / dist_halite + ((cell_halite) / dist_shipyard)
            cell_value = np.nan_to_num(cell_value)


            for n in range(total_ships + len(mined_positions)):
                goal_x, goal_y = np.unravel_index(cell_value.argmax(), cell_value.shape)
                goal_pos = Position(goal_x, goal_y)
                # logging.info(goal_pos)
                if game_map[goal_pos].is_empty and (goal_pos.x, goal_pos.y) not in mined_positions:
                    next_step, waiting[ship.id], costly_pos[ship.id] = navigate(ship, goal_pos, game_map, waiting[ship.id], costly_pos[ship.id])
                    # previous_pos[ship.id] = ship.position
                    command_queue.append(
                        ship.move(
                            next_step))
                    mined_positions.append((goal_pos.x, goal_pos.y))
                    ship_goal[ship.id] = goal_pos
                    # logging.info(" Our target, captain, is {}, and it has {} Halite".format(goal_pos, game_map[goal_pos].halite_amount))
                    # plt.show()

                    break
                cell_value[goal_x, goal_y] = 0

            else:
                logging.info(" This part of the code is broken")

        # if the ship has a destination, continue on to it
        elif ship_goal[ship.id] != me.shipyard.position :#or ship_goal[ship.id] != ship.position:
            next_step, waiting[ship.id], costly_pos[ship.id] = navigate(ship, ship_goal[ship.id], game_map, waiting[ship.id], costly_pos[ship.id])
            # previous_pos[ship.id] = ship.position
            command_queue.append(
                ship.move(
                    next_step))

        else: # stay put and mine
            command_queue.append(ship.stay_still())


    # If there is enough halite to spawn a ship, and the shipyard is unoccupied, spawn a ship based on hard coded max ship limits
    # Ship limits have been experimentally set, and are based on how far into the game we are and how many ships we have already
    if me.halite_amount >= constants.SHIP_COST and not game_map[me.shipyard].is_occupied:
        if game.turn_number <= total_turns*0.5 and len(me.get_ships()) < total_turns/31:
            command_queue.append(me.shipyard.spawn())
        elif game.turn_number <= total_turns*0.6 and len(me.get_ships()) < total_turns/41:
            command_queue.append(me.shipyard.spawn())
        elif game.turn_number <= total_turns*0.7 and len(me.get_ships()) < total_turns/61:
            command_queue.append(me.shipyard.spawn())

    game.end_turn(command_queue)
