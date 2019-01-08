# Initial imports are specific to the Halite game system
# Import the Halite SDK, which will let you interact with the game.
import hlt
# This library contains constant values.
from hlt import constants

# This library contains direction metadata to better interface with the game.
from hlt.positionals import Direction
from hlt.positionals import Position
from hlt import positionals
# from hlt.positionals import convert
# from hlt.positionals import invert

import random

# Logging allows you to save messages for yourself. This is required because the regular STDOUT
#   (print statements) are reserved for the engine-bot communication.
import logging

from collections import OrderedDict
import operator

""" <<<Game Begin>>> """

# This game object contains the initial game state.
game = hlt.Game()
# At this point "game" variable is populated with initial map data.
# This is a good place to do computationally expensive start-up pre-processing.
low_halite = constants.MAX_HALITE / 10
worthwhile_halite = constants.MAX_HALITE / 5
deposit_level = constants.MAX_HALITE * 0.9
ships_return = []
# arrays to define the reference coordinates around a position to cover a 3x3 square centered on said position
one_sq = ((-1, -1), (0, -1), (1, -1), (-1, 0), (1, 1), (0, 1), (-1, 1), (0, -1))
two_sq = ((-2, -2), (-1, -2), (0, -2), (1, -2), (2, -2),
          (-2, -1), (2, -1),
          (-2, 0), (2, 0),
          (-2, 1), (2, 1),
          (-2, 2), (-1, 2), (0, 2), (1, 2), (2, 2)
          )


# As soon as you call "ready" function below, the 2 second per turn timer will start.
game.ready("The-Professor-v8")

# Now that your bot is initialized, save a message to yourself in the log file with some important information.
#   Here, you log here your id, which you can always fetch from the game object by using my_id.
logging.info("Successfully created bot! My Player ID is {}.".format(game.my_id))

""" <<<Game Loop>>> """

while True:
    # This loop handles each turn of the game. The game object changes every turn, and you refresh that state by
    #   running update_frame().
    game.update_frame()
    # You extract player metadata and the updated map metadata here for convenience.
    me = game.me
    game_map = game.game_map
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
    # A command queue holds all the commands you will run this turn. You build this list up and submit it at the
    #   end of the turn.
    command_queue = []

    dropoffs = me.get_dropoffs()

    for ship in me.get_ships():
        # For each of your ships, move randomly if the ship is on a low halite location or the ship is full.
        #   Else, collect halite.        

        if ship.halite_amount == 0 and ship.id in ships_return:
            ships_return.remove(ship.id)

        # Master branch of what the ship should do

        # Return to shipyard without collisions
        if ship.id in ships_return:
            next_step = game_map.get_unsafe_moves(ship.position, me.shipyard.position)
            next_step = next_step[0]
            next_step_pos = ship.position.directional_offset(next_step)

            if game_map[next_step_pos].is_occupied is True:
                #remove occupied forward position, and back position
                poss_dir = [(0, -1), (0, 1), (1, 0), (-1, 0)]
                poss_dir.remove(next_step)
                poss_dir.remove(Direction.invert(next_step))

                next_step = random.choice(poss_dir) #pick a direction and check if it's occupied
                next_step_pos = ship.position.directional_offset(next_step)

                if game_map[next_step_pos].is_occupied is True:
                    poss_dir.remove(next_step)
                    if game_map[ship.position.directional_offset(poss_dir[0])].is_occupied is True: #if it's occupied, all squares are occupied, hit the lowest ship
                        near_ships = {'n': game_map[ship.position.directional_offset((0, -1))].halite_amount,
                                        'e': game_map[ship.position.directional_offset((0, 1))].halite_amount,
                                            's': game_map[ship.position.directional_offset((1, 0))].halite_amount,
                                                'w': game_map[ship.position.directional_offset((-1, 0))].halite_amount
                                                }
                        low_ship = min(near_ships.items(), key=operator.itemgetter(1))[0]
                        command_queue.append(
                            ship.move(
                                low_ship))
                    else:
                        command_queue.append(
                            ship.move(
                                poss_dir[0]))

                else:
                    command_queue.append(
                        ship.move(
                            next_step))
            else:
                command_queue.append(
                    ship.move(
                        game_map.naive_navigate(ship,me.shipyard.position)))

        elif ship.halite_amount > deposit_level and ship.id not in ships_return:
            ships_return.append(ship.id)
            command_queue.append(
                ship.move(
                    game_map.naive_navigate(ship,me.shipyard.position)))

        elif game_map.calculate_distance(ship.position, me.shipyard.position) > total_turns - game.turn_number + 3:
            ships_return.append(ship.id)
            command_queue.append(
                ship.move(
                    game_map.naive_navigate(ship,me.shipyard.position)))

        elif game_map[ship.position].halite_amount < low_halite:
            # move, look for worthwhile Halite within 1-sq distance
            goal_pos = Position(ship.position.x,ship.position.y)
            for coord in one_sq:
                next_pos =  game_map.normalize(Position(ship.position.x + coord[0], ship.position.y + coord[1]))
                if game_map[next_pos].halite_amount > game_map[goal_pos].halite_amount:
                    goal_pos = next_pos

            if game_map[goal_pos].halite_amount > worthwhile_halite:
                #the first 'circle' around the ship has enough Halite to be worthwhile
                command_queue.append(
                    ship.move(game_map.naive_navigate(ship, goal_pos)))

            else: #otherwise look within 2-sq distance for largest deposit
                for coord in two_sq:
                    next_pos =  game_map.normalize(Position(ship.position.x + coord[0], ship.position.y + coord[1]))
                    if game_map[next_pos].halite_amount > game_map[goal_pos].halite_amount:
                            goal_pos = next_pos
                #head towards largest deposit in 2-square sight
                command_queue.append(
                    ship.move(game_map.naive_navigate(ship, goal_pos)))
        # mine
        else:
            command_queue.append(ship.stay_still())

    # If the game is in the first 200 turns and you have enough halite, spawn a ship.
    # Don't spawn a ship if you currently have a ship at port, though - the ships will collide.
    if me.halite_amount >= constants.SHIP_COST and not game_map[me.shipyard].is_occupied:
        if game.turn_number <= total_turns*0.5 and len(me.get_ships()) < total_turns/41:
            command_queue.append(me.shipyard.spawn())
        elif game.turn_number <= total_turns*0.7 and len(me.get_ships()) < total_turns/50:
            command_queue.append(me.shipyard.spawn())
        elif game.turn_number <= total_turns*0.85 and len(me.get_ships()) < total_turns/60:
            command_queue.append(me.shipyard.spawn())
    # Send your moves back to the game environment, ending this turn.
    game.end_turn(command_queue)
