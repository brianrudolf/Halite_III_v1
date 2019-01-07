# Halite_III_v1.0
A bot that has been designed for the [Halite III challenge](https://halite.io/). A resource management game, the goal is to build a bot that efficiently navigates the seas collecting halite, a luminous energy resource, while in competition with other players.

## 'The Professor' v1.0
The final bot (v1.0), playfully named 'The Professor', is a culmination of a month's work and has been competing against a rotating pool of bots for over 2 months and 2000+ games. The bot has maintained its place within the top 25% of players, even as the total number of contestants grew from ~500 to over 3000. The bot's performance metrics can be viewed on [Halite's website](https://halite.io/user/?user_id=1460), including visual replays of the games against other players.

This version of the bot utilized custom functions, located in [optimizers.py](https://github.com/brianrudolf/Halite_III_v1/blob/master/hlt/optimizers.py), which:
- Mapped the amount of Halite (the game resource) on the sea floor (map_halite)
- Calculated the distance from each ship to the different locations on the map (dist_to_shipyard, dist_to_cells)
- Optimized the movement of all of the ships in play via two navigation routines (costly_navigate, navigate)

## 'The Professor' v0.8
