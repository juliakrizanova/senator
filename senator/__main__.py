from senator.utility import *
from senator.game import *

votes, initial_utility = parse_data(load_data(FILE_PATH))

game = Game(initial_utility,votes)

game.run_greedy_search()