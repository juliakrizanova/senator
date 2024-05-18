from senator.utility import *
from senator.game import *

OWNER_LOSS = 0.59
OTHER_LOSS = 0.19
OWNER_TRASHOLD = 0.18 #Set 2.0 to change the model TODO: find better solution than 2.0


votes, initial_utility, is_owner = parse_data(load_data(FILE_PATH))
game = Game(initial_utility, votes, is_owner, OWNER_LOSS, OTHER_LOSS, OWNER_TRASHOLD)
#game = Game(np.zeros((3,17)),votes)

actions_in_steps, final_utility, strategies = game.run_greedy_search()
print(strategies)
#print(final_utility.sum(axis=1))
