
from diplomacy import Game
game = Game()
print('Game created:', game.get_current_phase())
possible_orders = game.get_all_possible_orders()
print('Orders available for:', list(game.powers.keys()))
from diplomacy.utils.export import to_saved_game_format
print('Export works too')
print('ALL GOOD')


