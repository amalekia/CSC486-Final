from texasholdem.game.game import TexasHoldEm
from texasholdem.gui.text_gui import TextGUI
from texasholdem.game.action_type  import ActionType
from texasholdem.game.player_state import PlayerState
from typing import Tuple
import time
from texasholdem.evaluator.evaluator import evaluate, get_five_card_rank_percentage

def csc486_agent(game: TexasHoldEm) -> Tuple[ActionType, int]:
    player = game.players[game.current_player]
    current_board = game.board
    my_hand = game.hands[game.current_player]
    
    if len(my_hand) + len(current_board) < 5:
        if player.state == PlayerState.TO_CALL:
            return ActionType.CALL, None  # Call preflop if needed
        return ActionType.CHECK, None  # Otherwise, check
    
    # Evaluate hand strength only if we have 5+ cards
    hand_strength = get_five_card_rank_percentage(evaluate(my_hand, current_board))

    # Calculate number of outs and hand potential
    outs, draw_type = get_outs_and_draw_type(my_hand, current_board)  # Custom function
    pot_odds = calculate_pot_odds(game)  # Custom function

    # Risk-adjusted betting
    if hand_strength > 0.7:  # Strong hand (e.g., top pair, better)
        action = ActionType.RAISE
        bet_amount = min(player.chips, game.last_raise * 2)
    
    elif 0.4 < hand_strength <= 0.7 or draw_type in ["flush draw", "open-ended straight"]:
        # If pot odds justify it, call; otherwise, fold
        if outs and pot_odds > (outs / 47):  
            action = ActionType.CALL
            bet_amount = None
        else:
            action = ActionType.FOLD
            bet_amount = None

    else:  # Weak hand
        if game.last_raise == 0:
            action = ActionType.CHECK
            bet_amount = None
        else:
            action = ActionType.FOLD
            bet_amount = None

    return action, bet_amount

from collections import Counter
from itertools import combinations

def get_outs_and_draw_type(hand, board):
    """
    Determines the number of outs and type of draw (if any).
    Returns (number_of_outs, draw_type)
    """
    all_cards = hand + board
    suits = [card.suit for card in all_cards]
    ranks = sorted([card.rank for card in all_cards])
    suit_counts = Counter(suits)
    rank_counts = Counter(ranks)
    outs = 0
    draw_type = "None"
    
    # Check for flush draw
    for suit, count in suit_counts.items():
        if count == 4:  # Missing one card for a flush
            outs += 9  # 9 remaining cards of the same suit
            draw_type = "Flush Draw"
    
    # Check for straight draw (open-ended or gutshot)
    unique_ranks = sorted(set(ranks))
    for i in range(len(unique_ranks) - 3):
        four_card_seq = unique_ranks[i:i+4]
        possible_straight_high = four_card_seq[-1] + 1
        possible_straight_low = four_card_seq[0] - 1
        
        if possible_straight_high <= 14 and possible_straight_high not in unique_ranks:
            outs += 1  # One card missing for a straight
            draw_type = "Open-Ended Straight Draw"
        if possible_straight_low >= 2 and possible_straight_low not in unique_ranks:
            outs += 1
            draw_type = "Open-Ended Straight Draw"
    
    # Gutshot (inside straight draw)
    for four_card_combo in combinations(unique_ranks, 4):
        min_rank, max_rank = min(four_card_combo), max(four_card_combo)
        if max_rank - min_rank == 4:  # One missing card in between
            missing_card = set(range(min_rank, max_rank + 1)) - set(four_card_combo)
            if missing_card:
                outs += 1  # Gutshot has 4 outs
                draw_type = "Gutshot Straight Draw"
    
    return outs, draw_type

def calculate_pot_odds(game):
    """ Calculates the pot odds for decision making. """
    pot = sum(player.chips for player in game.players)
    call_cost = game.last_raise
    return call_cost / pot if pot > 0 else 0


game = TexasHoldEm(buyin=500, big_blind=5, small_blind=2, max_players=2)
gui = TextGUI(game=game)

while game.is_game_running():
    game.start_hand()

    while game.is_hand_running():
        if game.current_player == 1:
            game.take_action(*csc486_agent(game))
            time.sleep(1)
        else:
            gui.run_step()
            time.sleep(1)

    path = game.export_history('./pgns')     # save history
    gui.replay_history(path)      