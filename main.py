from texasholdem.game.game import TexasHoldEm
from texasholdem.gui.text_gui import TextGUI
from texasholdem.game.action_type  import ActionType
from texasholdem.game.player_state import PlayerState
<<<<<<< HEAD
# from texasholdem.game.card import Card
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

=======
from typing import Tuple
import time
from texasholdem.evaluator.evaluator import evaluate, get_five_card_rank_percentage
from collections import Counter
from itertools import combinations


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
    outs, draw_type = get_outs_and_draw_type(my_hand, current_board)
    pot_odds = calculate_pot_odds(game)

    # Kelly Criterion implementation for bet sizing
    if hand_strength > 0.7:  # Strong hand (e.g., top pair, better)
        win_probability = hand_strength
        # Kelly formula: f* = (p(b+1) - 1)/b where p is win probability, b is odds received (pot/bet)
        pot_amount = game.pots[0].amount  # Access the amount field of the Pot object
        odds = pot_amount / max(1, game.last_raise)  # Avoid division by zero
        kelly_fraction = max(0, (win_probability * (odds + 1) - 1) / odds)

        # Apply a fraction of Kelly (half Kelly) to be more conservative
        kelly_fraction = kelly_fraction * 0.5

        # Calculate bet amount based on Kelly Criterion
        kelly_bet = int(player.chips * kelly_fraction)

        # Ensure bet meets minimum raise requirements and doesn't exceed chips
        min_raise = max(game.last_raise * 2, game.big_blind)
        bet_amount = max(min_raise, min(kelly_bet, player.chips))

        return ActionType.RAISE, bet_amount

    elif 0.4 < hand_strength <= 0.7 or draw_type in ["flush draw", "open-ended straight"]:
        # If pot odds justify it, call; otherwise, fold
        # Convert outs to approximate win probability
        win_probability = outs / 47 if outs else hand_strength

        if win_probability > pot_odds:
            # Consider small raise if we have decent implied odds
            if win_probability > pot_odds * 1.5 and player.chips > game.last_raise * 3:
                # Use fractional Kelly for semi-bluff raises
                kelly_fraction = max(0, (win_probability * 2 - 1))
                kelly_bet = int(player.chips * kelly_fraction * 0.25)  # Quarter Kelly for draws

                # Ensure bet meets minimum requirements
                min_raise = max(game.last_raise * 2, game.big_blind)
                if kelly_bet >= min_raise:
                    return ActionType.RAISE, min(kelly_bet, player.chips)

            return ActionType.CALL, None
        else:
            if game.last_raise == 0:
                return ActionType.CHECK, None
            else:
                return ActionType.FOLD, None

    else:  # Weak hand
        if game.last_raise == 0:
            # Occasionally bluff with a very poor hand (5% of the time)
            import random
            if random.random() < 0.05:
                # Small bluff bet - using a tiny fraction of chips
                bluff_amount = min(player.chips, game.big_blind * 4)
                return ActionType.RAISE, bluff_amount
            return ActionType.CHECK, None
        else:
            return ActionType.FOLD, None


>>>>>>> 04d7c2616cda8c47c8620aa9255a99bfca919a3f
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
        four_card_seq = unique_ranks[i:i + 4]
        possible_straight_high = four_card_seq[-1] + 1
        possible_straight_low = four_card_seq[0] - 1
        
        if possible_straight_high <= 14 and possible_straight_high not in unique_ranks:
            outs += 4  # Four cards can complete the straight
            draw_type = "Open-Ended Straight Draw"
        if possible_straight_low >= 2 and possible_straight_low not in unique_ranks:
            outs += 4
            draw_type = "Open-Ended Straight Draw"
    
    # Gutshot (inside straight draw)
    for four_card_combo in combinations(unique_ranks, 4):
        min_rank, max_rank = min(four_card_combo), max(four_card_combo)
        if max_rank - min_rank == 4:  # One missing card in between
            missing_card = set(range(min_rank, max_rank + 1)) - set(four_card_combo)
            if missing_card:
                outs += 4  # Gutshot has 4 outs
                draw_type = "Gutshot Straight Draw"
    
    return outs, draw_type


def calculate_pot_odds(game):
    """ Calculates the pot odds for decision making. """
    pot_amount = game.pots[0].amount  # Access the amount field of the Pot object
    call_cost = game.last_raise
    return call_cost / (pot_amount + call_cost) if (pot_amount + call_cost) > 0 else 0


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

<<<<<<< HEAD
    path = game.export_history('./pgns')     # save history
    gui.replay_history(path)    
=======
    path = game.export_history('./pgns')  # save history
    gui.replay_history(path)
>>>>>>> 04d7c2616cda8c47c8620aa9255a99bfca919a3f
