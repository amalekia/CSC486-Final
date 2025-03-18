import json
import csv
from texasholdem.game.game import TexasHoldEm
from texasholdem.gui.text_gui import TextGUI
from texasholdem.game.action_type import ActionType
from texasholdem.game.player_state import PlayerState
from typing import Tuple
import time
from texasholdem.evaluator.evaluator import evaluate, get_five_card_rank_percentage
from collections import Counter
from itertools import combinations


def read_attributes(file_path='pad.json'):
    """
    Reads a JSON file and returns a dictionary containing the values
    of attributes P, A, and D.
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            attributes = {
                'P': data.get('P'),
                'A': data.get('A'),
                'D': data.get('D')
            }
            return attributes
    except (FileNotFoundError, json.JSONDecodeError, AttributeError):
        return {}



def logDecision(data):
    with open('decision_log.txt', mode='a', newline='') as file:
        file.write(data + '\n')


def write_to_csv(data: dict, file_path='data.csv'):
    """Write the provided data to a CSV file."""
    fieldnames = ['P', 'A', 'D', 'current_hand', 'original_bet_size', 'augmented_bet_size']
    try:
        # Check if the file exists to append or write new
        file_exists = False
        try:
            with open(file_path, 'r'):
                file_exists = True
        except FileNotFoundError:
            pass

        with open(file_path, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()  # Write header if the file is new

            writer.writerow(data)
    except Exception as e:
        print(f"Error writing to CSV: {e}")


def kelly_criterion(p: float, odds: float) -> float:
    """Calculate bet size using Kelly Criterion."""
    q = 1 - p
    return (p * odds - q) / odds if odds > 0 else 0


def csc486_agent(game: TexasHoldEm) -> Tuple[ActionType, int]:
    player = game.players[game.current_player]
    current_board = game.board
    my_hand = game.hands[game.current_player]

    # Read opponent's PAD values
    opponent_attributes = read_attributes('pad.json')
    opponent_pleasure = opponent_attributes.get('P', 0)  # Default to 0 if not available
    opponent_dominance = opponent_attributes.get('D', 0)  # Default to 0 if not available
    # opponent_

    # Evaluate hand strength only if we have 5+ cards
    if len(my_hand) + len(current_board) < 5:
        if player.state == PlayerState.TO_CALL:
            return ActionType.CALL, None  # Call preflop if needed
        return ActionType.CHECK, None  # Otherwise, check

    hand_strength = get_five_card_rank_percentage(evaluate(my_hand, current_board))

    # Calculate number of outs and hand potential
    outs, draw_type = get_outs_and_draw_type(my_hand, current_board)  # Custom function
    pot_odds = calculate_pot_odds(game)  # Custom function

    # Calculate original bet size
    original_bet_size = 0


    if hand_strength > 0.4:  # Strong hand (e.g., top pair, better)
        # original_bet_size = min(player.chips, game.last_raise * 2)
        original_bet_size = kelly_criterion(hand_strength, 1)
        action = ActionType.RAISE
        bet_amount = original_bet_size

        logDecision(f"Strong hand, hand_strength {hand_strength}, potOdds {pot_odds}, obs {original_bet_size}")
    elif 0.2 < hand_strength <= 0.4 or draw_type in ["flush draw", "open-ended straight"]:
        if outs and pot_odds > (outs / 47):
            action = ActionType.CALL
            bet_amount = None
            logDecision(f"Call, hand_strength {hand_strength}, outs {outs}, pot_odds {pot_odds}")
        else:
            action = ActionType.FOLD
            bet_amount = None
            logDecision(f"Fold, hand_strength {hand_strength}, outs {outs}, pot_odds {pot_odds}")

    else:  # Weak hand
        if game.last_raise == 0:
            action = ActionType.CHECK
            bet_amount = None
            logDecision(f"Check, hand_strength {hand_strength}, obs {original_bet_size}")
        else:
            action = ActionType.FOLD
            bet_amount = None
            logDecision(f"Fold, hand_strength {hand_strength}, obs {original_bet_size}")

    # Adjust the bet size based on the opponent's PAD values
    bet_multiplier = 1.0  # Default multiplier for bet size
    if opponent_pleasure > 0.7:
        bet_multiplier *= 0.8  # Decrease bet size if opponent is feeling pleased
    if opponent_dominance < 0.3:
        bet_multiplier *= 1.2  # Increase bet size if opponent is less dominant

    augmented_bet_size = bet_amount * bet_multiplier if bet_amount else None

    # Prepare data for CSV
    current_hand = [f'{card.rank}{card.suit}' for card in my_hand]  # Format the hand as [RankSuit]
    data = {
        'P': opponent_pleasure,
        'A': opponent_attributes.get('A', 0),  # Default to 0 if not available
        'D': opponent_dominance,
        'current_hand': ', '.join(current_hand),
        'original_bet_size': original_bet_size,
        'augmented_bet_size': augmented_bet_size if augmented_bet_size else 'N/A'
    }

    # Write data to CSV
    write_to_csv(data)

    return action, augmented_bet_size


# Helper functions for calculating outs and pot odds
def get_outs_and_draw_type(hand, board):
    """ Determines the number of outs and type of draw (if any). """
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
    # pot = sum(player.chips for player in game.players)
    pot = game.pots[0].amount
    call_cost = game.last_raise
    return pot / call_cost if pot > 0 else 0



# clear log file
open('decision_log.txt', 'w').close()

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
    # input("Press Enter to continue...")
    time.sleep(1000)



