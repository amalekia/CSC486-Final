from texasholdem.game.game import TexasHoldEm
from texasholdem.gui.text_gui import TextGUI
from texasholdem.game.action_type import ActionType
from texasholdem.game.player_state import PlayerState
from typing import Tuple, Dict, List, Optional
import time
import csv
import os
from texasholdem.evaluator.evaluator import evaluate, get_five_card_rank_percentage
from collections import Counter, deque
from itertools import combinations

class PADReader:
    """
    Class to read and process PAD (Pleasure-Arousal-Dominance) values from a CSV file.
    The CSV is expected to have columns for pleasure, arousal, and dominance in that order.
    """
    def __init__(self, csv_path="pad_values.csv", log_path="bot_decisions.csv"):
        self.csv_path = csv_path
        self.log_path = log_path
        self.last_modified_time = 0
        self.pad_history = {
            "pleasure": deque(maxlen=50),
            "arousal": deque(maxlen=50),
            "dominance": deque(maxlen=50)
        }
        self.baseline = {"pleasure": 0, "arousal": 0, "dominance": 0}
        self.is_calibrated = False

        # Game state correlation
        self.hand_strength_history = []
        self.bet_size_history = []
        self.action_history = []

        # Create empty CSV if it doesn't exist
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['pleasure', 'arousal', 'dominance'])

        # Create decision log CSV with headers
        with open(self.log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp',
                'pleasure',
                'arousal',
                'dominance',
                'hand_cards',
                'board_cards',
                'game_phase',
                'hand_strength',
                'adjusted_win_probability',
                'original_bet',
                'pad_adjusted_bet',
                'action_taken',
                'pot_size',
                'opponent_likely_bluffing',
                'opponent_strong_hand',
                'opponent_uncertain'
            ])

    def log_decision(self, current_pad, hand, board, game_phase, hand_strength,
                     adjusted_win_prob, original_bet, adjusted_bet, action,
                     pot_size, opponent_insights):
        """
        Log the bot's decision and all relevant context to CSV.
        """
        try:
            # Format hand and board cards as strings
            hand_str = '-'.join([f"{card.rank}{card.suit[0]}" for card in hand]) if hand else "Unknown"
            board_str = '-'.join([f"{card.rank}{card.suit[0]}" for card in board]) if board else "Empty"

            # Get current timestamp
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

            # Prepare data row
            row = [
                timestamp,
                current_pad.get("pleasure", 0),
                current_pad.get("arousal", 0),
                current_pad.get("dominance", 0),
                hand_str,
                board_str,
                game_phase,
                hand_strength,
                adjusted_win_prob,
                original_bet if original_bet is not None else 0,
                adjusted_bet if adjusted_bet is not None else 0,
                action.name if hasattr(action, 'name') else str(action),
                pot_size,
                opponent_insights.get("likely_bluffing", False),
                opponent_insights.get("strong_hand", False),
                opponent_insights.get("uncertain", False)
            ]

            # Append to log file
            with open(self.log_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)

        except Exception as e:
            print(f"Error logging decision: {e}")
            # Continue execution even if logging fails
    def calibrate(self, num_samples=30):
        """
        Calibrate baseline PAD values using the first num_samples readings.
        """
        print(f"Calibrating PAD baseline using {num_samples} samples...")
        pad_values = {"pleasure": [], "arousal": [], "dominance": []}

        # Collect samples
        for _ in range(num_samples):
            current_pad = self.get_current_pad()
            if current_pad:
                for key in pad_values:
                    pad_values[key].append(current_pad[key])
            time.sleep(0.1)

        # Calculate baseline as average of collected values
        for key in self.baseline:
            if pad_values[key]:
                self.baseline[key] = sum(pad_values[key]) / len(pad_values[key])

        self.is_calibrated = True
        print("PAD baseline calibration complete.")
        print(f"Baseline values: {self.baseline}")

    def get_current_pad(self) -> Optional[Dict[str, float]]:
        """
        Read the latest PAD values from the CSV file.
        Returns a dictionary with pleasure, arousal, and dominance values.
        """
        try:
            # Check if file has been modified
            current_mtime = os.path.getmtime(self.csv_path)
            if current_mtime <= self.last_modified_time:
                # No new data
                if self.pad_history["pleasure"]:  # Return most recent if available
                    return {
                        "pleasure": self.pad_history["pleasure"][-1],
                        "arousal": self.pad_history["arousal"][-1],
                        "dominance": self.pad_history["dominance"][-1]
                    }
                return None

            self.last_modified_time = current_mtime

            # Read the latest line from CSV
            with open(self.csv_path, 'r') as f:
                lines = f.readlines()
                if len(lines) <= 1:  # Only header or empty
                    return None

                # Get last line with data
                last_line = lines[-1].strip()
                if not last_line:
                    return None

                # Parse CSV line
                values = last_line.split(',')
                if len(values) >= 3:
                    pad_values = {
                        "pleasure": float(values[0]),
                        "arousal": float(values[1]),
                        "dominance": float(values[2])
                    }

                    # Update history
                    for key, value in pad_values.items():
                        self.pad_history[key].append(value)

                    return pad_values

            return None
        except Exception as e:
            print(f"Error reading PAD values: {e}")
            return None

    def get_normalized_pad(self) -> Dict[str, float]:
        """
        Get normalized PAD values relative to the baseline.
        Returns values between -1 and 1.
        """
        current = self.get_current_pad()
        if not current or not self.is_calibrated:
            return {"pleasure": 0, "arousal": 0, "dominance": 0}

        normalized = {}
        for key in current:
            # Normalize against baseline with clamping
            delta = current[key] - self.baseline[key]
            # Scale factor can be adjusted based on typical value ranges
            scale_factor = 1.0
            normalized[key] = max(min(delta / scale_factor, 1.0), -1.0)

        return normalized

    def detect_emotional_patterns(self, game_phase, hand_strength=None, last_action=None):
        """
        Analyze PAD patterns based on the current game state.
        Returns insights about opponent's likely hand strength.
        """
        normalized_pad = self.get_normalized_pad()

        # Store current game context with PAD values
        if hand_strength is not None:
            self.hand_strength_history.append((hand_strength, normalized_pad))

        if last_action is not None:
            self.action_history.append((last_action, normalized_pad))

        # Analyze current emotional state
        # High arousal + low pleasure often indicates bluffing or weak hand
        # High pleasure + high dominance often indicates strong hand
        # These thresholds can be fine-tuned based on observation

        insights = {}

        # Detect bluffing patterns
        if normalized_pad["arousal"] > 0.4 and normalized_pad["pleasure"] < -0.2:
            insights["likely_bluffing"] = True
            insights["confidence"] = min(0.7, normalized_pad["arousal"] - normalized_pad["pleasure"])
        else:
            insights["likely_bluffing"] = False

        # Detect strong hand patterns
        if normalized_pad["pleasure"] > 0.3 and normalized_pad["dominance"] > 0.3:
            insights["strong_hand"] = True
            insights["confidence"] = min(0.8, (normalized_pad["pleasure"] + normalized_pad["dominance"]) / 2)
        else:
            insights["strong_hand"] = False

        # Detect uncertainty
        if abs(normalized_pad["arousal"]) > 0.5 and abs(normalized_pad["dominance"]) < 0.2:
            insights["uncertain"] = True
        else:
            insights["uncertain"] = False

        return insights


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

def determine_game_phase(game):
    """Determine the current phase of the game based on visible cards."""
    if not game.board:
        return "preflop"
    elif len(game.board) == 3:
        return "flop"
    elif len(game.board) == 4:
        return "turn"
    else:
        return "river"

def evaluate_preflop_hand(hand):
    """
    Improved preflop hand evaluator based on Chen Formula and poker fundamentals.
    Returns a score between 0 and 1 representing hand strength.
    """
    if len(hand) != 2:
        return 0.3  # Default value if we don't have exactly 2 cards

    # Extract card information
    card1, card2 = hand
    rank1, rank2 = card1.rank, card2.rank
    suited = card1.suit == card2.suit

    # Sort ranks for easier processing
    high_rank, low_rank = max(rank1, rank2), min(rank1, rank2)
    gap = high_rank - low_rank

    # Base score calculation (simplified Chen formula)
    score = 0

    # Highest card
    if high_rank == 14:  # Ace
        score += 10
    elif high_rank == 13:  # King
        score += 8
    elif high_rank == 12:  # Queen
        score += 7
    elif high_rank == 11:  # Jack
        score += 6
    else:
        score += high_rank / 2

    # Pair bonus
    if high_rank == low_rank:
        if high_rank >= 10:  # High pairs
            score = max(score, high_rank + 9)  # PP gets big boost
        else:  # Medium/low pairs
            score = max(score, high_rank + 6)

    # Suited bonus
    if suited:
        score += 2

    # Connector bonus (closeness bonus)
    if gap == 0:
        pass  # Already handled in pairs
    elif gap == 1:
        score += 1  # Connector
    elif gap == 2:
        score += 0.5  # One-gap
    elif gap == 3:
        score += 0.3  # Two-gap

    # Gap penalty for unconnected cards
    if gap > 1 and high_rank < 12:
        score -= gap * 0.5

    # Normalize to 0-1 scale (20 is roughly max possible score)
    normalized_score = min(max(score / 20, 0), 1)

    # Additional adjustments for specific hand types

    # Premium hands
    if (high_rank >= 13 and low_rank >= 10) or (high_rank == low_rank and high_rank >= 10):
        normalized_score = max(normalized_score, 0.7)

    # Speculative hands with good potential
    if suited and gap <= 3 and low_rank >= 5:
        normalized_score = max(normalized_score, 0.5)

    return normalized_score

def calculate_kelly_bet(win_probability, odds=1):
    """
    Calculate optimal bet size using the Kelly Criterion.

    Args:
        win_probability: Probability of winning (0-1)
        odds: Payoff odds (default is 1:1 for most poker situations)

    Returns:
        fraction: Fraction of bankroll to bet (0-1)
    """
    # Standard Kelly formula: f* = (bp - q) / b
    # where:
    # f* = fraction of bankroll to bet
    # b = odds received on the bet (b to 1)
    # p = probability of winning
    # q = probability of losing (1-p)

    # Ensure win_probability is within bounds
    win_probability = max(0.01, min(0.99, win_probability))

    # Calculate Kelly fraction
    lose_probability = 1 - win_probability
    kelly_fraction = (win_probability * odds - lose_probability) / odds

    # Limit the Kelly bet to avoid over-betting
    # Using half-Kelly as a more conservative approach
    half_kelly = max(0, kelly_fraction * 0.5)

    return half_kelly

def pad_adjusted_agent(game: TexasHoldEm, pad_reader: PADReader) -> Tuple[ActionType, int]:
    """
    Poker agent that uses PAD analysis to inform its decisions and Kelly criterion for optimal betting.
    """
    player = game.players[game.current_player]
    current_board = game.board
    my_hand = game.hands[game.current_player]
    game_phase = determine_game_phase(game)

    # Get opponent's likely emotional state
    opponent_insights = pad_reader.detect_emotional_patterns(game_phase)

    # Get current PAD values for logging
    current_pad = pad_reader.get_current_pad() or {"pleasure": 0, "arousal": 0, "dominance": 0}

    # Calculate hand strength with improved preflop evaluation
    if game_phase == "preflop":
        hand_strength = evaluate_preflop_hand(my_hand)
    elif len(my_hand) + len(current_board) >= 5:
        hand_strength = get_five_card_rank_percentage(evaluate(my_hand, current_board))
    else:
        hand_strength = 0.3  # Default value

    # Calculate outs and pot odds
    outs, draw_type = get_outs_and_draw_type(my_hand, current_board)
    pot_odds = calculate_pot_odds(game)

    # Adjust winning probability based on PAD insights and game state
    adjusted_win_probability = hand_strength

    # If opponent shows signs of bluffing, increase our win probability
    if opponent_insights.get("likely_bluffing", False):
        bluff_confidence = opponent_insights.get("confidence", 0.5)
        adjusted_win_probability += bluff_confidence * 0.2

    # If opponent shows signs of a strong hand, decrease our win probability
    if opponent_insights.get("strong_hand", False):
        strong_confidence = opponent_insights.get("confidence", 0.5)
        adjusted_win_probability -= strong_confidence * 0.15

    # If opponent seems uncertain, exploit with higher win probability
    if opponent_insights.get("uncertain", False):
        adjusted_win_probability += 0.1

    # Clamp to valid probability range
    adjusted_win_probability = max(0.01, min(0.99, adjusted_win_probability))

    # Variables to track original and adjusted bet amounts
    original_bet = None
    bet_amount = None
    action = None

    # Preflop decision logic with position awareness
    if game_phase == "preflop":
        position = "early" if game.current_player < len(game.players) // 2 else "late"

        # Premium hands
        if hand_strength > 0.7:
            if player.state == PlayerState.TO_CALL and game.last_raise > game.bb * 4:
                # Re-evaluate calling big raises even with premium hands
                action = ActionType.CALL if adjusted_win_probability > 0.6 else ActionType.FOLD
                bet_amount = None
            else:
                # Raise with premium hands
                action = ActionType.RAISE

                # Calculate original bet without PAD adjustment
                kelly_fraction = calculate_kelly_bet(hand_strength, 1.2)
                pot_size = sum(p.chips_bet for p in game.players) + game.pot
                original_bet = int(kelly_fraction * pot_size)

                # Then calculate PAD-adjusted bet
                kelly_fraction = calculate_kelly_bet(adjusted_win_probability, 1.2)
                kelly_bet = int(kelly_fraction * pot_size)

                # Ensure minimum raise and maximum chip constraints
                min_raise = max(game.bb, game.last_raise * 2)
                bet_amount = max(min_raise, min(kelly_bet, player.chips))

        # Playable hands
        elif hand_strength > 0.5:
            if position == "late" and game.last_raise <= game.bb * 2:
                action = ActionType.RAISE

                # Calculate original bet without PAD adjustment
                original_bet = min(player.chips, game.bb * 3)

                # Calculate PAD-adjusted bet
                if opponent_insights.get("likely_bluffing", False):
                    bet_amount = min(player.chips, game.bb * 4)
                elif opponent_insights.get("strong_hand", False):
                    bet_amount = min(player.chips, game.bb * 2)
                else:
                    bet_amount = original_bet
            elif player.state == PlayerState.TO_CALL and game.last_raise <= game.bb * 2:
                action = ActionType.CALL
                bet_amount = None
                original_bet = None
            else:
                action = ActionType.FOLD
                bet_amount = None
                original_bet = None

        # Marginal hands
        elif hand_strength > 0.3:
            if position == "late" and game.last_raise == 0:
                action = ActionType.RAISE
                original_bet = min(player.chips, game.bb * 2)

                # Adjust based on opponent's emotional state
                if opponent_insights.get("uncertain", False):
                    bet_amount = min(player.chips, game.bb * 2.5)
                else:
                    bet_amount = original_bet
            elif position == "late" and game.last_raise <= game.bb and opponent_insights.get("uncertain", False):
                action = ActionType.CALL
                bet_amount = None
                original_bet = None
            elif player.state != PlayerState.TO_CALL:
                action = ActionType.CHECK
                bet_amount = None
                original_bet = None
            else:
                action = ActionType.FOLD
                bet_amount = None
                original_bet = None

        # Weak hands
        else:
            if position == "late" and game.last_raise == 0 and (
                opponent_insights.get("uncertain", False) or
                opponent_insights.get("likely_bluffing", False)
            ):
                # Occasional steal attempt
                if time.time() % 10 < 3:  # 30% chance to steal
                    action = ActionType.RAISE
                    original_bet = min(player.chips, game.bb * 2.5)
                    bet_amount = original_bet
                else:
                    action = ActionType.CHECK if player.state != PlayerState.TO_CALL else ActionType.FOLD
                    bet_amount = None
                    original_bet = None
            else:
                action = ActionType.CHECK if player.state != PlayerState.TO_CALL else ActionType.FOLD
                bet_amount = None
                original_bet = None

    # Post-flop decision logic
    else:
        # Strong hands (top pair+, strong draws with good equity)
        if adjusted_win_probability > 0.7:
            action = ActionType.RAISE

            # Calculate original bet size without PAD adjustment
            kelly_fraction = calculate_kelly_bet(hand_strength, 1.5)
            pot_size = sum(p.chips_bet for p in game.players) + game.pot
            original_bet = int(kelly_fraction * pot_size)

            # Apply Kelly Criterion with PAD adjustment for optimal bet sizing
            kelly_fraction = calculate_kelly_bet(adjusted_win_probability, 1.5)
            kelly_bet = int(kelly_fraction * pot_size)

            # Adjust bet size based on opponent emotional state
            if opponent_insights.get("uncertain", False) or opponent_insights.get("likely_bluffing", False):
                kelly_bet = int(kelly_bet * 1.3)  # More aggressive when opponent is uncertain/bluffing

            # Ensure minimum raise and maximum chip constraints
            min_raise = max(game.bb, game.last_raise * 2)
            bet_amount = max(min_raise, min(kelly_bet, player.chips))

        # Medium strength hands and good draws
        elif adjusted_win_probability > 0.4 or draw_type in ["Flush Draw", "Open-Ended Straight Draw"]:
            # Calculate pot odds threshold based on draw strength
            pot_odds_threshold = outs / 47 if outs else 0.2

            # Adjust threshold based on opponent's emotional state
            if opponent_insights.get("likely_bluffing", False):
                pot_odds_threshold *= 0.7  # More likely to call against bluffers
            if opponent_insights.get("strong_hand", False):
                pot_odds_threshold *= 1.3  # More cautious against strong hands

            # Decision based on pot odds and position
            if pot_odds > pot_odds_threshold:
                # Consider raising with semi-bluffs in position with good equity
                if game.last_raise == 0 and draw_type in ["Flush Draw", "Open-Ended Straight Draw"]:
                    action = ActionType.RAISE

                    # Calculate original bet size
                    kelly_fraction = calculate_kelly_bet(hand_strength, 1.0) * 0.7
                    pot_size = sum(p.chips_bet for p in game.players) + game.pot
                    original_bet = min(player.chips, max(game.bb, int(kelly_fraction * pot_size)))

                    # Conservative Kelly for semi-bluffs with PAD adjustment
                    kelly_fraction = calculate_kelly_bet(adjusted_win_probability, 1.0) * 0.7
                    pot_size = sum(p.chips_bet for p in game.players) + game.pot
                    bet_amount = min(player.chips, max(game.bb, int(kelly_fraction * pot_size)))
                else:
                    action = ActionType.CALL
                    bet_amount = None
                    original_bet = None
            else:
                # Check if no bet to call, otherwise fold
                if game.last_raise == 0:
                    action = ActionType.CHECK
                    bet_amount = None
                    original_bet = None
                # Still call sometimes against likely bluffers with decent hand
                elif opponent_insights.get("likely_bluffing", False) and opponent_insights.get("confidence", 0) > 0.6:
                    action = ActionType.CALL
                    bet_amount = None
                    original_bet = None
                else:
                    action = ActionType.FOLD
                    bet_amount = None
                    original_bet = None

        # Weak hands and bad draws
        else:
            # Check if possible, otherwise consider bluffing
            if game.last_raise == 0:
                # Occasionally bluff with weak hands in the right conditions
                if not opponent_insights.get("strong_hand", False) and (
                    opponent_insights.get("uncertain", False) or
                    time.time() % 10 < 2  # 20% random bluff frequency
                ):
                    action = ActionType.RAISE

                    # Small bluff sizing - original
                    kelly_fraction = 0.1  # Small fraction for bluffs
                    pot_size = sum(p.chips_bet for p in game.players) + game.pot
                    original_bet = min(player.chips, max(game.bb, int(kelly_fraction * pot_size)))

                    # PAD-adjusted bluff sizing
                    if opponent_insights.get("uncertain", False):
                        kelly_fraction = 0.15  # Larger bluff when opponent is uncertain
                    bet_amount = min(player.chips, max(game.bb, int(kelly_fraction * pot_size)))
                else:
                    action = ActionType.CHECK
                    bet_amount = None
                    original_bet = None
            else:
                action = ActionType.FOLD
                bet_amount = None
                original_bet = None

    # Calculate pot size for logging
    pot_size = sum(p.chips_bet for p in game.players) + game.pot

    # Log the decision with all context
    pad_reader.log_decision(
        current_pad=current_pad,
        hand=my_hand,
        board=current_board,
        game_phase=game_phase,
        hand_strength=hand_strength,
        adjusted_win_prob=adjusted_win_probability,
        original_bet=original_bet,
        adjusted_bet=bet_amount,
        action=action,
        pot_size=pot_size,
        opponent_insights=opponent_insights
    )

    # Update PAD reader with our decision
    pad_reader.detect_emotional_patterns(game_phase, hand_strength, action)

    return action, bet_amount


# Main game loop
def main():
    # Initialize PAD reader
    pad_reader = PADReader(csv_path="pad_values.csv")

    # Calibrate PAD baseline
    print("Starting PAD calibration. Please maintain neutral emotions...")
    pad_reader.calibrate(num_samples=30)

    # Initialize game
    game = TexasHoldEm(buyin=500, big_blind=5, small_blind=2, max_players=2)
    gui = TextGUI(game=game)

    print("Starting poker game with PAD analysis...")

    while game.is_game_running():
        game.start_hand()

        print("\n--- New Hand ---")

        while game.is_hand_running():
            if game.current_player == 1:  # Bot's turn
                action, amount = pad_adjusted_agent(game, pad_reader)
                print(f"Bot action: {action} {amount if amount else ''}")
                game.take_action(action, amount)
                time.sleep(1)
            else:  # Human player's turn
                gui.run_step()
                time.sleep(1)

        path = game.export_history('./pgns')  # save history
        gui.replay_history(path)


if __name__ == "__main__":
    main()