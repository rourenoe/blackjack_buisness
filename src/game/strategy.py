from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

CARD_RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'A']
CARD_VALUE = {rank: int(rank) if rank != 'A' else 1 for rank in CARD_RANKS}
CARD_VALUE['10'] = 10

# Infinite-shoe approximation for 6 decks (52 cards per deck)
DRAW_COUNTS = {
    '2': 4,
    '3': 4,
    '4': 4,
    '5': 4,
    '6': 4,
    '7': 4,
    '8': 4,
    '9': 4,
    '10': 16,
    'A': 4,
}
TOTAL_CARDS = sum(DRAW_COUNTS.values())
DRAW_PROBS = {rank: count / TOTAL_CARDS for rank, count in DRAW_COUNTS.items()}

ActionName = str

@dataclass(frozen=True)
class ActionResult:
    ev: float
    win_prob: float
    push_prob: float
    loss_prob: float

    def total_probability(self) -> float:
        return self.win_prob + self.push_prob + self.loss_prob


def add_card_value(total: int, usable_ace: bool, rank: str) -> Tuple[int, bool]:
    if rank == 'A':
        total += 11
        usable_ace = True
    else:
        total += CARD_VALUE[rank]

    if total > 21 and usable_ace:
        total -= 10
        usable_ace = False

    return total, usable_ace


def compute_hand_value(ranks: Tuple[str, ...]) -> Tuple[int, bool]:
    total = 0
    aces = 0
    for rank in ranks:
        if rank == 'A':
            aces += 1
        else:
            total += CARD_VALUE[rank]

    if aces:
        if total + 11 + (aces - 1) <= 21:
            total += 11 + (aces - 1)
            usable_ace = True
        else:
            total += aces
            usable_ace = False
    else:
        usable_ace = False

    return total, usable_ace


@lru_cache(maxsize=None)
def dealer_final_distribution(total: int, usable_ace: bool) -> Dict[str, float]:
    if total > 21:
        if usable_ace:
            return dealer_final_distribution(total - 10, False)
        return {'bust': 1.0}

    if total >= 17:
        return {str(total): 1.0}

    distribution: Dict[str, float] = {}
    for rank, probability in DRAW_PROBS.items():
        next_total, next_usable = add_card_value(total, usable_ace, rank)
        sub_distribution = dealer_final_distribution(next_total, next_usable)
        for outcome, sub_prob in sub_distribution.items():
            distribution[outcome] = distribution.get(outcome, 0.0) + probability * sub_prob

    return distribution


@lru_cache(maxsize=None)
def dealer_start_distribution(upcard_rank: str) -> Dict[str, float]:
    distribution: Dict[str, float] = {}
    for rank, probability in DRAW_PROBS.items():
        dealer_total, usable_ace = compute_hand_value((upcard_rank, rank))
        sub_distribution = dealer_final_distribution(dealer_total, usable_ace)
        for outcome, sub_prob in sub_distribution.items():
            distribution[outcome] = distribution.get(outcome, 0.0) + probability * sub_prob
    return distribution


def compare_player_vs_dealer(player_total: int, dealer_dist: Dict[str, float]) -> ActionResult:
    if player_total > 21:
        return ActionResult(ev=-1.0, win_prob=0.0, push_prob=0.0, loss_prob=1.0)

    win_prob = 0.0
    push_prob = 0.0
    loss_prob = 0.0
    for outcome, probability in dealer_dist.items():
        if outcome == 'bust':
            win_prob += probability
        else:
            dealer_score = int(outcome)
            if player_total > dealer_score:
                win_prob += probability
            elif player_total == dealer_score:
                push_prob += probability
            else:
                loss_prob += probability

    ev = win_prob - loss_prob
    return ActionResult(ev=ev, win_prob=win_prob, push_prob=push_prob, loss_prob=loss_prob)


@lru_cache(maxsize=None)
def evaluate_state(
    player_total: int,
    usable_ace: bool,
    num_cards: int,
    dealer_upcard: str,
    can_double: bool,
    can_split: bool,
    pair_rank: Optional[str],
    split_aces_one_card: bool = True,
) -> Tuple[ActionName, ActionResult, Dict[ActionName, ActionResult]]:
    if player_total > 21:
        if usable_ace:
            player_total -= 10
            usable_ace = False
        else:
            terminal = ActionResult(ev=-1.0, win_prob=0.0, push_prob=0.0, loss_prob=1.0)
            return 'bust', terminal, {}

    if player_total == 21 and num_cards == 2:
        # If the player has a natural 21 from the initial two cards, stand is the only meaningful action.
        stand_result = compare_player_vs_dealer(player_total, dealer_start_distribution(dealer_upcard))
        return 'stand', stand_result, {'stand': stand_result}

    dealer_distribution = dealer_start_distribution(dealer_upcard)
    stand_result = compare_player_vs_dealer(player_total, dealer_distribution)
    action_results: Dict[ActionName, ActionResult] = {'stand': stand_result}

    # Hit action always remains valid until bust.
    hit_win = hit_push = hit_loss = hit_ev = 0.0
    for rank, probability in DRAW_PROBS.items():
        next_total, next_usable = add_card_value(player_total, usable_ace, rank)
        if next_total > 21:
            hit_loss += probability
            hit_ev -= probability
        else:
            _, next_result, _ = evaluate_state(
                next_total,
                next_usable,
                num_cards + 1,
                dealer_upcard,
                can_double=False,
                can_split=False,
                pair_rank=None,
                split_aces_one_card=split_aces_one_card,
            )
            hit_win += probability * next_result.win_prob
            hit_push += probability * next_result.push_prob
            hit_loss += probability * next_result.loss_prob
            hit_ev += probability * next_result.ev
    action_results['hit'] = ActionResult(ev=hit_ev, win_prob=hit_win, push_prob=hit_push, loss_prob=hit_loss)

    if can_double and num_cards == 2:
        double_win = double_push = double_loss = double_ev = 0.0
        for rank, probability in DRAW_PROBS.items():
            next_total, next_usable = add_card_value(player_total, usable_ace, rank)
            if next_total > 21:
                double_loss += probability
                double_ev -= 2 * probability
            else:
                result = compare_player_vs_dealer(next_total, dealer_distribution)
                double_win += probability * result.win_prob
                double_push += probability * result.push_prob
                double_loss += probability * result.loss_prob
                double_ev += 2 * result.ev * probability
        action_results['double'] = ActionResult(ev=double_ev, win_prob=double_win, push_prob=double_push, loss_prob=double_loss)

    if can_split and num_cards == 2 and pair_rank is not None:
        if pair_rank == 'A' and split_aces_one_card:
            split_win = split_push = split_loss = split_ev = 0.0
            for rank, probability in DRAW_PROBS.items():
                next_total, next_usable = compute_hand_value(('A', rank))
                result = compare_player_vs_dealer(next_total, dealer_distribution)
                split_win += probability * result.win_prob
                split_push += probability * result.push_prob
                split_loss += probability * result.loss_prob
                split_ev += 2 * probability * result.ev
            action_results['split'] = ActionResult(
                ev=split_ev,
                win_prob=split_win,
                push_prob=split_push,
                loss_prob=split_loss,
            )
        else:
            split_win = split_push = split_loss = split_ev = 0.0
            for rank, probability in DRAW_PROBS.items():
                new_pair = tuple(sorted((pair_rank, rank), key=lambda x: CARD_RANKS.index(x)))
                next_total, next_usable = compute_hand_value(new_pair)
                _, next_result, _ = evaluate_state(
                    next_total,
                    next_usable,
                    num_cards=2,
                    dealer_upcard=dealer_upcard,
                    can_double=True,
                    can_split=False,
                    pair_rank=None,
                    split_aces_one_card=split_aces_one_card,
                )
                split_win += probability * next_result.win_prob
                split_push += probability * next_result.push_prob
                split_loss += probability * next_result.loss_prob
                split_ev += 2 * probability * next_result.ev
            action_results['split'] = ActionResult(ev=split_ev, win_prob=split_win, push_prob=split_push, loss_prob=split_loss)

    best_action = max(action_results.items(), key=lambda item: (item[1].ev, item[1].win_prob))[0]
    return best_action, action_results[best_action], action_results


def build_strategy_grid() -> Dict[str, List[List[Dict[str, object]]]]:
    dealer_upcards = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'A']

    hard_totals = list(range(5, 18))
    soft_totals = list(range(13, 21))
    pair_ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'A']

    def build_rows(state_labels: List[str], state_factory):
        rows = []
        for label in state_labels:
            row = []
            for dealer_upcard in dealer_upcards:
                best_action, best_result, action_results = state_factory(label, dealer_upcard)
                cell = {
                    'label': label,
                    'dealer_upcard': dealer_upcard,
                    'best_action': best_action,
                    'best_ev': best_result.ev,
                    'actions': {
                        action: {
                            'ev': result.ev,
                            'win_prob': result.win_prob,
                            'push_prob': result.push_prob,
                            'loss_prob': result.loss_prob,
                        }
                        for action, result in action_results.items()
                    },
                }
                row.append(cell)
            rows.append(row)
        return rows

    def make_hard_state(label: int, dealer_upcard: str):
        return evaluate_state(
            player_total=label,
            usable_ace=False,
            num_cards=2,
            dealer_upcard=dealer_upcard,
            can_double=True,
            can_split=False,
            pair_rank=None,
        )

    def make_soft_state(label: int, dealer_upcard: str):
        rank = str(label - 11)
        return evaluate_state(
            player_total=label,
            usable_ace=True,
            num_cards=2,
            dealer_upcard=dealer_upcard,
            can_double=True,
            can_split=False,
            pair_rank=None,
        )

    def make_pair_state(label: str, dealer_upcard: str):
        total, usable_ace = compute_hand_value((label, label))
        return evaluate_state(
            player_total=total,
            usable_ace=usable_ace,
            num_cards=2,
            dealer_upcard=dealer_upcard,
            can_double=True,
            can_split=True,
            pair_rank=label,
        )

    return {
        'hard_totals': build_rows(hard_totals, make_hard_state),
        'soft_totals': build_rows(soft_totals, make_soft_state),
        'pairs': build_rows(pair_ranks, make_pair_state),
        'dealer_upcards': dealer_upcards,
    }


def format_percentage(value: float) -> str:
    return f"{value * 100:.1f}%"
