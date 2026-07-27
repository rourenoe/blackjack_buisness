from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

try:
    import pygame
except ImportError as exc:
    raise ImportError('The training GUI requires pygame. Install it with `pip install pygame`.') from exc

from game.card import Card
from game.strategy import compute_hand_value, evaluate_state

OUTPUT_DIR = Path(__file__).resolve().parents[2]
SAVE_ROOT = OUTPUT_DIR / 'save'
WINDOW_WIDTH = 1600
WINDOW_HEIGHT = 1180
CARD_WIDTH = 120
CARD_HEIGHT = 174
BUTTON_WIDTH = 180
BUTTON_HEIGHT = 50
FOOTER_HEIGHT = 90
FPS = 60

def format_hand_score(ranks: tuple[str, ...]) -> str:
    total_with_aces = sum(1 if rank == 'A' else Card(suit='hearts', rank=rank).value() for rank in ranks)
    aces = sum(1 for rank in ranks if rank == 'A')
    totals = set()
    for extra_aces in range(aces + 1):
        totals.add(total_with_aces + 10 * extra_aces)
    totals = sorted(totals, reverse=True)
    valid_totals = [total for total in totals if total <= 21]
    if valid_totals:
        totals = valid_totals
    elif totals:
        totals = [min(totals)]
    return '/'.join(str(total) for total in totals)

COLORS = {
    'stand': {'bg': (25, 71, 140), 'text': (255, 255, 255)},
    'hit': {'bg': (210, 145, 23), 'text': (0, 0, 0)},
    'double': {'bg': (210, 70, 70), 'text': (255, 255, 255)},
    'split': {'bg': (190, 95, 155), 'text': (255, 255, 255)},
}
PROGRESS_COLORS = {
    'correct': (92, 179, 88),
    'incorrect': (220, 72, 72),
    'unknown': (50, 50, 70),
}
BACKGROUND = (10, 18, 30)
CARD_BG = (245, 245, 245)
TEXT_COLOR = (240, 240, 240)

HARD_HANDS = {
    5: ('2', '3'),
    6: ('2', '4'),
    7: ('2', '5'),
    8: ('2', '6'),
    9: ('2', '7'),
    10: ('2', '8'),
    11: ('2', '9'),
    12: ('2', '10'),
    13: ('3', '10'),
    14: ('4', '10'),
    15: ('5', '10'),
    16: ('6', '10'),
    17: ('7', '10'),
}
SOFT_HANDS = {
    total: ('A', str(total - 11))
    for total in range(13, 21)
}
PAIR_RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'A']
DEALER_RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'A']
SUITS = ['hearts', 'diamonds', 'clubs', 'spades']


@dataclass
class TrainingScenario:
    player_ranks: tuple[str, str]
    dealer_rank: str
    category: str
    label: str
    best_action: str
    best_result: object
    action_results: Dict[str, object]

    @property
    def key(self) -> str:
        return f"{self.player_ranks[0]}-{self.player_ranks[1]}|{self.dealer_rank}"

    @property
    def player_label(self) -> str:
        return self.label

    @property
    def dealer_label(self) -> str:
        return self.dealer_rank

    @property
    def cards(self) -> List[Card]:
        rank1, rank2 = self.player_ranks
        suit1 = SUITS[0]
        suit2 = SUITS[1] if rank1 != rank2 else SUITS[2]
        return [Card(suit1, rank1), Card(suit2, rank2)]

    @property
    def dealer_card(self) -> Card:
        return Card(SUITS[3], self.dealer_rank)

    @property
    def can_split(self) -> bool:
        return self.player_ranks[0] == self.player_ranks[1]

    @property
    def valid_actions(self) -> List[str]:
        actions = ['stand', 'hit', 'double']
        if self.can_split:
            actions.append('split')
        return actions

    def description(self) -> str:
        return f"{self.player_ranks[0]}+{self.player_ranks[1]} vs {self.dealer_rank}"


class TrainingSession:
    def __init__(self, user_name: str) -> None:
        self.user_name = user_name
        self.save_dir = SAVE_ROOT / self.user_name
        self.history_path = self.save_dir / 'training_history.txt'
        self.errors_path = self.save_dir / 'training_errors.txt'
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.errors: set[str] = set()
        self.latest_attempts: Dict[str, bool] = {}
        self.total_attempts = 0
        self.correct_attempts = 0
        self.incorrect_attempts = 0

        self.load_errors()
        self.load_history()

    def load_errors(self) -> None:
        if self.errors_path.exists():
            self.errors = {
                line.strip()
                for line in self.errors_path.read_text(encoding='utf-8').splitlines()
                if line.strip() and not line.strip().startswith('#')
            }
        else:
            self.errors = set()

    def load_history(self) -> None:
        self.latest_attempts = {}
        self.total_attempts = 0
        self.correct_attempts = 0
        self.incorrect_attempts = 0
        if not self.history_path.exists():
            return

        for line in self.history_path.read_text(encoding='utf-8').splitlines():
            if not line.strip() or line.startswith('#'):
                continue
            parts = line.split(',')
            if len(parts) < 7:
                continue
            key = parts[3]
            try:
                correct = bool(int(parts[6]))
            except ValueError:
                continue
            self.latest_attempts[key] = correct
            self.total_attempts += 1
            if correct:
                self.correct_attempts += 1
            else:
                self.incorrect_attempts += 1

    def save_errors(self) -> None:
        self.errors_path.write_text('\n'.join(sorted(self.errors)) + ('\n' if self.errors else ''), encoding='utf-8')

    def log_attempt(self, scenario: TrainingScenario, guess: str, correct: bool) -> None:
        self.total_attempts += 1
        self.latest_attempts[scenario.key] = correct
        if correct:
            self.correct_attempts += 1
            self.errors.discard(scenario.key)
        else:
            self.incorrect_attempts += 1
            self.errors.add(scenario.key)
        self.save_errors()
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        with self.history_path.open('a', encoding='utf-8') as history_file:
            history_file.write(
                f"{datetime.now().isoformat()},{scenario.category},{scenario.label},{scenario.key},{guess},"
                f"{scenario.best_action},{int(correct)}\n"
            )


class BlackjackTrainer:
    def __init__(self, errors_only: bool = False) -> None:
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption('Blackjack Training')
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 28)
        self.small_font = pygame.font.Font(None, 20)
        self.large_font = pygame.font.Font(None, 40)

        self.errors_only = errors_only
        self.user_selection_active = True
        self.user_name: Optional[str] = None
        self.username_input = ''
        self.user_buttons: List[Dict[str, object]] = []
        self.existing_users = self.load_existing_users()

        self.session: Optional[TrainingSession] = None
        self.scenarios: List[TrainingScenario] = []
        self.current_index = 0
        self.feedback_text = ''
        self.feedback_color = (255, 255, 255)
        self.progress: Dict[str, Dict[str, Dict[str, Optional[bool]]]] = self.init_progress()
        self.buttons: List[Dict[str, object]] = []
        self.running = True

    def load_existing_users(self) -> List[str]:
        SAVE_ROOT.mkdir(parents=True, exist_ok=True)
        return sorted(
            [item.name for item in SAVE_ROOT.iterdir() if item.is_dir()]
        )

    def sanitize_user_name(self, name: str) -> str:
        sanitized = ''.join(
            ch if ch.isalnum() or ch in ('_', '-') else '_'
            for ch in name.strip()
        )
        return sanitized or 'user'

    def select_user(self, user_name: str) -> None:
        user_name = self.sanitize_user_name(user_name)
        self.user_name = user_name
        self.session = TrainingSession(user_name)
        self.scenarios = self.build_scenarios()
        if self.errors_only and self.session.errors:
            error_scenarios = [s for s in self.scenarios if s.key in self.session.errors]
            if error_scenarios:
                self.scenarios = error_scenarios
        random.shuffle(self.scenarios)
        self.current_index = 0
        self.feedback_text = ''
        self.feedback_color = (255, 255, 255)
        self.progress = self.init_progress()
        self.load_progress_from_history()
        self.buttons = self.create_buttons()
        self.user_selection_active = False
        self.existing_users = self.load_existing_users()

    def init_progress(self) -> Dict[str, Dict[str, Dict[str, Optional[bool]]]]:
        progress = {'hard': {}, 'soft': {}, 'pair': {}}
        for total in HARD_HANDS.keys():
            progress['hard'][str(total)] = {rank: None for rank in DEALER_RANKS}
        for total in SOFT_HANDS.keys():
            progress['soft'][str(total)] = {rank: None for rank in DEALER_RANKS}
        for rank in PAIR_RANKS:
            progress['pair'][rank] = {dealer: None for dealer in DEALER_RANKS}
        return progress

    def draw_user_selection(self) -> None:
        self.screen.fill(BACKGROUND)
        title = self.large_font.render('Select or create a user', True, TEXT_COLOR)
        self.screen.blit(title, (50, 40))

        prompt = self.font.render('Enter username and press Enter, or click an existing user:', True, TEXT_COLOR)
        self.screen.blit(prompt, (50, 100))

        input_rect = pygame.Rect(50, 150, 560, 42)
        pygame.draw.rect(self.screen, (255, 255, 255), input_rect, border_radius=8)
        pygame.draw.rect(self.screen, (100, 100, 120), input_rect, 2, border_radius=8)
        input_text = self.font.render(self.username_input or 'New user name...', True, (0, 0, 0) if self.username_input else (150, 150, 150))
        self.screen.blit(input_text, (input_rect.x + 12, input_rect.y + 8))

        existing_title = self.font.render('Existing users:', True, TEXT_COLOR)
        self.screen.blit(existing_title, (50, 220))

        self.user_buttons = []
        for idx, user in enumerate(self.existing_users):
            rect = pygame.Rect(50 + (idx % 4) * 260, 260 + (idx // 4) * 52, 240, 42)
            pygame.draw.rect(self.screen, (40, 60, 90), rect, border_radius=10)
            pygame.draw.rect(self.screen, (100, 120, 160), rect, 2, border_radius=10)
            text = self.font.render(user, True, TEXT_COLOR)
            self.screen.blit(text, (rect.x + 12, rect.y + 8))
            self.user_buttons.append({'rect': rect, 'user': user})

        instructions = self.small_font.render('You can store one session per user in the save folder.', True, TEXT_COLOR)
        self.screen.blit(instructions, (50, 460))

    def handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif self.user_selection_active:
                self.handle_user_selection_event(event)
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                self.handle_click(event.pos)
            elif event.type == pygame.KEYDOWN:
                self.handle_key(event.key)

    def handle_user_selection_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            for button in self.user_buttons:
                if button['rect'].collidepoint(event.pos):
                    self.select_user(button['user'])
                    return
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_BACKSPACE:
                self.username_input = self.username_input[:-1]
            elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                candidate = self.username_input.strip()
                if candidate:
                    self.select_user(candidate)
            elif event.unicode and len(self.username_input) < 20:
                self.username_input += event.unicode

    def load_progress_from_errors(self) -> None:
        if not self.session:
            return
        for scenario in self.scenarios:
            if scenario.key in self.session.errors:
                category = self.progress_key(scenario.category)
                self.progress[category][scenario.label][scenario.dealer_rank] = False

    def load_progress_from_history(self) -> None:
        if not self.session:
            return
        for scenario in self.scenarios:
            latest = self.session.latest_attempts.get(scenario.key)
            if latest is True:
                category = self.progress_key(scenario.category)
                self.progress[category][scenario.label][scenario.dealer_rank] = True
            elif latest is False:
                category = self.progress_key(scenario.category)
                self.progress[category][scenario.label][scenario.dealer_rank] = False

    def progress_key(self, category: str) -> str:
        return 'hard' if category == 'Hard Totals' else 'soft' if category == 'Soft Totals' else 'pair'

    def build_scenarios(self) -> List[TrainingScenario]:
        scenarios: List[TrainingScenario] = []
        for total, player_ranks in HARD_HANDS.items():
            for dealer in DEALER_RANKS:
                best_action, best_result, action_results = evaluate_state(
                    total, False, 2, dealer, True, False, None
                )
                scenario = TrainingScenario(player_ranks, dealer, 'Hard Totals', str(total), best_action, best_result, action_results)
                scenarios.append(scenario)

        for total, player_ranks in SOFT_HANDS.items():
            for dealer in DEALER_RANKS:
                best_action, best_result, action_results = evaluate_state(
                    total, True, 2, dealer, True, False, None
                )
                scenario = TrainingScenario(player_ranks, dealer, 'Soft Totals', str(total), best_action, best_result, action_results)
                scenarios.append(scenario)

        for rank in PAIR_RANKS:
            player_ranks = (rank, rank)
            total, usable = compute_hand_value(player_ranks)
            for dealer in DEALER_RANKS:
                best_action, best_result, action_results = evaluate_state(
                    total, usable, 2, dealer, True, True, rank
                )
                scenario = TrainingScenario(player_ranks, dealer, 'Pairs', rank, best_action, best_result, action_results)
                scenarios.append(scenario)

        return scenarios

    def create_buttons(self) -> List[Dict[str, object]]:
        buttons = []
        labels = ['Stand', 'Hit', 'Double', 'Split']
        actions = ['stand', 'hit', 'double', 'split']
        x_start = 50
        y_start = WINDOW_HEIGHT - BUTTON_HEIGHT - 60
        for idx, (label, action) in enumerate(zip(labels, actions)):
            rect = pygame.Rect(x_start + idx * (BUTTON_WIDTH + 20), y_start, BUTTON_WIDTH, BUTTON_HEIGHT)
            buttons.append({'rect': rect, 'label': label, 'action': action})
        return buttons

    def draw_card(self, card: Card, x: int, y: int) -> None:
        try:
            current_dir = Path(__file__).parent
            image_path = current_dir.parent.parent / card.path_to_image
            card_image = pygame.image.load(str(image_path))
            card_image = pygame.transform.scale(card_image, (CARD_WIDTH, CARD_HEIGHT))
            self.screen.blit(card_image, (x, y))
        except (pygame.error, FileNotFoundError):
            rect = pygame.Rect(x, y, CARD_WIDTH, CARD_HEIGHT)
            pygame.draw.rect(self.screen, CARD_BG, rect)
            pygame.draw.rect(self.screen, (0, 0, 0), rect, 2)
            text = self.font.render(f"{card.rank}{card.suit[0]}", True, (0, 0, 0))
            text_rect = text.get_rect(center=rect.center)
            self.screen.blit(text, text_rect)

    def draw(self) -> None:
        if self.user_selection_active:
            self.draw_user_selection()
            return

        self.screen.fill(BACKGROUND)
        self.draw_headers()
        self.draw_current_situation()
        self.draw_buttons()
        self.draw_progress_tables()
        self.draw_feedback()
        self.draw_footer()

    def draw_headers(self) -> None:
        title = self.large_font.render('Blackjack Training', True, TEXT_COLOR)
        self.screen.blit(title, (50, 20))
        user_label = f'User: {self.user_name}' if self.user_name else 'User: unknown'
        summary = self.small_font.render(
            f'{user_label} | Attempt {self.current_index + 1}/{len(self.scenarios)} | Correct: {self.session.correct_attempts} | Wrong: {self.session.incorrect_attempts} | Errors left: {len(self.session.errors)}',
            True,
            TEXT_COLOR,
        )
        self.screen.blit(summary, (50, 70))

    def draw_current_situation(self) -> None:
        scenario = self.scenarios[self.current_index]
        self.draw_card(scenario.dealer_card, 50, 150)
        dealer_score_text = format_hand_score((scenario.dealer_rank,))
        self.screen.blit(self.font.render(f'Dealer: {dealer_score_text}', True, TEXT_COLOR), (50 + CARD_WIDTH + 20, 180))
        for idx, card in enumerate(scenario.cards):
            self.draw_card(card, 50 + idx * (CARD_WIDTH + 20), 400)
        player_score_text = format_hand_score((scenario.player_ranks[0], scenario.player_ranks[1]))
        self.screen.blit(self.font.render(f'Player: {player_score_text}', True, TEXT_COLOR), (50 + CARD_WIDTH * 2 + 40, 430))
        details = [
            f'Category: {scenario.category}',
            f'Hand: {scenario.player_label}',
        ]
        for idx, text in enumerate(details):
            self.screen.blit(self.font.render(text, True, TEXT_COLOR), (50 + CARD_WIDTH * 2 + 60, 170 + idx * 32))

    def draw_buttons(self) -> None:
        scenario = self.scenarios[self.current_index]
        for button in self.buttons:
            enabled = button['action'] in scenario.valid_actions
            color = (120, 120, 120) if not enabled else (70, 130, 180)
            pygame.draw.rect(self.screen, color, button['rect'], border_radius=10)
            text = self.font.render(button['label'], True, TEXT_COLOR)
            text_rect = text.get_rect(center=button['rect'].center)
            self.screen.blit(text, text_rect)

    def draw_progress_tables(self) -> None:
        x = 50
        y = 640
        self.draw_progress_table('Hard', self.progress['hard'], x, y)
        self.draw_progress_table('Soft', self.progress['soft'], x + 480, y)
        self.draw_progress_table('Pair', self.progress['pair'], x + 960, y)

    def draw_progress_table(self, title: str, table_data: Dict[str, Dict[str, Optional[bool]]], x: int, y: int) -> None:
        self.screen.blit(self.font.render(f'{title} progress', True, TEXT_COLOR), (x, y))
        y += 28
        cell = 18
        for row_idx, (row_label, row_data) in enumerate(table_data.items()):
            self.screen.blit(self.small_font.render(str(row_label), True, TEXT_COLOR), (x, y + row_idx * (cell + 2)))
            for col_idx, dealer in enumerate(DEALER_RANKS):
                value = row_data[dealer]
                color = PROGRESS_COLORS['unknown'] if value is None else PROGRESS_COLORS['correct'] if value else PROGRESS_COLORS['incorrect']
                rect = pygame.Rect(x + 40 + col_idx * (cell + 2), y + row_idx * (cell + 2), cell, cell)
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, (100, 100, 120), rect, 1)

    def draw_feedback(self) -> None:
        if self.feedback_text:
            text = self.large_font.render(self.feedback_text, True, self.feedback_color)
            rect = text.get_rect(center=(WINDOW_WIDTH // 2 + 100, 240))
            self.screen.blit(text, rect)

    def draw_footer(self) -> None:
        info_lines = [
            'Click the button for the best action. Correct answer marks the scenario green, wrong marks it red.',
            'Each user session is stored in save/<username>/training_history.txt and training_errors.txt.',
        ]
        for idx, line in enumerate(info_lines):
            self.screen.blit(self.small_font.render(line, True, TEXT_COLOR), (50, WINDOW_HEIGHT - FOOTER_HEIGHT + 16 + idx * 22))

    def run(self) -> None:
        while self.running:
            self.handle_events()
            self.draw()
            pygame.display.flip()
            self.clock.tick(FPS)
        pygame.quit()

    def handle_click(self, pos: tuple[int, int]) -> None:
        for button in self.buttons:
            if button['rect'].collidepoint(pos):
                self.try_action(button['action'])
                return

    def handle_key(self, key: int) -> None:
        mapping = {
            pygame.K_s: 'stand',
            pygame.K_h: 'hit',
            pygame.K_d: 'double',
            pygame.K_p: 'split',
        }
        action = mapping.get(key)
        if action:
            self.try_action(action)

    def try_action(self, action: str) -> None:
        scenario = self.scenarios[self.current_index]
        if action not in scenario.valid_actions:
            self.feedback_text = f'Action {action} not valid for this situation.'
            self.feedback_color = (220, 180, 60)
            return
        correct = action == scenario.best_action
        self.session.log_attempt(scenario, action, correct)
        category = self.progress_key(scenario.category)
        self.progress[category][scenario.label][scenario.dealer_rank] = correct
        self.feedback_text = 'Correct!' if correct else f'Wrong - best was {scenario.best_action}'
        self.feedback_color = (92, 179, 88) if correct else (220, 72, 72)
        self.current_index = (self.current_index + 1) % len(self.scenarios)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Blackjack training GUI.')
    parser.add_argument('--errors', action='store_true', help='Practice only saved errors.')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trainer = BlackjackTrainer(errors_only=args.errors)
    trainer.run()


if __name__ == '__main__':
    main()
