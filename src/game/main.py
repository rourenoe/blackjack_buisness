import pygame
import sys
from pathlib import Path
from typing import List, Tuple
from game.multiple_deck import Multiple_deck
from game.card import Card

# Initialize Pygame
pygame.init()

# Constants
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800
CARD_WIDTH = 100
CARD_HEIGHT = 145
BUTTON_WIDTH = 200
BUTTON_HEIGHT = 50
FPS = 60

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 128, 0)
RED = (255, 0, 0)
GRAY = (128, 128, 128)

class BlackjackGame:
    screen: pygame.Surface
    clock: pygame.time.Clock
    font: pygame.font.Font
    deck: Multiple_deck
    player_hand: List[Card]
    dealer_hand: List[Card]
    game_over: bool
    player_stand: bool
    hit_button: pygame.Rect
    stand_button: pygame.Rect
    new_game_button: pygame.Rect

    def __init__(self) -> None:
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Blackjack")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        
        # Game state
        self.deck = Multiple_deck()
        self.player_hand = []
        self.dealer_hand = []
        self.game_over = False
        self.player_stand = False
        
        # Statistics
        self.total_games = 0
        self.wins = 0
        self.losses = 0
        self.pushes = 0
        
        # Buttons
        self.hit_button = pygame.Rect(50, WINDOW_HEIGHT - 100, BUTTON_WIDTH, BUTTON_HEIGHT)
        self.stand_button = pygame.Rect(300, WINDOW_HEIGHT - 100, BUTTON_WIDTH, BUTTON_HEIGHT)
        self.new_game_button = pygame.Rect(550, WINDOW_HEIGHT - 100, BUTTON_WIDTH, BUTTON_HEIGHT)
        
        self.start_new_game()

    def start_new_game(self) -> None:
        self.deck = Multiple_deck(num_decks=6)
        self.player_hand = []
        self.dealer_hand = []
        self.game_over = False
        self.player_stand = False
        
        # Initial deal
        self.player_hand.append(self.deck.deal_card())
        self.dealer_hand.append(self.deck.deal_card())
        self.player_hand.append(self.deck.deal_card())
        self.dealer_hand.append(self.deck.deal_card())

    def calculate_score(self, hand: List[Card]) -> int:
        score = 0
        aces = 0
        
        for card in hand:
            if card.rank == 'A':
                aces += 1
            else:
                score += card.value()
        
        # Add aces
        for _ in range(aces):
            if score + 11 <= 21:
                score += 11
            else:
                score += 1
                
        return score

    def dealer_play(self) -> None:
        while self.calculate_score(self.dealer_hand) < 17:
            self.dealer_hand.append(self.deck.deal_card())

    def check_winner(self) -> str:
        player_score = self.calculate_score(self.player_hand)
        dealer_score = self.calculate_score(self.dealer_hand)
        
        self.total_games += 1
        
        if player_score > 21:
            self.losses += 1
            return "Dealer wins! Player busted!"
        elif dealer_score > 21:
            self.wins += 1
            return "Player wins! Dealer busted!"
        elif player_score > dealer_score:
            self.wins += 1
            return "Player wins!"
        elif dealer_score > player_score:
            self.losses += 1
            return "Dealer wins!"
        else:
            self.pushes += 1
            return "Push! It's a tie!"

    def draw_card(self, card: Card, x: int, y: int) -> None:
        try:

            # Get the current file's directory and construct the path to the image
            current_dir = Path(__file__).parent
            image_path = current_dir.parent.parent / card.path_to_image
            card_image = pygame.image.load(str(image_path))
            card_image = pygame.transform.scale(card_image, (CARD_WIDTH, CARD_HEIGHT))
            self.screen.blit(card_image, (x, y))
        except (pygame.error, FileNotFoundError):
            # Fallback to drawing a basic card if image is not found
            card_rect = pygame.Rect(x, y, CARD_WIDTH, CARD_HEIGHT)
            pygame.draw.rect(self.screen, WHITE, card_rect)
            pygame.draw.rect(self.screen, BLACK, card_rect, 2)
            
            # Draw card text as fallback
            text = self.font.render(f"{card.rank} {card.suit[0]}", True, BLACK)
            text_rect = text.get_rect(center=card_rect.center)
            self.screen.blit(text, text_rect)

    def draw(self) -> None:
        self.screen.fill(GREEN)
        
        # Draw dealer's cards
        for i, card in enumerate(self.dealer_hand):
            if i == 0 and not self.player_stand and not self.game_over:
                # Draw face down card
                card_rect = pygame.Rect(50 + i * (CARD_WIDTH + 20), 50, CARD_WIDTH, CARD_HEIGHT)
                pygame.draw.rect(self.screen, RED, card_rect)
                pygame.draw.rect(self.screen, BLACK, card_rect, 2)
            else:
                self.draw_card(card, 50 + i * (CARD_WIDTH + 20), 50)
        
        # Draw player's cards
        for i, card in enumerate(self.player_hand):
            self.draw_card(card, 50 + i * (CARD_WIDTH + 20), WINDOW_HEIGHT - 250)
        
        # Draw scores
        player_score = self.calculate_score(self.player_hand)
        player_text = self.font.render(f"Player Score: {player_score}", True, WHITE)
        self.screen.blit(player_text, (50, WINDOW_HEIGHT - 300))
        
        if self.player_stand or self.game_over:
            dealer_score = self.calculate_score(self.dealer_hand)
            dealer_text = self.font.render(f"Dealer Score: {dealer_score}", True, WHITE)
            self.screen.blit(dealer_text, (50, 20))
        
        # Draw buttons
        pygame.draw.rect(self.screen, GRAY, self.hit_button)
        hit_text = self.font.render("Hit", True, BLACK)
        hit_rect = hit_text.get_rect(center=self.hit_button.center)
        self.screen.blit(hit_text, hit_rect)
        
        pygame.draw.rect(self.screen, GRAY, self.stand_button)
        stand_text = self.font.render("Stand", True, BLACK)
        stand_rect = stand_text.get_rect(center=self.stand_button.center)
        self.screen.blit(stand_text, stand_rect)
        
        pygame.draw.rect(self.screen, GRAY, self.new_game_button)
        new_game_text = self.font.render("New Game", True, BLACK)
        new_game_rect = new_game_text.get_rect(center=self.new_game_button.center)
        self.screen.blit(new_game_text, new_game_rect)
        
        # Draw game over message
        if self.game_over:
            result = self.check_winner()
            result_text = self.font.render(result, True, WHITE)
            result_rect = result_text.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2))
            self.screen.blit(result_text, result_rect)
            self.game_over = False  
        
        # Draw statistics
        winrate = (self.wins / self.total_games * 100) if self.total_games > 0 else 0
        stats_text = self.font.render(
            f"Games: {self.total_games} | Wins: {self.wins} | Losses: {self.losses} | Pushes: {self.pushes} | Winrate: {winrate:.1f}%",
            True, WHITE
        )
        self.screen.blit(stats_text, (WINDOW_WIDTH // 2 - stats_text.get_width() // 2, 10))

    def run(self) -> None:
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    mouse_pos = event.pos
                    
                    if not self.game_over and not self.player_stand:
                        if self.hit_button.collidepoint(mouse_pos):
                            self.player_hand.append(self.deck.deal_card())
                            if self.calculate_score(self.player_hand) > 21:
                                self.game_over = True
                                self.player_stand = True
                                
                        elif self.stand_button.collidepoint(mouse_pos):
                            self.player_stand = True
                            self.dealer_play()
                            self.game_over = True
                            
                    if self.new_game_button.collidepoint(mouse_pos):
                        self.start_new_game()
            
            self.draw()
            pygame.display.flip()
            self.clock.tick(FPS)

        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    game = BlackjackGame()
    game.run()
