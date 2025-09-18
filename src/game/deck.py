from game.card import Card
import random


class Deck:
    def __init__(self):
        #init a 52 card deck
        suits = ['hearts', 'diamonds', 'clubs', 'spades']
        ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
        self.cards = [Card(suit, rank) for suit in suits for rank in ranks]

    def shuffle(self):
        random.shuffle(self.cards)

    def deal_card(self):
        if len(self.cards) == 0:
            raise ValueError("No cards left in the deck")
        return self.cards.pop()