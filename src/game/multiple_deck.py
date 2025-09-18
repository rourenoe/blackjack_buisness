from game.deck import Deck


class Multiple_deck(Deck):
    def __init__(self, num_decks=6):
        self.cards = []
        for _ in range(num_decks):
            deck = Deck()
            self.cards.extend(deck.cards)
        self.shuffle()