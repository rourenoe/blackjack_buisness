class Card:
    
    def __init__(self, suit, rank):
        self.suit = suit
        self.rank = rank
        self.path_to_image = f"images/cards/{self.suit}_{self.rank}.png"
        
    def __str__(self):
        return f"{self.rank} of {self.suit}"
    
    def value(self):
        if self.rank in ['J', 'Q', 'K']:
            return 10
        elif self.rank == 'A':
            return 11  # Initially consider Ace as 11
        else:
            return int(self.rank)