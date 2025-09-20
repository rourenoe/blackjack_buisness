from game.deck import Deck
import random


class Multiple_deck(Deck):
    def __init__(self, num_decks=6):
        self.cards = []
        for _ in range(num_decks):
            deck = Deck()
            self.cards.extend(deck.cards)
        self.shuffle()

    def draw_until_exact(self, target):
        """
        Tire des cartes jusqu'à atteindre exactement le nombre cible.
        À chaque tirage, ne considère que les cartes qui peuvent nous permettre d'atteindre le nombre exact.
        
        Args:
            target (int): Le nombre exact à atteindre
            
        Returns:
            list[Card]: Liste des cartes tirées pour atteindre le nombre exact
        """
        drawn_cards = []
        remaining_target = target
        current_cards = self.cards.copy()  # On travaille sur une copie pour ne pas modifier le deck original
        
        while remaining_target > 0:
            # Filtrer les cartes valides en tenant compte que les As peuvent valoir 1
            valid_cards = []
            for card in current_cards:
                if card.rank == 'A':
                    # Pour un As, on considère la valeur 1
                    if 1 <= remaining_target:
                        valid_cards.append(card)
                else:
                    if card.value() <= remaining_target:
                        valid_cards.append(card)
            
            if not valid_cards:
                raise ValueError(f"Impossible d'atteindre exactement {target} avec les cartes restantes")
            
            # Choisir une carte au hasard parmi les cartes valides
            chosen_card = random.choice(valid_cards)
            drawn_cards.append(chosen_card)
            current_cards.remove(chosen_card)
            
            # Pour un As, on utilise la valeur 1 si nécessaire
            card_value = 1 if chosen_card.rank == 'A' else chosen_card.value()
            
            # Mettre à jour la cible restante avec la valeur appropriée
            remaining_target -= card_value
            
        # Retirer les cartes tirées du deck principal
        for card in drawn_cards:
            self.cards.remove(card)
            
        return drawn_cards