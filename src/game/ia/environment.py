import numpy as np
from typing import Tuple, List
from game.multiple_deck import Multiple_deck
from game.card import Card

class BlackjackEnv:
    def __init__(self, num_decks: int = 6):
        self.deck = Multiple_deck(num_decks=num_decks)
        self.player_hand = []
        self.dealer_hand = []
        self.done = False
        self.observation_space_dim = 6  # [player_sum, dealer_card, has_usable_ace, num_cards, bust_prob, win_prob]
        self.action_space_n = 2  # [hit, stand]
        
    def reset(self) -> np.ndarray:
        """Reset l'environnement pour une nouvelle partie"""
        self.deck = Multiple_deck(num_decks=6)
        self.player_hand = []
        self.dealer_hand = []
        self.done = False
        
        # Distribution initiale
        self.player_hand.append(self.deck.deal_card())
        self.dealer_hand.append(self.deck.deal_card())
        self.player_hand.append(self.deck.deal_card())
        self.dealer_hand.append(self.deck.deal_card())
        
        return self._get_state()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """Exécute une action dans l'environnement
        
        Args:
            action (int): 0 pour hit, 1 pour stand
            
        Returns:
            tuple: (nouvel_état, récompense, terminé, info_supplémentaire)
        """
        assert action in [0, 1], "Action invalide"
        
        reward = 0.0
        info = {}
        player_score_before = self._calculate_hand_value(self.player_hand)
        dealer_card = self.dealer_hand[0].value()

        # Utiliser uniquement la récompense finale (+1/-1/0) et un shaping potentiel
        # basé sur la différence de probabilité de gain estimée (phi). Cela évite
        # des signaux heuristiques contradictoires qui font dégrader la politique.
        shaping_coeff = 0.1

        if action == 0:  # Hit
            # Appliquer l'action
            self.player_hand.append(self.deck.deal_card())
            player_score = self._calculate_hand_value(self.player_hand)

            # Résultats terminaux
            if player_score > 21:
                reward = -1.0
                self.done = True
            elif player_score == 21:
                reward = 1.0
                self.done = True
            else:
                # Shaping potentiel: difference de probabilité de gain
                new_win_prob = self._calculate_win_probability(player_score, dealer_card)
                old_win_prob = self._calculate_win_probability(player_score_before, dealer_card)
                reward = shaping_coeff * (new_win_prob - old_win_prob)

        else:  # Stand
            # Jouer la main du dealer et déterminer le résultat final
            self._play_dealer()
            dealer_score = self._calculate_hand_value(self.dealer_hand)

            if dealer_score > 21:
                reward = 1.0
            else:
                if player_score_before > dealer_score:
                    reward = 1.0
                elif player_score_before < dealer_score:
                    reward = -1.0
                else:
                    reward = 0.0  # Push

            self.done = True

        return self._get_state(), float(reward), self.done, info
    
    def _play_dealer(self):
        """Joue la main du dealer selon les règles standard"""
        while self._calculate_hand_value(self.dealer_hand) < 17:
            self.dealer_hand.append(self.deck.deal_card())
    
    def _get_state(self) -> np.ndarray:
        """Retourne l'état actuel du jeu avec des informations plus détaillées"""
        player_sum = self._calculate_hand_value(self.player_hand)
        dealer_card = self.dealer_hand[0].value()
        has_usable_ace = self._has_usable_ace(self.player_hand)
        num_cards = len(self.player_hand)
        bust_probability = self._calculate_bust_probability(player_sum)
        win_probability = self._calculate_win_probability(player_sum, dealer_card)
        
        return np.array([
            player_sum / 21.0,  # Score du joueur normalisé
            dealer_card / 11.0,  # Carte du dealer normalisée
            float(has_usable_ace),
            num_cards / 5.0,  # Nombre de cartes normalisé
            bust_probability,  # Probabilité de dépasser 21 à la prochaine carte
            win_probability  # Probabilité de gagner avec le score actuel
        ])
    
    def _calculate_hand_value(self, hand: List[Card]) -> int:
        """Calcule la valeur d'une main"""
        value = 0
        aces = 0
        
        for card in hand:
            if card.rank == 'A':
                aces += 1
            else:
                value += card.value()
        
        # Ajoute les as
        for _ in range(aces):
            if value + 11 <= 21:
                value += 11
            else:
                value += 1
                
        return value

    def _calculate_bust_probability(self, current_sum: int) -> float:
        """Calcule la probabilité de dépasser 21 à la prochaine carte"""
        if current_sum >= 21:
            return 1.0
        
        # Calculer le nombre de cartes qui feraient dépasser 21
        safe_value = 21 - current_sum
        bust_cards = 0
        total_cards = 52  # Approximation pour simplifier
        
        for value in range(2, 12):  # 2-10 et As (qui peut valoir 1)
            if value > safe_value:
                if value == 11:  # As
                    if safe_value < 1:  # Même en comptant l'As comme 1, ça bust
                        bust_cards += 4
                else:
                    bust_cards += 4 if value <= 10 else 12  # 12 pour J,Q,K qui valent 10
                    
        return bust_cards / total_cards
    
    def _calculate_win_probability(self, player_sum: int, dealer_card: int) -> float:
        """Calcule une estimation de la probabilité de gagner avec le score actuel"""
        if player_sum > 21:
            return 0.0
        if player_sum == 21:
            return 1.0
            
        # Probabilité de gagner basée sur la stratégie de base du blackjack
        if player_sum >= 17:
            # Bonnes chances de gagner avec 17+, sauf si le dealer a un As ou une carte forte
            if dealer_card >= 7:
                return 0.5
            return 0.8
        elif player_sum >= 13:
            # Position moyenne, dépend beaucoup de la carte du dealer
            if dealer_card >= 7:
                return 0.3
            return 0.6
        else:
            # Main faible, mais pas sans espoir
            return 0.4
    
    def _has_usable_ace(self, hand: List[Card]) -> bool:
        """Détermine si la main contient un as utilisable comme 11"""
        value = 0
        aces = 0
        
        for card in hand:
            if card.rank == 'A':
                aces += 1
            else:
                value += card.value()
        
        # Si on peut utiliser au moins un as comme 11 sans dépasser 21
        return aces > 0 and value + 11 + (aces - 1) <= 21
    
    def get_valid_actions(self) -> List[int]:
        """Retourne la liste des actions valides dans l'état actuel"""
        if self.done:
            return []
        return [0, 1]  # [hit, stand]
    
    @property
    def state_shape(self) -> tuple:
        """Retourne la forme de l'espace d'état"""
        return (self.observation_space_dim,)