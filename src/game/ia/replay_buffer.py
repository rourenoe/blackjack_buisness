import numpy as np
from collections import deque
import random
from typing import Tuple, List

class ReplayBuffer:
    def __init__(self, capacity: int):
        """Initialise le buffer avec une capacité maximale
        
        Args:
            capacity: Nombre maximum d'expériences à stocker
        """
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state: np.ndarray, action: int, reward: float, 
            next_state: np.ndarray, done: bool, valid_next_actions: List[int]):
        """Ajoute une expérience au buffer
        
        Args:
            state: État actuel
            action: Action effectuée
            reward: Récompense reçue
            next_state: État suivant
            done: Si l'épisode est terminé
            valid_next_actions: Actions valides dans l'état suivant
        """
        self.buffer.append((state, action, reward, next_state, done, valid_next_actions))
    
    def sample(self, batch_size: int) -> Tuple:
        """Échantillonne un batch d'expériences du buffer
        
        Args:
            batch_size: Taille du batch à échantillonner
            
        Returns:
            Tuple contenant les états, actions, récompenses, états suivants,
            drapeaux de fin et actions valides pour les états suivants
        """
        experiences = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones, valid_next_actions = zip(*experiences)
        
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones), list(valid_next_actions))
    
    def __len__(self) -> int:
        """Retourne le nombre d'expériences dans le buffer"""
        return len(self.buffer)