import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List
import random

class DQN(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 256):
        """Initialise le réseau de neurones pour le DQN
        
        Args:
            input_dim: Dimension de l'espace d'état
            output_dim: Dimension de l'espace d'action
            hidden_dim: Dimension des couches cachées
        """
        super(DQN, self).__init__()
        
        # Réseau plus simple mais robuste
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc4 = nn.Linear(hidden_dim // 2, output_dim)
        
        # Layer Normalization au lieu de Batch Normalization
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim // 2)
        
        # Initialisation des poids
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass du réseau
        
        Args:
            x: Tensor d'entrée représentant l'état
            
        Returns:
            Tensor de sortie représentant les Q-values pour chaque action
        """
        # Utilisation de LeakyReLU pour éviter les neurones morts
        x = F.leaky_relu(self.ln1(self.fc1(x)))
        x = F.dropout(x, p=0.1, training=self.training)
        x = F.leaky_relu(self.ln2(self.fc2(x)))
        x = F.dropout(x, p=0.1, training=self.training)
        x = F.leaky_relu(self.ln3(self.fc3(x)))
        return self.fc4(x)

class DQNAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,  # Augmentation de la capacité du réseau
        learning_rate: float = 0.0005,  # Taux d'apprentissage plus faible pour plus de stabilité
        gamma: float = 0.95,  # Facteur de réduction légèrement réduit pour favoriser les récompenses immédiates
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,  # Augmentation de l'exploration minimale
        epsilon_decay: float = 0.9995,  # Décroissance plus lente de l'exploration
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """Initialise l'agent DQN
        
        Args:
            state_dim: Dimension de l'espace d'état
            action_dim: Dimension de l'espace d'action
            hidden_dim: Dimension des couches cachées du réseau
            learning_rate: Taux d'apprentissage
            gamma: Facteur de réduction pour les récompenses futures
            epsilon_start: Valeur initiale d'epsilon pour l'exploration
            epsilon_end: Valeur minimale d'epsilon
            epsilon_decay: Facteur de décroissance d'epsilon
            device: Dispositif sur lequel exécuter les calculs (CPU/GPU)
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.gamma = gamma
        
        # Réseaux de neurones (principal et cible)
        self.policy_net = DQN(state_dim, action_dim, hidden_dim).to(device)
        self.target_net = DQN(state_dim, action_dim, hidden_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Paramètres d'exploration
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Statistiques
        self.training_steps = 0
        self.wins = 0
        self.losses = 0
        self.draws = 0
        self.total_games = 0
        
    def select_action(self, state: np.ndarray, valid_actions: List[int], training: bool = True) -> int:
        """Sélectionne une action en utilisant la politique epsilon-greedy
        
        Args:
            state: État actuel
            valid_actions: Liste des actions valides
            training: Si True, utilise epsilon-greedy, sinon utilise la meilleure action
            
        Returns:
            Action sélectionnée
        """
        if training and random.random() < self.epsilon:
            return random.choice(valid_actions)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            
            # Masquer les actions invalides avec -inf
            mask = torch.ones(self.action_dim) * float('-inf')
            mask[valid_actions] = 0
            q_values += mask.to(self.device)
            
            return q_values.argmax().item()
    
    def update_epsilon(self):
        """Met à jour le paramètre epsilon pour l'exploration"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def train_step(self, state, action, reward, next_state, done, valid_next_actions):
        """Effectue une étape d'apprentissage
        
        Args:
            state: État actuel ou batch d'états
            action: Action effectuée ou batch d'actions
            reward: Récompense reçue ou batch de récompenses
            next_state: État suivant ou batch d'états suivants
            done: Si l'épisode est terminé ou batch de drapeaux
            valid_next_actions: Actions valides dans l'état suivant
        """
        # Convertir en numpy arrays si ce n'est pas déjà fait
        if isinstance(state, list):
            state = np.array(state)
        if isinstance(next_state, list):
            next_state = np.array(next_state)
        if isinstance(action, list):
            action = np.array(action)
        if isinstance(reward, list):
            reward = np.array(reward)
        if isinstance(done, list):
            done = np.array(done)
            
        # Convertir en tensors
        state = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        action = torch.LongTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        done = torch.FloatTensor(done).to(self.device)
        
        # Ajouter une dimension si nécessaire
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if next_state.dim() == 1:
            next_state = next_state.unsqueeze(0)
        if action.dim() == 0:
            action = action.unsqueeze(0)
        if reward.dim() == 0:
            reward = reward.unsqueeze(0)
        if done.dim() == 0:
            done = done.unsqueeze(0)
        
        # Calculer la Q-value actuelle
        q_values = self.policy_net(state)
        current_q_value = q_values.gather(1, action.unsqueeze(1))
        
        # Calculer la Q-value cible
        with torch.no_grad():
            # Masquer les actions invalides pour le prochain état
            next_q_values = self.target_net(next_state)

            # Créer un masque pour les actions valides
            mask = torch.full_like(next_q_values, float('-inf'))
            for i, actions in enumerate(valid_next_actions):
                if len(actions) > 0:
                    mask[i, actions] = 0
            next_q_values = next_q_values + mask

            # Prendre la meilleure Q pour l'état suivant
            next_q_value = next_q_values.max(1)[0]

            # Remplacer les valeurs non-finies (p.ex. -inf) par 0 pour éviter NaN
            next_q_value = torch.where(torch.isfinite(next_q_value), next_q_value, torch.zeros_like(next_q_value))

            # Pour les transitions terminales, forcer la Q suivante à 0
            next_q_value = torch.where(done.bool(), torch.zeros_like(next_q_value), next_q_value)

            target_q_value = reward + (1 - done) * self.gamma * next_q_value
        
        # Calculer la perte et mettre à jour le réseau
        loss = F.smooth_l1_loss(current_q_value, target_q_value.unsqueeze(1))
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.training_steps += 1
        
        # Mettre à jour le réseau cible périodiquement
        if self.training_steps % 100 == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            
        return loss.item()
    
    def update_stats(self, reward):
        """Met à jour les statistiques de l'agent
        
        Args:
            reward: Récompense reçue à la fin de la partie
        """
        self.total_games += 1
        if reward > 0:
            self.wins += 1
        elif reward < 0:
            self.losses += 1
        else:
            self.draws += 1
    
    def get_win_rate(self) -> float:
        """Calcule le taux de victoire de l'agent
        
        Returns:
            Taux de victoire en pourcentage
        """
        if self.total_games == 0:
            return 0.0
        return (self.wins / self.total_games) * 100
    
    def save(self, path: str):
        """Sauvegarde le modèle
        
        Args:
            path: Chemin où sauvegarder le modèle
        """
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_steps': self.training_steps,
            'stats': {
                'wins': self.wins,
                'losses': self.losses,
                'draws': self.draws,
                'total_games': self.total_games
            }
        }, path)
    
    def load(self, path: str):
        """Charge un modèle sauvegardé
        
        Args:
            path: Chemin vers le modèle à charger
        """
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.training_steps = checkpoint['training_steps']
        stats = checkpoint['stats']
        self.wins = stats['wins']
        self.losses = stats['losses']
        self.draws = stats['draws']
        self.total_games = stats['total_games']