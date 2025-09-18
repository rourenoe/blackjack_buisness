# Blackjack Game

A beautiful and interactive Blackjack game implemented in Python using Pygame. Play against the dealer with a realistic card interface and smooth gameplay.

![Blackjack Game Screenshot](images/cards/back_dark.png)

## Features

- 🎮 Graphical user interface with beautiful card designs
- 🎲 Multiple deck support (default: 6 decks)
- 🃏 Realistic card graphics for all suits and ranks
- 🎯 Standard Blackjack rules implementation
- 🎨 Smooth animations and intuitive controls
- 🔄 New game functionality
- 💫 Fallback text-based cards if images are unavailable

## Requirements

- Python 3.11 or higher
- Pygame

## Installation

1. Clone the repository:
```bash
git clone https://github.com/rourennoe/blackjack_buisness.git
cd blackjack_buisness
```

2. Install the required dependencies:
```bash
pip install pygame
```

## How to Play

1. Run the game:
```bash
python main.py
```

2. Game Controls:
   - Click "Hit" to draw another card
   - Click "Stand" to keep your current hand
   - Click "New Game" to start a fresh game

3. Game Rules:
   - Try to get as close to 21 as possible without going over
   - Face cards (J, Q, K) are worth 10
   - Aces are worth 11 or 1, whichever benefits you more
   - Beat the dealer's hand to win!

## Project Structure

```
blackjack_buisness/
├── main.py              # Main game loop and GUI
├── card.py             # Card class implementation
├── deck.py             # Single deck implementation
├── multiple_deck.py    # Multiple deck handling
└── images/
    └── cards/          # Card image assets
```

## Classes

- `Card`: Represents a playing card with suit, rank, and image
- `Deck`: Manages a standard 52-card deck
- `Multiple_deck`: Handles multiple decks for casino-style play
- `BlackjackGame`: Main game logic and GUI implementation

## Credits

Card images are from a modified version of Grafik-fighter's deck, optimized for game projects. The original design can be found [here](https://www.sketchappsources.com/free-source/3060-cards-deck-template-sketch-freebie-resource.html).

## Contributing

Feel free to contribute to this project by:
1. Forking the repository
2. Creating your feature branch
3. Committing your changes
4. Pushing to the branch
5. Opening a Pull Request

## License

This project is open source and available under the MIT License.