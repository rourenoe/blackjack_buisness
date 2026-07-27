from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from PIL import Image, ImageDraw, ImageFont

from game.strategy import build_strategy_grid

OUTPUT_DIR = Path(__file__).resolve().parents[2]
COLORS = {
    'stand': {'bg': (25, 71, 140, 220), 'text': (255, 255, 255)},
    'hit': {'bg': (210, 145, 23, 220), 'text': (0, 0, 0)},
    'double': {'bg': (210, 70, 70, 220), 'text': (255, 255, 255)},
    'split': {'bg': (190, 95, 155, 220), 'text': (255, 255, 255)},
    'default': {'bg': (35, 35, 45, 255), 'text': (240, 240, 240)},
}
HEADER_BG = (20, 25, 40, 255)
GRID_LINE = (95, 95, 120, 255)
TEXT_COLOR = (240, 240, 240, 255)
TITLE_COLOR = (245, 245, 255, 255)

CELL_WIDTH = 160
LABEL_COL_WIDTH = 150
ROW_HEIGHT = 96
PADDING = 16
TITLE_HEIGHT = 100
FOOTER_HEIGHT = 70
FONT_SIZE = 16
TITLE_FONT_SIZE = 32
SMALL_FONT_SIZE = 14


def load_fonts():
    try:
        font = ImageFont.truetype('arial.ttf', FONT_SIZE)
        title_font = ImageFont.truetype('arial.ttf', TITLE_FONT_SIZE)
        small_font = ImageFont.truetype('arial.ttf', SMALL_FONT_SIZE)
    except OSError:
        font = ImageFont.load_default()
        title_font = font
        small_font = font
    return title_font, font, small_font


def format_percent_value(value: float) -> str:
    return str(int(round(value * 100)))


def draw_table_image(name: str, title: str, rows: List[List[Dict[str, object]]], dealer_upcards: List[str]) -> Path:
    title_font, font, small_font = load_fonts()
    num_cols = len(dealer_upcards) + 1
    num_rows = len(rows) + 1
    width = LABEL_COL_WIDTH + CELL_WIDTH * len(dealer_upcards) + PADDING * 2
    height = TITLE_HEIGHT + ROW_HEIGHT * num_rows + FOOTER_HEIGHT + PADDING * 2

    image = Image.new('RGBA', (width, height), (13, 18, 30, 255))
    draw = ImageDraw.Draw(image)

    # Title
    title_text = f'{title}'
    draw.text((PADDING, PADDING), title_text, font=title_font, fill=TITLE_COLOR)
    subtitle = 'Best action + probabilities (Win / Loss / EV)'
    draw.text((PADDING, PADDING + TITLE_FONT_SIZE + 8), subtitle, font=small_font, fill=(200, 200, 220, 255))

    top = PADDING + TITLE_HEIGHT
    left = PADDING

    # header row background
    draw.rectangle([left, top, width - PADDING, top + ROW_HEIGHT], fill=HEADER_BG)
    # first label header
    label_header = 'Player / Dealer'
    draw.text((left + 8, top + (ROW_HEIGHT - FONT_SIZE) / 2), label_header, font=font, fill=TEXT_COLOR)

    # dealer headers
    for idx, card in enumerate(dealer_upcards):
        x = left + LABEL_COL_WIDTH + idx * CELL_WIDTH
        draw.rectangle([x, top, x + CELL_WIDTH, top + ROW_HEIGHT], outline=GRID_LINE, width=1)
        draw.text((x + 8, top + (ROW_HEIGHT - FONT_SIZE) / 2), card, font=font, fill=TEXT_COLOR)

    # rows
    y = top + ROW_HEIGHT
    for row in rows:
        draw.rectangle([left, y, width - PADDING, y + ROW_HEIGHT], outline=GRID_LINE, width=1)
        # player label
        row_label = str(row[0]['label'])
        draw.text((left + 8, y + 10), row_label, font=font, fill=TEXT_COLOR)

        for col_idx, cell in enumerate(row):
            x = left + LABEL_COL_WIDTH + col_idx * CELL_WIDTH
            best_action = cell['best_action']
            bg = COLORS.get(best_action, COLORS['default'])['bg']
            text_color = COLORS.get(best_action, COLORS['default'])['text']
            draw.rectangle([x, y, x + CELL_WIDTH, y + ROW_HEIGHT], fill=bg, outline=GRID_LINE, width=1)

            # draw cell contents
            text_x = x + 8
            text_y = y + 8
            best_text = f"Best: {cell['best_action'].capitalize()}"
            draw.text((text_x, text_y), best_text, font=font, fill=text_color)
            text_y += FONT_SIZE + 6
            for action in ['stand', 'hit', 'double', 'split']:
                result = cell['actions'].get(action)
                if result is None:
                    continue
                line = f"{action[0].upper()}: {format_percent_value(result['win_prob'])}/{format_percent_value(result['loss_prob'])} {result['ev']:.2f}"
                draw.text((text_x, text_y), line, font=small_font, fill=text_color)
                text_y += SMALL_FONT_SIZE + 2

        y += ROW_HEIGHT

    # footer
    footer_text = 'Generated from the blackjack strategy grid. Colors indicate the best action for each scenario.'
    draw.text((PADDING, height - FOOTER_HEIGHT + 16), footer_text, font=small_font, fill=(200, 200, 210, 255))

    out_path = OUTPUT_DIR / f'{name}.png'
    image.save(out_path)
    return out_path


def main() -> None:
    grid = build_strategy_grid()
    dealer_upcards = grid['dealer_upcards']
    outputs = []

    outputs.append(draw_table_image('hard_totals', 'Hard Totals', grid['hard_totals'], dealer_upcards))
    outputs.append(draw_table_image('soft_totals', 'Soft Totals', grid['soft_totals'], dealer_upcards))
    outputs.append(draw_table_image('pairs', 'Pairs', grid['pairs'], dealer_upcards))

    print('Generated PNGs:')
    for path in outputs:
        print(path)


if __name__ == '__main__':
    main()
