from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from game.strategy import build_strategy_grid, format_percentage

OUTPUT_FILE = Path(__file__).resolve().parents[2] / 'strategy_table.html'

CATEGORY_TITLES = {
    'hard_totals': 'Hard Totals',
    'soft_totals': 'Soft Totals',
    'pairs': 'Pairs',
}

ACTION_LABELS = {
    'stand': 'Stand',
    'hit': 'Hit',
    'double': 'Double',
    'split': 'Split',
}

ACTION_ORDER = ['stand', 'hit', 'double', 'split']

STYLE = """
body {
  margin: 0;
  min-height: 100vh;
  background: radial-gradient(circle at top left, #16232e 0%, #0b1420 100%);
  color: #f3f7fb;
  font-family: Inter, 'Segoe UI', sans-serif;
  padding: 24px;
}
.container {
  max-width: 1400px;
  margin: 0 auto;
}
h1, h2 {
  margin: 24px 0 12px;
}
.strategy-section {
  margin-bottom: 40px;
}
table {
  width: 100%;
  border-collapse: collapse;
  margin-top: 16px;
}
th, td {
  padding: 10px 12px;
  border: 1px solid rgba(255, 255, 255, 0.14);
  text-align: left;
  vertical-align: top;
}
th {
  background: rgba(255, 255, 255, 0.08);
}
.cell {
  min-width: 220px;
  white-space: normal;
}
.cell-header {
  display: flex;
  justify-content: space-between;
  opacity: 0.75;
  margin-bottom: 4px;
  font-size: 0.92rem;
}
.best-action {
  font-weight: 700;
  margin-bottom: 6px;
}
.cell.best-stand { background: rgba(21, 85, 162, 0.16); border-color: rgba(21, 85, 162, 0.35); }
.cell.best-hit { background: rgba(255, 200, 54, 0.14); border-color: rgba(255, 200, 54, 0.35); }
.cell.best-double { background: rgba(255, 80, 80, 0.16); border-color: rgba(255, 80, 80, 0.35); }
.cell.best-split { background: rgba(230, 105, 180, 0.14); border-color: rgba(230, 105, 180, 0.35); }
.action-row {
  display: flex;
  justify-content: space-between;
  font-size: 0.95rem;
  padding: 4px 0;
}
.action-row span {
  display: inline-block;
  min-width: 52px;
}
.action-name {
  width: 72px;
}
.best {
  color: #82ff9e;
}
.stand { color: #79b8ff; }
.hit { color: #ffca5e; }
.double { color: #ffd47c; }
.split { color: #ff8ccc; }
.footer {
  margin-top: 24px;
  font-size: 0.95rem;
  opacity: 0.86;
}
"""


def render_action_row(action: str, result: Dict[str, Any], is_best: bool) -> str:
    label = ACTION_LABELS.get(action, action.title())
    ev = result['ev']
    win = format_percentage(result['win_prob'])
    push = format_percentage(result['push_prob'])
    loss = format_percentage(result['loss_prob'])
    best_class = ' best' if is_best else ''
    return (
        f"<div class='action-row {action}{best_class}'>"
        f"<span class='action-name'>{label}</span>"
        f"<span>{win}</span>"
        f"<span>{push}</span>"
        f"<span>{loss}</span>"
        f"<span>{ev:.3f}</span>"
        f"</div>"
    )


def render_cell(cell: Dict[str, Any]) -> str:
    best_action = cell['best_action']
    rows = []
    for action in ACTION_ORDER:
        result = cell['actions'].get(action)
        if result is None:
            continue
        rows.append(render_action_row(action, result, action == best_action))
    header = (
        "<div class='cell-header'>"
        "<span>Action</span>"
        "<span>Win</span>"
        "<span>Push</span>"
        "<span>Loss</span>"
        "<span>EV</span>"
        "</div>"
    )
    return (
        f"<td class='cell best-{best_action}'>"
        f"<div class='best-action'>Best: <span class='{best_action}'>{ACTION_LABELS.get(best_action, best_action)}</span></div>"
        f"{header}"
        f"{''.join(rows)}"
        f"</td>"
    )


def render_table(title: str, rows: List[List[Dict[str, Any]]], dealer_upcards: List[str]) -> str:
    header_cells = ''.join(f"<th>{card}</th>" for card in dealer_upcards)
    body_rows = []
    for row in rows:
        first_label = row[0]['label']
        cells = ''.join(render_cell(cell) for cell in row)
        body_rows.append(f"<tr><th>{first_label}</th>{cells}</tr>")
    return (
        f"<section class='strategy-section'><h2>{title}</h2>"
        f"<table><thead><tr><th>Player / Dealer</th>{header_cells}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody></table></section>"
    )


def generate_html(output_path: Path) -> None:
    grid = build_strategy_grid()
    dealer_upcards = grid['dealer_upcards']
    sections_html = []
    for category_key in ['hard_totals', 'soft_totals', 'pairs']:
        title = CATEGORY_TITLES[category_key]
        sections_html.append(render_table(title, grid[category_key], dealer_upcards))

    html = (
        '<!DOCTYPE html>'
        '<html lang="fr">'
        '<head>'
        '<meta charset="UTF-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1.0">'
        '<title>Blackjack Strategy Table</title>'
        f'<style>{STYLE}</style>'
        '</head>'
        '<body>'
        '<div class="container">'
        '<h1>Blackjack Strategy Table</h1>'
        "<p>Cet outil calcule pour chaque situation initiale la meilleure action basée sur l'espérance de gain, en affichant également les probabilités de victoire, push et défaite pour chaque action.</p>"
        f'{"".join(sections_html)}'
        '<div class="footer">' 
        '<p>Règle standard : 6 jeux, le croupier reste sur soft 17. Double autorisé sur les deux premières cartes ; split évalué sans resplit.</p>'
        '</div>'
        '</div>'
        '</body>'
        '</html>'
    )
    output_path.write_text(html, encoding='utf-8')


if __name__ == '__main__':
    output_path = OUTPUT_FILE
    generate_html(output_path)
    print(f'Generated strategy table at {output_path}')
