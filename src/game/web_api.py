from __future__ import annotations

import hashlib
import os
import secrets
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from game.strategy import compute_hand_value, evaluate_state

APP_ROOT = Path(__file__).resolve().parents[2]
WEB_DIR = APP_ROOT / "web"
DB_PATH = APP_ROOT / "blackjack_training.db"
REQUESTS_PER_MINUTE = 120

HARD_HANDS = {
    5: ("2", "3"),
    6: ("2", "4"),
    7: ("2", "5"),
    8: ("2", "6"),
    9: ("2", "7"),
    10: ("2", "8"),
    11: ("2", "9"),
    12: ("2", "10"),
    13: ("3", "10"),
    14: ("4", "10"),
    15: ("5", "10"),
    16: ("6", "10"),
    17: ("7", "10"),
}
SOFT_HANDS = {total: ("A", str(total - 11)) for total in range(13, 21)}
PAIR_RANKS = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "A"]
DEALER_RANKS = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "A"]
ALLOWED_ACTIONS = {"stand", "hit", "double", "split"}


@dataclass(frozen=True)
class Scenario:
    key: str
    category: str
    label: str
    player_ranks: tuple[str, str]
    dealer_rank: str
    best_action: str
    valid_actions: tuple[str, ...]

    def to_public(self) -> dict:
        return {
            "key": self.key,
            "category": self.category,
            "label": self.label,
            "player_ranks": list(self.player_ranks),
            "dealer_rank": self.dealer_rank,
            "valid_actions": list(self.valid_actions),
            "player_score": format_hand_score(self.player_ranks),
            "dealer_score": format_hand_score((self.dealer_rank,)),
        }


class StartResponse(BaseModel):
    username: str
    total_scenarios: int
    remaining_errors: int
    scenario: Optional[dict]


class StartRequest(BaseModel):
    password: str = Field(min_length=6, max_length=6, pattern="^[0-9]{6}$")
    mode: str = Field(pattern="^(login|register)$", default="login")


class AnswerRequest(BaseModel):
    scenario_key: str = Field(min_length=1, max_length=32)
    action: str = Field(min_length=1, max_length=16)


class AnswerResponse(BaseModel):
    correct: bool
    correct_action: str
    total_attempts: int
    correct_attempts: int
    incorrect_attempts: int
    remaining_errors: int
    next_scenario: Optional[dict]


class ProgressResponse(BaseModel):
    username: str
    total_attempts: int
    correct_attempts: int
    incorrect_attempts: int
    remaining_errors: int
    error_keys: List[str]


def normalize_username(raw_username: str) -> str:
    normalized = "".join(
        ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in raw_username.strip()
    )
    if not normalized:
        raise HTTPException(status_code=400, detail="Invalid username")
    if len(normalized) > 32:
        raise HTTPException(status_code=400, detail="Username too long")
    return normalized


def format_hand_score(ranks: tuple[str, ...]) -> str:
    total_with_aces = sum(1 if rank == "A" else int(rank) for rank in ranks)
    aces = sum(1 for rank in ranks if rank == "A")
    totals = sorted(
        {total_with_aces + 10 * extra_aces for extra_aces in range(aces + 1)},
        reverse=True,
    )
    valid_totals = [value for value in totals if value <= 21]
    if valid_totals:
        return "/".join(str(value) for value in valid_totals)
    return str(min(totals))


def build_scenarios() -> Dict[str, Scenario]:
    scenarios: Dict[str, Scenario] = {}

    def create_scenario(
        category: str,
        label: str,
        player_ranks: tuple[str, str],
        dealer_rank: str,
        player_total: int,
        usable_ace: bool,
        can_split: bool,
        pair_rank: Optional[str],
    ) -> None:
        best_action, _, _ = evaluate_state(
            player_total=player_total,
            usable_ace=usable_ace,
            num_cards=2,
            dealer_upcard=dealer_rank,
            can_double=True,
            can_split=can_split,
            pair_rank=pair_rank,
        )
        valid_actions = ["stand", "hit", "double"]
        if can_split:
            valid_actions.append("split")
        key = f"{player_ranks[0]}-{player_ranks[1]}|{dealer_rank}"
        scenarios[key] = Scenario(
            key=key,
            category=category,
            label=label,
            player_ranks=player_ranks,
            dealer_rank=dealer_rank,
            best_action=best_action,
            valid_actions=tuple(valid_actions),
        )

    for total, player_ranks in HARD_HANDS.items():
        for dealer in DEALER_RANKS:
            create_scenario(
                category="Hard Totals",
                label=str(total),
                player_ranks=player_ranks,
                dealer_rank=dealer,
                player_total=total,
                usable_ace=False,
                can_split=False,
                pair_rank=None,
            )

    for total, player_ranks in SOFT_HANDS.items():
        for dealer in DEALER_RANKS:
            create_scenario(
                category="Soft Totals",
                label=str(total),
                player_ranks=player_ranks,
                dealer_rank=dealer,
                player_total=total,
                usable_ace=True,
                can_split=False,
                pair_rank=None,
            )

    for rank in PAIR_RANKS:
        pair_ranks = (rank, rank)
        pair_total, usable_ace = compute_hand_value(pair_ranks)
        for dealer in DEALER_RANKS:
            create_scenario(
                category="Pairs",
                label=rank,
                player_ranks=pair_ranks,
                dealer_rank=dealer,
                player_total=pair_total,
                usable_ace=usable_ace,
                can_split=True,
                pair_rank=rank,
            )

    return scenarios


SCENARIOS = build_scenarios()
SCENARIO_ORDER = sorted(SCENARIOS.keys())
RATE_LIMIT_BUCKETS: Dict[str, List[float]] = {}
RATE_LIMIT_LOCK = Lock()


def open_db() -> sqlite3.Connection:
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def init_db() -> None:
    with open_db() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS user_state (
                username TEXT PRIMARY KEY,
                current_index INTEGER NOT NULL DEFAULT 0,
                total_attempts INTEGER NOT NULL DEFAULT 0,
                correct_attempts INTEGER NOT NULL DEFAULT 0,
                incorrect_attempts INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        # Ensure password_hash and password_salt columns exist (for backward compatibility and smooth deployment on Render.com)
        cursor = connection.execute("PRAGMA table_info(user_state)")
        columns = [row["name"] for row in cursor.fetchall()]
        if "password_hash" not in columns:
            connection.execute("ALTER TABLE user_state ADD COLUMN password_hash TEXT")
        if "password_salt" not in columns:
            connection.execute("ALTER TABLE user_state ADD COLUMN password_salt TEXT")

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS user_errors (
                username TEXT NOT NULL,
                scenario_key TEXT NOT NULL,
                PRIMARY KEY (username, scenario_key)
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS user_attempts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                scenario_key TEXT NOT NULL,
                action TEXT NOT NULL,
                correct_action TEXT NOT NULL,
                is_correct INTEGER NOT NULL,
                created_at INTEGER NOT NULL
            )
            """
        )


def enforce_rate_limit(request: Request) -> None:
    forwarded = request.headers.get("cf-connecting-ip") or request.headers.get(
        "x-forwarded-for"
    )
    if forwarded:
        client_key = forwarded.split(",")[0].strip()
    elif request.client:
        client_key = request.client.host
    else:
        client_key = "unknown"

    now = time.time()
    window_start = now - 60.0
    with RATE_LIMIT_LOCK:
        bucket = RATE_LIMIT_BUCKETS.setdefault(client_key, [])
        while bucket and bucket[0] < window_start:
            bucket.pop(0)
        if len(bucket) >= REQUESTS_PER_MINUTE:
            raise HTTPException(status_code=429, detail="Too many requests")
        bucket.append(now)


def get_or_create_state(connection: sqlite3.Connection, username: str) -> sqlite3.Row:
    row = connection.execute(
        "SELECT username, current_index, total_attempts, correct_attempts, incorrect_attempts "
        "FROM user_state WHERE username = ?",
        (username,),
    ).fetchone()
    if row:
        return row
    connection.execute(
        "INSERT INTO user_state (username, current_index, total_attempts, correct_attempts, incorrect_attempts) "
        "VALUES (?, 0, 0, 0, 0)",
        (username,),
    )
    return connection.execute(
        "SELECT username, current_index, total_attempts, correct_attempts, incorrect_attempts "
        "FROM user_state WHERE username = ?",
        (username,),
    ).fetchone()


def get_current_scenario(current_index: int) -> Scenario:
    return SCENARIOS[SCENARIO_ORDER[current_index % len(SCENARIO_ORDER)]]


def count_errors(connection: sqlite3.Connection, username: str) -> int:
    row = connection.execute(
        "SELECT COUNT(*) AS total FROM user_errors WHERE username = ?", (username,)
    ).fetchone()
    return int(row["total"])


app = FastAPI(title="Blackjack Trainer API", version="1.0.0")
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


@app.on_event("startup")
def startup_event() -> None:
    init_db()


@app.middleware("http")
async def security_headers(request: Request, call_next):
    enforce_rate_limit(request)
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self'; "
        "style-src 'self'; "
        "img-src 'self' data:; "
        "connect-src 'self'; "
        "frame-ancestors 'none'; "
        "base-uri 'self'"
    )
    return response


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "scenarios": len(SCENARIOS), "db_path": str(DB_PATH.name)}


@app.get("/api/leaderboard")
def get_leaderboard() -> List[dict]:
    with open_db() as connection:
        cursor = connection.execute(
            """
            SELECT 
                username, 
                total_attempts, 
                correct_attempts, 
                incorrect_attempts,
                (correct_attempts * 100 / CASE WHEN total_attempts = 0 THEN 1 ELSE total_attempts END) AS accuracy,
                (SELECT COUNT(*) FROM user_errors WHERE user_errors.username = user_state.username) AS remaining_errors
            FROM user_state
            WHERE total_attempts > 0
            ORDER BY accuracy DESC, total_attempts DESC
            LIMIT 10
            """
        )
        return [dict(row) for row in cursor.fetchall()]


@app.get("/api/strategy-map")
def get_strategy_map() -> dict:
    return {key: scenario.best_action for key, scenario in SCENARIOS.items()}


class DeleteUserRequest(BaseModel):
    admin_key: str = Field(min_length=1, max_length=128)
    username: str = Field(min_length=1, max_length=32)


@app.post("/api/admin/delete-user")
def delete_user(payload: DeleteUserRequest) -> dict:
    expected_key = os.getenv("BLACKJACK_ADMIN_KEY", "admin123")
    if payload.admin_key != expected_key:
        raise HTTPException(status_code=403, detail="Clé secrète admin invalide.")
    
    target_username = normalize_username(payload.username)
    with open_db() as connection:
        # Check if user exists
        row = connection.execute(
            "SELECT username FROM user_state WHERE username = ?", (target_username,)
        ).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Utilisateur introuvable.")
        
        connection.execute("DELETE FROM user_state WHERE username = ?", (target_username,))
        connection.execute("DELETE FROM user_errors WHERE username = ?", (target_username,))
        connection.execute("DELETE FROM user_attempts WHERE username = ?", (target_username,))
        
    return {"ok": True, "detail": f"Utilisateur {target_username} supprimé avec succès."}


def hash_pin(pin: str, salt: Optional[str] = None) -> tuple[str, str]:
    if salt is None:
        salt = secrets.token_hex(16)
    # Standard SHA256 PBKDF2 with 100,000 iterations for secure PIN hashing
    pwd_hash = hashlib.pbkdf2_hmac(
        "sha256", 
        pin.encode("utf-8"), 
        salt.encode("utf-8"), 
        100000
    ).hex()
    return pwd_hash, salt


@app.post("/api/users/{username}/start", response_model=StartResponse)
def start_user(username: str, payload: StartRequest) -> StartResponse:
    normalized = normalize_username(username)
    pin = payload.password
    mode = payload.mode
    
    with open_db() as connection:
        # Verify or register password based on mode
        row = connection.execute(
            "SELECT password_hash, password_salt FROM user_state WHERE username = ?",
            (normalized,),
        ).fetchone()
        
        if mode == "login":
            if not row:
                raise HTTPException(status_code=404, detail="Utilisateur introuvable. Veuillez créer un compte.")
            
            db_hash = row["password_hash"]
            db_salt = row["password_salt"]
            
            if db_hash is not None and db_salt is not None:
                # User is password protected: authenticate
                pwd_hash, _ = hash_pin(pin, db_salt)
                if pwd_hash != db_hash:
                    raise HTTPException(status_code=401, detail="Code PIN incorrect pour cet utilisateur.")
            else:
                # Legacy user has no password yet: register this PIN as their login
                pwd_hash, pwd_salt = hash_pin(pin)
                connection.execute(
                    "UPDATE user_state SET password_hash = ?, password_salt = ? WHERE username = ?",
                    (pwd_hash, pwd_salt, normalized),
                )
        else:  # mode == "register"
            if row:
                raise HTTPException(status_code=400, detail="Ce pseudonyme est déjà pris. Veuillez en choisir un autre.")
            
            # New user: register with this PIN (state will be initialized by get_or_create_state below)
            pwd_hash, pwd_salt = hash_pin(pin)
            connection.execute(
                "INSERT INTO user_state (username, current_index, total_attempts, correct_attempts, incorrect_attempts, password_hash, password_salt) "
                "VALUES (?, 0, 0, 0, 0, ?, ?)",
                (normalized, pwd_hash, pwd_salt),
            )
            
        state = get_or_create_state(connection, normalized)
        scenario = get_current_scenario(int(state["current_index"]))
        return StartResponse(
            username=normalized,
            total_scenarios=len(SCENARIOS),
            remaining_errors=count_errors(connection, normalized),
            scenario=scenario.to_public(),
        )


@app.get("/api/users/{username}/progress", response_model=ProgressResponse)
def get_progress(username: str) -> ProgressResponse:
    normalized = normalize_username(username)
    with open_db() as connection:
        state = get_or_create_state(connection, normalized)
        # Fetch active error keys
        cursor = connection.execute(
            "SELECT scenario_key FROM user_errors WHERE username = ?", (normalized,)
        )
        error_keys = [row["scenario_key"] for row in cursor.fetchall()]
        return ProgressResponse(
            username=normalized,
            total_attempts=int(state["total_attempts"]),
            correct_attempts=int(state["correct_attempts"]),
            incorrect_attempts=int(state["incorrect_attempts"]),
            remaining_errors=count_errors(connection, normalized),
            error_keys=error_keys,
        )


@app.post("/api/users/{username}/answer", response_model=AnswerResponse)
def submit_answer(username: str, payload: AnswerRequest) -> AnswerResponse:
    normalized = normalize_username(username)
    action = payload.action.lower().strip()
    if action not in ALLOWED_ACTIONS:
        raise HTTPException(status_code=400, detail="Invalid action")

    scenario = SCENARIOS.get(payload.scenario_key)
    if scenario is None:
        raise HTTPException(status_code=400, detail="Invalid scenario key")
    if action not in scenario.valid_actions:
        raise HTTPException(status_code=400, detail="Action not valid for this scenario")

    is_correct = action == scenario.best_action
    with open_db() as connection:
        state = get_or_create_state(connection, normalized)
        total_attempts = int(state["total_attempts"]) + 1
        correct_attempts = int(state["correct_attempts"]) + (1 if is_correct else 0)
        incorrect_attempts = int(state["incorrect_attempts"]) + (0 if is_correct else 1)
        next_index = (int(state["current_index"]) + 1) % len(SCENARIO_ORDER)

        connection.execute(
            "UPDATE user_state SET current_index = ?, total_attempts = ?, correct_attempts = ?, incorrect_attempts = ? "
            "WHERE username = ?",
            (next_index, total_attempts, correct_attempts, incorrect_attempts, normalized),
        )

        if is_correct:
            connection.execute(
                "DELETE FROM user_errors WHERE username = ? AND scenario_key = ?",
                (normalized, scenario.key),
            )
        else:
            connection.execute(
                "INSERT OR IGNORE INTO user_errors (username, scenario_key) VALUES (?, ?)",
                (normalized, scenario.key),
            )

        connection.execute(
            "INSERT INTO user_attempts (username, scenario_key, action, correct_action, is_correct, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                normalized,
                scenario.key,
                action,
                scenario.best_action,
                1 if is_correct else 0,
                int(time.time()),
            ),
        )

        next_scenario = get_current_scenario(next_index)
        remaining_errors = count_errors(connection, normalized)

    return AnswerResponse(
        correct=is_correct,
        correct_action=scenario.best_action,
        total_attempts=total_attempts,
        correct_attempts=correct_attempts,
        incorrect_attempts=incorrect_attempts,
        remaining_errors=remaining_errors,
        next_scenario=next_scenario.to_public(),
    )


def main() -> None:
    import uvicorn

    host = os.getenv("BLACKJACK_WEB_HOST", "127.0.0.1")
    port = int(os.getenv("BLACKJACK_WEB_PORT", "8000"))
    uvicorn.run("game.web_api:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
