import asyncio
import json
import math
import random
import re
import time
import uuid

from collections import deque
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

WIDTH = 480
HEIGHT = 720

RIVER_TOP = 340
RIVER_BOTTOM = 380
BRIDGES = (130, 350)

MATCH_SECONDS = 180
SUDDEN_DEATH_SECONDS = 60
TICK_SECONDS = 1 / 20

DECK_SIZE = 8
MAX_ELIXIR = 10.0
STARTING_ELIXIR = 5.0
ELIXIR_PER_SECOND = 1.0
SUDDEN_DEATH_ELIXIR_PER_SECOND = 2.0

MAX_UNITS_PER_PLAYER = 40
MAX_MESSAGE_SIZE = 4096
MAX_MESSAGES_PER_SECOND = 30


# ============================================================
# CARD DATABASE — 14 AVAILABLE CARDS
# ============================================================

CARDS = {
    "knight": {
        "cost": 3,
        "hp": 360,
        "damage": 48,
        "speed": 58,
        "range": 27,
        "cooldown": 0.8,
    },
    "archer": {
        "cost": 3,
        "hp": 155,
        "damage": 32,
        "speed": 52,
        "range": 125,
        "cooldown": 0.85,
    },
    "giant": {
        "cost": 5,
        "hp": 1000,
        "damage": 78,
        "speed": 33,
        "range": 32,
        "cooldown": 1.2,
        "buildings_only": True,
    },
    "goblin": {
        "cost": 2,
        "hp": 130,
        "damage": 27,
        "speed": 88,
        "range": 23,
        "cooldown": 0.5,
    },
    "brute": {
        "cost": 4,
        "hp": 530,
        "damage": 100,
        "speed": 42,
        "range": 30,
        "cooldown": 1.15,
    },
    "mage": {
        "cost": 4,
        "hp": 215,
        "damage": 68,
        "speed": 47,
        "range": 115,
        "cooldown": 1.05,
    },
    "goku": {
        "cost": 5,
        "hp": 620,
        "damage": 95,
        "speed": 68,
        "range": 95,
        "cooldown": 0.85,
    },
    "vegeta": {
        "cost": 5,
        "hp": 560,
        "damage": 115,
        "speed": 65,
        "range": 105,
        "cooldown": 1.0,
    },
    "piccolo": {
        "cost": 4,
        "hp": 540,
        "damage": 70,
        "speed": 49,
        "range": 110,
        "cooldown": 0.95,
    },
    "frieza": {
        "cost": 6,
        "hp": 650,
        "damage": 125,
        "speed": 61,
        "range": 130,
        "cooldown": 1.1,
    },
    "trunks": {
        "cost": 4,
        "hp": 440,
        "damage": 85,
        "speed": 73,
        "range": 30,
        "cooldown": 0.75,
    },
    "gohan": {
        "cost": 4,
        "hp": 480,
        "damage": 76,
        "speed": 66,
        "range": 85,
        "cooldown": 0.8,
    },
    "android18": {
        "cost": 3,
        "hp": 330,
        "damage": 48,
        "speed": 66,
        "range": 110,
        "cooldown": 0.75,
    },
    "cell": {
        "cost": 6,
        "hp": 900,
        "damage": 105,
        "speed": 47,
        "range": 35,
        "cooldown": 0.95,
    },
}

DEFAULT_DECK = (
    "knight",
    "archer",
    "giant",
    "goblin",
    "goku",
    "vegeta",
    "piccolo",
    "trunks",
)


# ============================================================
# DATA MODELS
# ============================================================

@dataclass
class Player:
    ws: WebSocket | None
    username: str
    deck: tuple[str, ...]

    send_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    room_id: str | None = None
    side: int = 0
    last_room_id: str | None = None


@dataclass
class Room:
    id: str
    players: tuple[Player, Player]

    towers: list[dict] = field(default_factory=list)
    units: dict[str, dict] = field(default_factory=dict)

    elixir: list[float] = field(
        default_factory=lambda: [STARTING_ELIXIR, STARTING_ELIXIR]
    )

    started_at: float = field(default_factory=time.monotonic)
    ended: bool = False

    is_sudden_death: bool = False
    sudden_death_started_at: float | None = None
    towers_at_sd_start: set[str] = field(default_factory=set)
    sudden_death_announced: bool = False
    mode: str = "standard"


# ============================================================
# GLOBAL STATE
#
# Run ONE Uvicorn worker. These values are process-local.
# ============================================================

active_connections: dict[WebSocket, Player] = {}
online_players: list[str] = []
leaderboard: dict[str, int] = {}

waiting_players: dict[str, Player | None] = {
    "standard": None,
    "sudden_death": None,
}

rooms: dict[str, Room] = {}
room_tasks: set[asyncio.Task] = set()


# ============================================================
# APPLICATION LIFECYCLE
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield

    tasks = list(room_tasks)

    for task in tasks:
        task.cancel()

    await asyncio.gather(*tasks, return_exceptions=True)


app = FastAPI(
    title="Bridgebound Legends Arena",
    lifespan=lifespan,
)


# ============================================================
# HTTP ROUTES
# ============================================================

@app.get("/")
async def index():
    return FileResponse(BASE_DIR / "index.html")


@app.get("/health")
async def health():
    return {"status": "ok"}


def stats_payload():
    ranked_players = sorted(
        leaderboard.items(),
        key=lambda item: (-item[1], item[0].lower()),
    )

    return {
        "type": "server_stats",
        "online_count": len(active_connections),
        "online_players": list(online_players),
        "leaderboard": [
            {
                "username": username,
                "wins": wins,
            }
            for username, wins in ranked_players[:10]
        ],
    }


@app.get("/stats")
async def stats():
    # The lobby can fetch statistics before entering matchmaking.
    return stats_payload()


# ============================================================
# NETWORK HELPERS
# ============================================================

async def send(player: Player, payload: dict):
    if player.ws is None:
        return

    try:
        async with player.send_lock:
            await asyncio.wait_for(
                player.ws.send_json(payload),
                timeout=2,
            )
    except Exception:
        # Closing a stalled socket triggers receive-loop cleanup.
        with suppress(Exception):
            await asyncio.wait_for(
                player.ws.close(),
                timeout=1,
            )


async def broadcast_stats():
    payload = stats_payload()

    await asyncio.gather(
        *(
            send(player, payload)
            for player in list(active_connections.values())
        )
    )


def unique_username(raw: str):
    # Display names only: this is not an authentication system.
    base = re.sub(r"[^\w -]", "", raw).strip()[:20] or "Guest"

    candidate = base
    suffix = 2

    while candidate in online_players:
        suffix_text = f"-{suffix}"
        candidate = f"{base[:20 - len(suffix_text)]}{suffix_text}"
        suffix += 1

    return candidate


# ============================================================
# ROOM CREATION
# ============================================================

def create_room(first: Player, second: Player):
    room = Room(
        id=uuid.uuid4().hex,
        players=(first, second),
    )

    # Canonical server coordinates:
    # side 0 = bottom
    # side 1 = top
    #
    # Each frontend rotates/mirrors the board so its own side
    # always appears at the bottom.

    tower_layout = (
        ("king", 240, 654, 1600),
        ("princess", 100, 560, 1000),
        ("princess", 380, 560, 1000),
    )

    for side in (0, 1):
        for index, (kind, x, y, hp) in enumerate(tower_layout):
            if side == 1:
                x = WIDTH - x
                y = HEIGHT - y

            room.towers.append({
                "id": f"{side}-{kind}-{index}",
                "owner": side,
                "kind": kind,
                "x": x,
                "y": y,
                "hp": hp,
                "max_hp": hp,
                "cooldown": 0.0,
            })

    for side, player in enumerate(room.players):
        player.side = side
        player.room_id = room.id
        player.last_room_id = None

    rooms[room.id] = room

    return room


# ============================================================
# MOVEMENT AND TARGETING
# ============================================================

def distance(a: dict, b: dict):
    return math.hypot(
        a["x"] - b["x"],
        a["y"] - b["y"],
    )


def attack_distance(attacker: dict, target: dict):
    target_radius = 23 if "kind" in target else 9

    return max(
        0,
        distance(attacker, target) - target_radius,
    )


def move_toward(unit: dict, target: dict, speed: float, dt: float):
    tx = target["x"]
    ty = target["y"]

    x = unit["x"]
    y = unit["y"]

    bridge = min(
        BRIDGES,
        key=lambda bridge_x: abs(bridge_x - x),
    )

    # Ground units cross the river only at a bridge.
    if y > RIVER_BOTTOM and ty < RIVER_BOTTOM:
        if abs(x - bridge) > 6:
            tx = bridge
            ty = max(y, RIVER_BOTTOM + 16)
        else:
            tx = bridge
            ty = RIVER_TOP - 16

    elif y < RIVER_TOP and ty > RIVER_TOP:
        if abs(x - bridge) > 6:
            tx = bridge
            ty = min(y, RIVER_TOP - 16)
        else:
            tx = bridge
            ty = RIVER_BOTTOM + 16

    elif RIVER_TOP <= y <= RIVER_BOTTOM:
        tx = bridge
        ty = (
            RIVER_TOP - 16
            if ty < y
            else RIVER_BOTTOM + 16
        )

    dx = tx - x
    dy = ty - y
    length = math.hypot(dx, dy)

    if length <= 0:
        return

    step = min(length, speed * dt)

    unit["x"] = max(
        12,
        min(WIDTH - 12, x + dx / length * step),
    )

    unit["y"] = max(
        12,
        min(HEIGHT - 12, y + dy / length * step),
    )


# ============================================================
# AUTHORITATIVE COMBAT SIMULATION
# ============================================================

def simulate(room: Room, dt: float):
    elixir_rate = (
        SUDDEN_DEATH_ELIXIR_PER_SECOND
        if room.is_sudden_death
        else ELIXIR_PER_SECOND
    )
    for side in (0, 1):
        room.elixir[side] = min(
            MAX_ELIXIR,
            room.elixir[side] + elixir_rate * dt,
        )

    units = list(room.units.values())
    damage_events = []

    # Unit movement and attacks.
    for unit in units:
        if unit["hp"] <= 0:
            continue

        card = CARDS[unit["card"]]

        unit["cooldown"] = max(
            0,
            unit["cooldown"] - dt,
        )

        candidates = []

        if not card.get("buildings_only", False):
            candidates = [
                enemy
                for enemy in units
                if enemy["owner"] != unit["owner"]
                and enemy["hp"] > 0
                and distance(unit, enemy) <= 145
            ]

        if not candidates:
            candidates = [
                tower
                for tower in room.towers
                if tower["owner"] != unit["owner"]
                and tower["hp"] > 0
            ]

        if not candidates:
            continue

        target = min(
            candidates,
            key=lambda enemy: distance(unit, enemy),
        )

        if attack_distance(unit, target) <= card["range"]:
            if unit["cooldown"] <= 0:
                damage_events.append(
                    (target, card["damage"])
                )

                unit["cooldown"] = card["cooldown"]
        else:
            move_toward(
                unit,
                target,
                card["speed"],
                dt,
            )

    # Tower attacks.
    for tower in room.towers:
        if tower["hp"] <= 0:
            continue

        tower["cooldown"] = max(
            0,
            tower["cooldown"] - dt,
        )

        tower_range = (
            150
            if tower["kind"] == "king"
            else 165
        )

        enemies = [
            unit
            for unit in units
            if unit["owner"] != tower["owner"]
            and unit["hp"] > 0
            and distance(tower, unit) <= tower_range
        ]

        if enemies and tower["cooldown"] <= 0:
            target = min(
                enemies,
                key=lambda unit: distance(tower, unit),
            )

            damage_events.append((target, 44))
            tower["cooldown"] = 0.9

    # Apply damage together so simultaneous King destruction
    # can correctly result in a draw.
    for target, damage in damage_events:
        target["hp"] = max(
            0,
            target["hp"] - damage,
        )

    room.units = {
        unit_id: unit
        for unit_id, unit in room.units.items()
        if unit["hp"] > 0
    }


# ============================================================
# MATCH RESULTS
# ============================================================

def outcome(room: Room):
    """
    Return:
        None -> match still running
        -1   -> draw
         0   -> side 0 wins
         1   -> side 1 wins
    """

    kings = {
        tower["owner"]: tower
        for tower in room.towers
        if tower["kind"] == "king"
    }

    side_0_dead = kings[0]["hp"] <= 0
    side_1_dead = kings[1]["hp"] <= 0

    if side_0_dead and side_1_dead:
        return -1

    if side_0_dead:
        return 1

    if side_1_dead:
        return 0

    now = time.monotonic()

    # Sudden death active: First tower down wins immediately!
    if room.is_sudden_death:
        dead_since_sd = [
            t for t in room.towers
            if t["hp"] <= 0 and t["id"] in room.towers_at_sd_start
        ]
        if dead_since_sd:
            dead_owners = {t["owner"] for t in dead_since_sd}
            if 0 in dead_owners and 1 in dead_owners:
                return -1
            if 0 in dead_owners:
                return 1
            if 1 in dead_owners:
                return 0

        # Sudden death timer expired: Tiebreaker
        sd_elapsed = now - (room.sudden_death_started_at or now)
        if sd_elapsed >= SUDDEN_DEATH_SECONDS:
            # Lowest tower health tiebreaker
            min_hp_0 = min((t["hp"] for t in room.towers if t["owner"] == 0), default=0)
            min_hp_1 = min((t["hp"] for t in room.towers if t["owner"] == 1), default=0)
            if min_hp_0 != min_hp_1:
                return 0 if min_hp_0 > min_hp_1 else 1

            total_health = [
                sum(
                    tower["hp"]
                    for tower in room.towers
                    if tower["owner"] == side
                )
                for side in (0, 1)
            ]

            if total_health[0] == total_health[1]:
                return -1

            return (
                0
                if total_health[0] > total_health[1]
                else 1
            )

        return None

    # Normal regulation time check
    elapsed = now - room.started_at

    if elapsed >= MATCH_SECONDS:
        side_0_lost = sum(
            1 for t in room.towers
            if t["owner"] == 0 and t["hp"] <= 0
        )
        side_1_lost = sum(
            1 for t in room.towers
            if t["owner"] == 1 and t["hp"] <= 0
        )

        if side_0_lost < side_1_lost:
            return 0
        elif side_1_lost < side_0_lost:
            return 1
        else:
            # Tower destructions are tied -> Enter SUDDEN DEATH!
            room.is_sudden_death = True
            room.sudden_death_started_at = now
            room.towers_at_sd_start = {
                t["id"] for t in room.towers if t["hp"] > 0
            }
            room.sudden_death_announced = False
            return None

    return None


async def send_snapshot(room: Room):
    unit_fields = (
        "id",
        "owner",
        "card",
        "x",
        "y",
        "hp",
        "max_hp",
    )

    units = [
        {
            key: unit[key]
            for key in unit_fields
        }
        for unit in room.units.values()
    ]

    towers = [
        {
            key: value
            for key, value in tower.items()
            if key != "cooldown"
        }
        for tower in room.towers
    ]

    if room.is_sudden_death:
        sd_elapsed = time.monotonic() - (
            room.sudden_death_started_at or room.started_at
        )
        remaining = max(0, SUDDEN_DEATH_SECONDS - sd_elapsed)
    else:
        remaining = max(
            0,
            MATCH_SECONDS - (time.monotonic() - room.started_at),
        )

    common = {
        "type": "state",
        "room_id": room.id,
        "units": units,
        "towers": towers,
        "remaining": remaining,
        "sudden_death": room.is_sudden_death,
        "elixir_rate": (
            SUDDEN_DEATH_ELIXIR_PER_SECOND
            if room.is_sudden_death
            else ELIXIR_PER_SECOND
        ),
    }

    await asyncio.gather(
        *(
            send(
                player,
                {
                    **common,
                    "elixir": round(room.elixir[side], 2),
                },
            )
            for side, player in enumerate(room.players)
        )
    )


async def finish_room(
    room: Room,
    winner: int,
    reason: str,
):
    if room.ended:
        return

    # Finalize before any await: a result can only be awarded once.
    room.ended = True
    rooms.pop(room.id, None)

    winner_name = None

    if winner in (0, 1):
        winner_name = room.players[winner].username

        leaderboard[winner_name] = (
            leaderboard.get(winner_name, 0) + 1
        )

    for player in room.players:
        player.room_id = None
        player.last_room_id = room.id

    if reason == "disconnect" and winner in (0, 1):
        await send(
            room.players[winner],
            {
                "type": "opponent_disconnected",
                "room_id": room.id,
            },
        )

    await send_snapshot(room)

    await asyncio.gather(
        *(
            send(
                player,
                {
                    "type": "game_over",
                    "room_id": room.id,
                    "winner": winner_name,
                    "result": (
                        "draw"
                        if winner == -1
                        else "victory"
                        if side == winner
                        else "defeat"
                    ),
                    "reason": reason,
                },
            )
            for side, player in enumerate(room.players)
        )
    )

    await broadcast_stats()


# ============================================================
# ROOM GAME LOOP
# ============================================================

async def room_loop(room: Room):
    await asyncio.gather(
        *(
            send(
                player,
                {
                    "type": "init",
                    "room_id": room.id,
                    "player_index": side,
                    "username": player.username,
                    "opponent": room.players[1 - side].username,
                    "deck": list(player.deck),
                    "width": WIDTH,
                    "height": HEIGHT,
                    "sudden_death": room.is_sudden_death,
                },
            )
            for side, player in enumerate(room.players)
        )
    )

    if room.ended:
        return

    room.started_at = time.monotonic()
    if room.is_sudden_death:
        room.sudden_death_started_at = room.started_at
        room.towers_at_sd_start = {t["id"] for t in room.towers if t["hp"] > 0}
        room.sudden_death_announced = True

    previous = room.started_at
    tick = 0
    bot_timer = 0.0

    await send_snapshot(room)

    while not room.ended:
        now = time.monotonic()
        dt = min(now - previous, 0.1)
        previous = now

        # Automated Trainer Bot logic for solo practice
        for bot_player in room.players:
            if bot_player.ws is None:
                bot_timer += dt
                if bot_timer >= 2.6:
                    bot_timer = 0.0
                    bot_side = bot_player.side
                    current_elixir = room.elixir[bot_side]
                    affordable = [
                        c for c in bot_player.deck
                        if CARDS[c]["cost"] <= current_elixir
                    ]
                    if affordable:
                        chosen_card = random.choice(affordable)
                        deploy_x = random.uniform(80, WIDTH - 80)
                        deploy_y = random.uniform(430, HEIGHT - 60)
                        await deploy(bot_player, {
                            "card": chosen_card,
                            "x": deploy_x,
                            "y": deploy_y,
                        })

        simulate(room, dt)

        # Notify clients if Sudden Death was triggered mid-match
        if room.is_sudden_death and not room.sudden_death_announced:
            room.sudden_death_announced = True
            await asyncio.gather(
                *(
                    send(
                        player,
                        {
                            "type": "sudden_death",
                            "room_id": room.id,
                            "duration": SUDDEN_DEATH_SECONDS,
                            "elixir_rate": SUDDEN_DEATH_ELIXIR_PER_SECOND,
                        },
                    )
                    for player in room.players
                )
            )

        result = outcome(room)

        if result is not None:
            kings_destroyed = any(
                tower["kind"] == "king"
                and tower["hp"] <= 0
                for tower in room.towers
            )

            if kings_destroyed:
                reason = "king_destroyed"
            elif room.is_sudden_death:
                sd_elapsed = time.monotonic() - (
                    room.sudden_death_started_at or time.monotonic()
                )
                if sd_elapsed >= SUDDEN_DEATH_SECONDS:
                    reason = "tiebreaker"
                else:
                    reason = "sudden_death_tower"
            else:
                reason = "time"

            await finish_room(room, result, reason)
            return

        tick += 1

        # Combat runs at approximately 20 Hz.
        # Clients receive snapshots at approximately 10 Hz.
        if tick % 2 == 0:
            await send_snapshot(room)

        elapsed = time.monotonic() - now

        await asyncio.sleep(
            max(0, TICK_SECONDS - elapsed)
        )


# ============================================================
# DEPLOYMENT VALIDATION
# ============================================================

async def deploy(player: Player, data: dict):
    room = rooms.get(player.room_id)

    if not room or room.ended:
        return

    card_id = data.get("card")

    if (
        not isinstance(card_id, str)
        or card_id not in player.deck
    ):
        await send(
            player,
            {
                "type": "error",
                "message": "Card not in your deck.",
            },
        )
        return

    try:
        # Avoid treating JSON booleans as numeric coordinates.
        if isinstance(data["x"], bool) or isinstance(data["y"], bool):
            return

        x = float(data["x"])
        y = float(data["y"])

    except (KeyError, TypeError, ValueError, OverflowError):
        return

    # Input coordinates are always from the sender's local view.
    if not (
        math.isfinite(x)
        and math.isfinite(y)
        and 24 <= x <= WIDTH - 24
        and 400 <= y <= HEIGHT - 24
    ):
        await send(
            player,
            {
                "type": "error",
                "message": "Deploy on your side, below the river.",
            },
        )
        return

    card = CARDS[card_id]
    side = player.side

    if room.elixir[side] < card["cost"]:
        await send(
            player,
            {
                "type": "error",
                "message": "Not enough elixir.",
            },
        )
        return

    unit_count = sum(
        unit["owner"] == side
        for unit in room.units.values()
    )

    if unit_count >= MAX_UNITS_PER_PLAYER:
        await send(
            player,
            {
                "type": "error",
                "message": "Unit limit reached.",
            },
        )
        return

    room.elixir[side] -= card["cost"]

    if side == 0:
        canonical_x = x
        canonical_y = y
    else:
        canonical_x = WIDTH - x
        canonical_y = HEIGHT - y

    unit_id = uuid.uuid4().hex

    room.units[unit_id] = {
        "id": unit_id,
        "owner": side,
        "card": card_id,
        "x": canonical_x,
        "y": canonical_y,
        "hp": card["hp"],
        "max_hp": card["hp"],
        "cooldown": 0.0,
    }

    event = {
        "room_id": room.id,
        "id": unit_id,
        "card": card_id,

        # Sender-local coordinates.
        # The opponent frontend mirrors x and y.
        "x": x,
        "y": y,

        "hp": card["hp"],
        "max_hp": card["hp"],
    }

    await asyncio.gather(
        send(
            player,
            {
                **event,
                "type": "deploy_ack",
                "elixir": room.elixir[side],
            },
        ),
        send(
            room.players[1 - side],
            {
                **event,
                "type": "opponent_deploy",
            },
        ),
    )


# ============================================================
# DISCONNECTION CLEANUP
# ============================================================

async def disconnect(player: Player):
    for mode_key in waiting_players:
        if waiting_players[mode_key] is player:
            waiting_players[mode_key] = None

    if active_connections.pop(player.ws, None) is None:
        return

    room = rooms.get(player.room_id)

    if room and not room.ended:
        await finish_room(
            room,
            1 - player.side,
            "disconnect",
        )
    else:
        await broadcast_stats()


# ============================================================
# WEBSOCKET ENDPOINT
# ============================================================

@app.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    username: str = "Guest",
    deck: str = ",".join(DEFAULT_DECK),
    mode: str = "standard",
    vs_bot: bool = False,
):
    await websocket.accept()

    selected_deck = tuple(deck.split(","))

    if (
        len(selected_deck) != DECK_SIZE
        or len(set(selected_deck)) != DECK_SIZE
        or any(card not in CARDS for card in selected_deck)
    ):
        await websocket.close(
            code=1008,
            reason="Choose eight different cards.",
        )
        return

    player = Player(
        ws=websocket,
        username=unique_username(username),
        deck=selected_deck,
    )

    active_connections[websocket] = player
    online_players.append(player.username)

    if vs_bot:
        bot_deck = tuple(random.sample(list(CARDS.keys()), DECK_SIZE))
        bot = Player(
            ws=None,
            username="Trainer Bot",
            deck=bot_deck,
        )
        new_room = create_room(player, bot)
        if mode == "sudden_death":
            new_room.is_sudden_death = True
            new_room.mode = "sudden_death"

        task = asyncio.create_task(room_loop(new_room))
        room_tasks.add(task)
        task.add_done_callback(room_tasks.discard)

    else:
        norm_mode = "sudden_death" if mode == "sudden_death" else "standard"
        waiting = waiting_players.get(norm_mode)

        if waiting is None:
            waiting_players[norm_mode] = player
            new_room = None
        else:
            opponent = waiting
            waiting_players[norm_mode] = None

            new_room = create_room(opponent, player)
            if norm_mode == "sudden_death":
                new_room.is_sudden_death = True
                new_room.mode = "sudden_death"

            task = asyncio.create_task(room_loop(new_room))
            room_tasks.add(task)
            task.add_done_callback(room_tasks.discard)

    recent_messages = deque()

    try:
        if new_room is None:
            await send(
                player,
                {
                    "type": "waiting",
                    "username": player.username,
                },
            )

        await broadcast_stats()

        while True:
            message = await websocket.receive()

            if message["type"] == "websocket.disconnect":
                break

            raw = message.get("text")

            # This protocol only accepts text JSON messages.
            if raw is None:
                await websocket.close(
                    code=1003,
                    reason="Text JSON messages required.",
                )
                break

            now = time.monotonic()

            while (
                recent_messages
                and recent_messages[0] < now - 1
            ):
                recent_messages.popleft()

            recent_messages.append(now)

            if (
                len(raw.encode("utf-8")) > MAX_MESSAGE_SIZE
                or len(recent_messages) > MAX_MESSAGES_PER_SECOND
            ):
                await websocket.close(
                    code=1008,
                    reason="Message limit exceeded.",
                )
                break

            try:
                data = json.loads(raw)
            except (json.JSONDecodeError, RecursionError):
                continue

            if not isinstance(data, dict):
                continue

            event_type = data.get("type")

            if event_type == "deploy":
                if data.get("room_id") == player.room_id:
                    await deploy(player, data)

            elif event_type == "game_over":
                # Client reports cannot manufacture wins.
                # Only a server-verified result updates the leaderboard.
                room = rooms.get(player.room_id)

                if (
                    room
                    and data.get("room_id") == room.id
                ):
                    result = outcome(room)

                    if result is not None:
                        await finish_room(
                            room,
                            result,
                            "verified_result",
                        )

            elif event_type == "ping":
                await send(
                    player,
                    {"type": "pong"},
                )

    except (WebSocketDisconnect, RuntimeError, OSError):
        pass

    finally:
        await disconnect(player)


# ============================================================
# OPTIONAL: RUN WITH `python server.py`
# ============================================================

if __name__ == "__main__":
    import os
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8000")),
        ws_max_size=MAX_MESSAGE_SIZE,
    )
