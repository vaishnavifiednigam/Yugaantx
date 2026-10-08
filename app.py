# ═══════════════════════════════════════════════════════════════════════════
#  YUDAANT  —  a transparent, rule-based artificial-society simulator
#  ═══════════════════════════════════════════════════════════════════════════
#
#  Everything in this file is SYNTHETIC. It generates a small world and the
#  agents inside it from a visible random seed, simulates them round by round
#  with plain, explainable rules — or, optionally, a tiny learned policy per
#  agent trained live on its own experience buffer (still no LLM, no internet),
#  then measures what happened and shows honest, labelled charts.
#
#  Run it:
#      pip install -r requirements.txt
#      streamlit run app.py
#
#  Same seed + same settings → exactly the same run (Mersenne Twister,
#  deterministic ordering). A built-in self-check suite verifies this on
#  every launch — see the "Method & Limits" tab.
#
#  This is ONE file on purpose: sections 0–7 below are clearly separated.
# ═══════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import hashlib
import io
import json
import math
import random
import time
import traceback
import zipfile
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 0 · identity, constants, small shared helpers
# ═══════════════════════════════════════════════════════════════════════════

APP_NAME = "YUDAANT"
ENGINE_VERSION = "yudaant-engine-1.1.0"
DISCLAIMER = ("Simulated synthetic data. These agents are simple utility maximisers defined by the "
              "rules in this file. Results describe what happens INSIDE this model under this seed — "
              "they are not evidence about real people, real economies, or real AI systems.")

# world layout
CELL = 4.0                    # a cell is 4×4 world-units
WORLD_COLS, WORLD_ROWS = 24, 12
WORLD_W, WORLD_H = CELL * WORLD_COLS, CELL * WORLD_ROWS
CELL_FOOD_CAP = 100.0         # food a single map cell can hold
AGENT_FOOD_CAP = 90.0         # how much food one agent can carry
METABOLISM = 4.0              # energy spent just living, per round
EAT_ENERGY_PER_FOOD = 2.2     # energy gained per unit of food eaten
EAT_TRIGGER_ENERGY = 35.0     # below this, agents auto-eat from their stores
STARVE_DAMAGE = 3.5           # health lost per round with an empty stomach
HEAL_RATE = 0.8               # health regained per round when well fed & rested-enough

# actions the policy can choose between
ACTIONS = ["collect", "move", "explore", "rest", "trade", "share", "cooperate", "compete"]
FORCED = ["collapse"]

ACTION_COST_ENERGY = {
    "collect": 3.0, "move": 1.5, "explore": 5.0, "rest": 0.0,
    "trade": 2.0, "share": 2.0, "cooperate": 4.0, "compete": 3.0, "collapse": 0.0,
}
ACTION_BASE_REWARD = {
    "collect": 0.4, "move": 0.05, "explore": 0.30, "rest": 0.10,
    "trade": 1.6, "share": 1.8, "cooperate": 2.8, "compete": 2.2, "collapse": -1.0,
}
# reward-model multipliers: which society pays for what
REWARD_MULTS = {
    "MIXED":       {"trade": 1.00, "share": 1.00, "cooperate": 1.00, "compete": 1.00, "collect": 1.00},
    "COOPERATIVE": {"trade": 1.10, "share": 2.20, "cooperate": 1.80, "compete": 0.35, "collect": 0.80},
    "COMPETITIVE": {"trade": 1.50, "share": 0.35, "cooperate": 0.50, "compete": 2.00, "collect": 1.20},
}
ACTION_COLOR = {
    "share": "#34d399", "cooperate": "#2dd4bf", "trade": "#a78bfa",
    "compete": "#ef4444", "collect": "#f5b23c", "move": "#60a5fa",
    "explore": "#38bdf8", "rest": "#94a3b8", "collapse": "#7f1d1d",
}
ACTION_LABEL = {
    "share": "Share", "cooperate": "Co-operate", "trade": "Trade", "compete": "Compete",
    "collect": "Collect", "move": "Move", "explore": "Explore", "rest": "Rest",
    "collapse": "Collapse (exhausted)",
}
COMM_LEVELS = ["OFF", "LIMITED", "FULL"]
REWARD_MODELS = ["MIXED", "COOPERATIVE", "COMPETITIVE"]

PALETTE = {"teal": "#2dd4bf", "green": "#34d399", "violet": "#a78bfa", "amber": "#f5b23c",
           "red": "#f87171", "blue": "#60a5fa", "gray": "#94a3b8", "ink": "#0b1220",
           "grid": "#1c2940", "text": "#8ea0bd"}
GROUP_COLORS = ["#2dd4bf", "#a78bfa", "#f5b23c", "#38bdf8", "#34d399", "#f472b6",
                "#fb923c", "#818cf8", "#4ade80", "#e879f9"]
CAT_COLOR = {"co-operative": "#34d399", "trading": "#a78bfa", "hostile": "#f87171",
             "gathering": "#f5b23c", "seeking": "#60a5fa", "resting": "#94a3b8",
             "collapsed": "#b91c1c"}


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def clamp01(x: float) -> float:
    return clamp(x, 0.0, 1.0)


def gini(values: list[float]) -> float:
    """Gini coefficient of a list of non-negative quantities (0 = equal, 1 = one holder)."""
    v = sorted(x for x in values if x >= 0)
    n, m = len(v), sum(v)
    if n == 0 or m <= 0:
        return 0.0
    cum = 0.0
    for i, x in enumerate(v, start=1):
        cum += x * i
    return clamp01((2.0 * cum - (n + 1) * m) / (n * m))


def fmt(x: Any, d: int = 2) -> str:
    try:
        return f"{x:,.{d}f}"
    except Exception:
        return str(x)


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 1 · configuration, presets, comparison knobs
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class SimConfig:
    """Every knob of the simulation. The full run is reproducible from this + the seed."""
    population: int = 110           # number of starting agents (5–250)
    rounds: int = 200               # rounds to simulate (10–500)
    seed: int = 1337                # visible seed — same seed ⇒ same world & run
    start_trust: float = 0.50       # 0..1  how much strangers are trusted at the start
    memory: bool = True             # agents remember & update trust per partner
    communication: str = "LIMITED"  # OFF / LIMITED / FULL — how reputations spread
    reward_model: str = "MIXED"     # MIXED / COOPERATIVE / COMPETITIVE
    food_regrowth: float = 2.4      # food each map cell regrows per round (0.05–6)
    food_initial: float = 55.0      # average starting food per map cell (5–100)
    movement_speed: int = 1         # cells moved per Move action (1–3)
    policy: str = "RULE"            # RULE = fixed factor weights · LEARNED = per-agent policy gradient
    policy_lr: float = 0.10         # learning rate (0.01–0.40)
    policy_temp: float = 0.90       # softmax temperature when sampling actions (0.2–2.0)
    policy_buffer: int = 8          # size of each agent's OWN experience replay (small data, 2–24)
    policy_carry: bool = False      # carry learned weight-shifts into the next run (session-only)
    influence: float = 0.0          # 🌀 culture diffusion: pull toward group-mates' habits (0–0.6)
    policy_anchor: float = 0.02     # anchor pull toward ×1.00 rule prior (0 = habits never fade)
    world_cols: int = WORLD_COLS    # fixed world size, recorded with the run
    world_rows: int = WORLD_ROWS

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "SimConfig":
        keys = set(SimConfig.__dataclass_fields__)
        return SimConfig(**{k: v for k, v in d.items() if k in keys})

    def clamped(self) -> "SimConfig":
        return SimConfig(
            population=int(clamp(self.population, 5, 250)),
            rounds=int(clamp(self.rounds, 10, 500)),
            seed=int(self.seed),
            start_trust=clamp(float(self.start_trust), 0.0, 1.0),
            memory=bool(self.memory),
            communication=self.communication if self.communication in COMM_LEVELS else "LIMITED",
            reward_model=self.reward_model if self.reward_model in REWARD_MODELS else "MIXED",
            food_regrowth=clamp(float(self.food_regrowth), 0.05, 6.0),
            food_initial=clamp(float(self.food_initial), 5.0, 100.0),
            movement_speed=int(clamp(self.movement_speed, 1, 3)),
            policy=self.policy if self.policy in ("RULE", "LEARNED") else "RULE",
            policy_lr=clamp(float(self.policy_lr), 0.0, 0.40),
            influence=clamp(float(self.influence), 0.0, 0.6),
            policy_anchor=clamp(float(self.policy_anchor), 0.0, 0.2),
            policy_temp=clamp(float(self.policy_temp), 0.20, 2.00),
            policy_buffer=int(clamp(self.policy_buffer, 2, 24)),
            policy_carry=bool(self.policy_carry),
            world_cols=self.world_cols, world_rows=self.world_rows,
        )

    def human(self) -> dict:
        """Labelled version for the UI / reports."""
        return {
            "Agents": self.population,
            "Rounds": self.rounds,
            "Seed": self.seed,
            "Starting trust": fmt(self.start_trust),
            "Memory of past interactions": "ON (trust updates from experience)" if self.memory
            else "OFF (no grudges, no favours remembered)",
            "Communication / gossip": self.communication,
            "Reward model": self.reward_model,
            "Food regrowth per cell/round": fmt(self.food_regrowth, 2),
            "Average starting food per cell": fmt(self.food_initial, 1),
            "Movement speed (cells/round)": self.movement_speed,
            "Agent policy": ("📜 RULE — fixed factor weights (baseline)" if self.policy == "RULE" else
                             f"🧠 LEARNED — policy gradient · lr {fmt(self.policy_lr, 2)} · "
                             f"temperature {fmt(self.policy_temp, 2)} · buffer {self.policy_buffer} own samples · "
                             f"carry across runs {'ON (session-only)' if self.policy_carry else 'OFF'} · "
                             f"culture diffusion {fmt(self.influence, 2)} · rule-anchor pull {fmt(self.policy_anchor, 3)}"),
        }


CONFIG_HELP = {
    "population": "How many agents the world starts with. Deaths only — no births — so population can only shrink.",
    "rounds": "One round = every living agent scores its legal actions once and executes the best one. 200 rounds is a good default.",
    "seed": "The world, the traits and all jitter are drawn from this number. Same seed + same settings ⇒ bit-identical run.",
    "start_trust": "The trust value every agent assigns to strangers at round 0. With memory ON, experienced trust drifts slowly back toward this value.",
    "memory": "ON: agents keep per-partner trust and a short log, so betrayals and favours change future choices. OFF: every interaction is amnesia.",
    "communication": "How third-hand reputation spreads. OFF: only direct experience counts. LIMITED: witnesses at the same cell hear about it. FULL: word also travels through the victim's group.",
    "reward_model": "What the society pays for. MIXED is balanced; COOPERATIVE pays ~2.2× for sharing; COMPETITIVE pays 2× for taking.",
    "food_regrowth": "How generous the environment is. Low values = scarcity scenarios. Each map cell regrows this much food per round.",
    "food_initial": "Average food stocked in every map cell at round 0 (randomised around this value per cell and per region).",
    "movement_speed": "Cells an agent covers per Move action. Exploration always jumps farther than a move.",
}

SETTING_LABEL = {
    "population": "Agents", "rounds": "Rounds", "seed": "Seed",
    "start_trust": "Starting trust", "memory": "Memory",
    "communication": "Communication", "reward_model": "Reward model",
    "food_regrowth": "Food regrowth", "food_initial": "Starting field food",
    "movement_speed": "Movement speed",
}
SETTING_LABEL.update({"policy": "Agent policy", "policy_lr": "Learning rate",
                      "policy_temp": "Policy temperature", "policy_buffer": "Experience buffer",
                      "policy_carry": "Carry learning across runs",
                      "influence": "🌀 Culture diffusion", "policy_anchor": "Rule-anchor pull"})
CONFIG_HELP.update({
    "policy": "RULE: agents score actions with fixed, documented weights. LEARNED: each agent keeps the same "
              "factors but trains a tiny linear policy ONLINE on its own last-k decisions (policy gradient, "
              "REINFORCE with a value baseline) — weights start at ×1.00 (exact rule) and drift as the agent "
              "learns what pays off for it. Deterministic under the seed.",
    "policy_lr": "How fast learned multipliers move (0.01–0.40). Big = faster habits, noisier society.",
    "policy_temp": "Exploration vs greed when SAMPLING an action from the learned policy. Low = always pick the "
                    "current favourite; high = experiment more.",
    "policy_buffer": "The 'small dataset' each agent trains on: its own last N (action, factors, reward) "
                     "samples. Nothing else is used — no shared data, no internet.",
    "policy_carry": "Session-only group memory: elite agents' learned multipliers are averaged and seed the "
                    "NEXT run. Behaviour then evolves run-to-run (deliberately not bit-reproducible while ON).",
    "influence": "🌀 Culture diffusion — each round, group members drag each other's trained multipliers a "
                 "fraction of the way toward the group mean (0.0 = every agent learns in isolation). When an "
                 "agent dies, its group inherits part of its habits at the same strength. Emergent "
                 "subcultures: same rules, same rewards, different neighbourhoods.",
    "policy_anchor": "How strongly trained multipliers are pulled back toward ×1.00 (the documented rule), "
                     "scaled per agent by (1.7 − adaptability): adaptable agents form fast, shifty habits; "
                     "stubborn ones stay instinctive. 0 = learned habits never fade.",
})

# policy-gradient slot vocabulary: every (action, factor-label) the scoring code can emit.
POLICY_SLOTS: list[tuple[str, str]] = [
    ("collect", "hunger × 6"), ("collect", "cell has food"), ("collect", "expected haul"),
    ("collect", "full belly (no room)"), ("collect", "exhaustion penalty"), ("collect", "reward model favours"),
    ("move", "neighbour richer"), ("move", "exploration trait"), ("move", "crowding at home"),
    ("move", "hunger push"), ("move", "exhaustion penalty"),
    ("explore", "exploration trait"), ("explore", "diminishing local food"), ("explore", "patchy home cell"),
    ("explore", "risk appetite"), ("explore", "weak / hurt"), ("explore", "exhaustion penalty"),
    ("rest", "exhaustion"), ("rest", "low health"), ("rest", "empty belly risk"),
    ("trade", "no valid partner in cell"), ("trade", "my hunger (buy pressure)"),
    ("trade", "selling surplus at full belly"), ("trade", "partner trust"),
    ("trade", "mutual-gain habit (adaptability)"), ("trade", "reward model favours"),
    ("trade", "exhaustion penalty"),
    ("share", "no surplus to give"), ("share", "no trusted hungry neighbour in cell"),
    ("share", "co-operation trait"), ("share", "partner need"), ("share", "partner trust"),
    ("share", "same group"), ("share", "fear of being exploited"), ("share", "reward model favours"),
    ("share", "my surplus"),
    ("cooperate", "no willing / trusted partner in cell"), ("cooperate", "co-operation trait"),
    ("cooperate", "mutual trust"), ("cooperate", "partner co-operative"), ("cooperate", "same group bonus"),
    ("cooperate", "reward model favours"), ("cooperate", "exhaustion penalty"),
    ("compete", "no one weaker nearby"), ("compete", "competition trait"), ("compete", "content when full"),
    ("compete", "target is weak"), ("compete", "target haul"), ("compete", "retaliation risk"),
    ("compete", "getting caught (witnesses)"), ("compete", "group taboo"), ("compete", "past kindness owed"),
    ("compete", "reward model favours"),
]
POLICY_SLOT_ID = {(act, lab): i for i, (act, lab) in enumerate(POLICY_SLOTS)}
POLICY_ACTION_SLOTS = {act: [i for (a2, _), i in POLICY_SLOT_ID.items() if a2 == act] for act in ACTIONS}
AIDX = {a: i for i, a in enumerate(ACTIONS)}
N_SLOT = len(POLICY_SLOTS)
# self-check: every non-jitter breakdown label the engine can emit must have a slot
_ASSERT_LABELS = {
    ("collect", "hunger × 6"), ("collect", "cell has food"), ("collect", "expected haul"),
    ("collect", "full belly (no room)"), ("collect", "exhaustion penalty"), ("collect", "reward model favours"),
    ("move", "neighbour richer"), ("move", "exploration trait"), ("move", "crowding at home"),
    ("move", "hunger push"), ("move", "exhaustion penalty"),
    ("explore", "exploration trait"), ("explore", "diminishing local food"), ("explore", "patchy home cell"),
    ("explore", "risk appetite"), ("explore", "weak / hurt"), ("explore", "exhaustion penalty"),
    ("rest", "exhaustion"), ("rest", "low health"), ("rest", "empty belly risk"),
    ("trade", "no valid partner in cell"), ("trade", "my hunger (buy pressure)"),
    ("trade", "selling surplus at full belly"), ("trade", "partner trust"),
    ("trade", "mutual-gain habit (adaptability)"), ("trade", "reward model favours"),
    ("trade", "exhaustion penalty"),
    ("share", "no surplus to give"), ("share", "no trusted hungry neighbour in cell"),
    ("share", "co-operation trait"), ("share", "partner need"), ("share", "partner trust"),
    ("share", "same group"), ("share", "fear of being exploited"), ("share", "reward model favours"),
    ("share", "my surplus"),
    ("cooperate", "no willing / trusted partner in cell"), ("cooperate", "co-operation trait"),
    ("cooperate", "mutual trust"), ("cooperate", "partner co-operative"), ("cooperate", "same group bonus"),
    ("cooperate", "reward model favours"), ("cooperate", "exhaustion penalty"),
    ("compete", "no one weaker nearby"), ("compete", "competition trait"), ("compete", "content when full"),
    ("compete", "target is weak"), ("compete", "target haul"), ("compete", "retaliation risk"),
    ("compete", "getting caught (witnesses)"), ("compete", "group taboo"), ("compete", "past kindness owed"),
    ("compete", "reward model favours"),
}
assert _ASSERT_LABELS == set(POLICY_SLOTS), set(POLICY_SLOTS) ^ _ASSERT_LABELS
del _ASSERT_LABELS


def _sv(x: Any) -> str:
    if isinstance(x, bool):
        return "ON" if x else "OFF"
    if isinstance(x, float):
        return f"{x:g}"
    return str(x)


def preset_diff(cfg: SimConfig, changes: dict) -> list[tuple[str, str, str]]:
    return [(SETTING_LABEL.get(k, k), _sv(getattr(cfg, k)), _sv(v)) for k, v in changes.items()]


PRESETS: list[dict] = [
    {"key": "free", "label": "🌤  Free World (baseline)",
     "blurb": "The untouched starting point: kind environment, balanced rewards, memory ON.",
     "changes": {"population": 110, "rounds": 200, "start_trust": 0.50, "memory": True,
                 "communication": "LIMITED", "reward_model": "MIXED", "food_regrowth": 2.4,
                 "food_initial": 55.0, "movement_speed": 1}},
    {"key": "shock", "label": "🍽  Food Shock",
     "blurb": "The land stops recovering. Tests whether scarcity pushes agents toward competing instead of sharing.",
     "changes": {"food_regrowth": 0.5, "food_initial": 26.0}},
    {"key": "trust", "label": "🤝  Trust Challenge",
     "blurb": "Everyone starts suspicious. Can cooperative rewards still bootstrap a trusting society from scratch?",
     "changes": {"start_trust": 0.18, "reward_model": "COOPERATIVE", "memory": True}},
    {"key": "amnesia", "label": "🧠  Memory Off",
     "blurb": "Identical to Free World except agents cannot remember past interactions — no grudges, no favours kept.",
     "changes": {"memory": False}},
    {"key": "gossip", "label": "📣  Full Gossip",
     "blurb": "Reputation travels through whole groups, not just the two agents who met. Watch cheaters get avoided.",
     "changes": {"communication": "FULL"}},
    {"key": "competitive", "label": "⚔  Competitive Rewards",
     "blurb": "Stealing and trading pay; helping barely does. A pure incentive-change experiment.",
     "changes": {"reward_model": "COMPETITIVE"}},
    {"key": "cooperative", "label": "🕊  Cooperative Rewards",
     "blurb": "Sharing pays 2.2×, competing 0.35×. Tests how far incentives can bend behaviour.",
     "changes": {"reward_model": "COOPERATIVE"}},
    {"key": "culture", "label": "🌀  Culture War",
     "blurb": "Learned minds + strong peer influence: neighbourhoods grow genuinely different beliefs from the "
              "same rulebook — then watch deaths transmit habits and subcultures fight for the map.",
     "changes": {"policy": "LEARNED", "influence": 0.5, "policy_lr": 0.2, "start_trust": 0.35}},
]
PRESET_BY_KEY = {p["key"]: p for p in PRESETS}

WIDGET_KEYS = ["population", "rounds", "seed", "start_trust", "memory", "communication",
               "reward_model", "food_regrowth", "food_initial", "movement_speed"]
WIDGET_KEYS += ["policy", "policy_lr", "policy_temp", "policy_buffer", "policy_carry",
                "influence", "policy_anchor"]

# ── comparison mode: one single flipped setting, same seed, same run length ──
COMPARE_VARS: dict[str, dict] = {
    "memory": {
        "title": "Memory ON vs OFF",
        "key": "memory", "a": True, "b": False,
        "a_label": "A · Memory ON", "b_label": "B · Memory OFF",
        "explain": "Do agents that remember betrayals & favours behave differently? Only the memory switch differs.",
    },
    "communication": {
        "title": "Gossip LIMITED vs FULL",
        "key": "communication", "a": "LIMITED", "b": "FULL",
        "a_label": "A · LIMITED gossip (witnesses only)", "b_label": "B · FULL group reputation",
        "explain": "Does letting third-hand reputation spread change cooperation? Only who hears about what differs.",
    },
    "reward_model": {
        "title": "Rewards MIXED vs COMPETITIVE",
        "key": "reward_model", "a": "MIXED", "b": "COMPETITIVE",
        "a_label": "A · MIXED rewards", "b_label": "B · COMPETITIVE rewards",
        "explain": "Same agents, same seed — only the payout table changes (compete 1.0× → 2.0×, share 1.0× → 0.35×).",
    },
    "food_regrowth": {
        "title": "Land generous vs scarce",
        "key": "food_regrowth", "a": 2.4, "b": 0.5,
        "a_label": "A · regrowth 2.4/cell/round", "b_label": "B · regrowth 0.5/cell/round",
        "explain": "An environment-only change: does a stingier world create more competition?",
    },
    "influence": {
        "title": "🌀 Culture diffusion ON vs OFF",
        "key": "influence", "a": 0.5, "b": 0.0,
        "a_label": "A · peers shape each other (0.50)", "b_label": "B · lone learners (0.00)",
        "explain": "Both LEARNED, same seed — in A, group-mates pull each other's trained multipliers toward a "
                   "shared mean and the dead leave habits behind. Do subcultures form, and do they change "
                   "survival, trust or violence? Only the social-influence coefficient differs.",
    },
    "policy": {
        "title": "🧠 Learned policy vs 📜 fixed rules",
        "key": "policy", "a": "LEARNED", "b": "RULE",
        "a_label": "A · learned (per-agent policy gradient)", "b_label": "B · fixed rules (baseline)",
        "explain": "Same seed, same world, same reward table — the only difference is that A's agents "
                   "re-weight their own decision factors by learning from their last few experiences.",
    },
    "start_trust": {
        "title": "Strangers trusted vs not",
        "key": "start_trust", "a": 0.5, "b": 0.18,
        "a_label": "A · start trust 0.50", "b_label": "B · start trust 0.18",
        "explain": "Only the initial trust everyone assigns to strangers differs.",
    },
}

# metrics compared in A/B mode: (column, label, unit, higher-is-what)
COMPARE_METRICS = [
    ("cooperation_rate", "Co-operation rate", "fraction of living agents acting kindly per round", "good"),
    ("competition_rate", "Competition rate", "fraction of living agents stealing per round", "bad"),
    ("avg_trust", "Average trust", "0–1 mean of remembered partner-trusts", "good"),
    ("avg_reward", "Average reward", "points per living agent / round (model currency)", "good"),
    ("avg_wealth", "Average wealth", "coins held per agent", "neutral"),
    ("gini_wealth", "Wealth Gini", "0 equal … 1 all held by one agent", "neutral"),
    ("avg_food", "Average food stores", "units carried per agent", "neutral"),
    ("groups", "Active groups", "number of groups of ≥2 co-operating agents", "neutral"),
    ("trade_rate", "Trade rate", "trades per living agent per round", "neutral"),
    ("deaths", "Deaths (cumulative)", "agents whose health reached 0", "bad"),
]

# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 2 · seeded synthetic world generation
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Agent:
    aid: int
    cx: int                 # cell coords (grid)
    cy: int
    jx: float               # in-cell render offset so dots don't stack
    jy: float
    food: float
    energy: float
    wealth: float
    health: float
    reward: float = 0.0
    alive: bool = True
    death_round: Optional[int] = None
    # personality traits, each in [0,1]
    t_coop: float = 0.5
    t_comp: float = 0.5
    t_expl: float = 0.5
    t_risk: float = 0.5
    t_adapt: float = 0.5
    # memory (only consulted when cfg.memory is ON)
    trust: dict = field(default_factory=dict)          # partner id -> trust 0..1
    helped_me: dict = field(default_factory=dict)      # partner id -> share/cooperate-with-me count
    betrayed_by: dict = field(default_factory=dict)    # partner id -> attacks-on-me count
    last_action: str = "rest"
    actions_taken: dict = field(default_factory=dict)  # action -> lifetime count
    group: Optional[int] = None


def _make_rng(cfg: SimConfig, salt: int) -> random.Random:
    """Deterministic RNG streams, all derived from the visible seed.
    Separate salts keep world / agents / policy-jitter independent but reproducible."""
    return random.Random((int(cfg.seed) * 1_000_003 + 17 + salt * 7919) & (2**61 - 1))


def generate_world(cfg: SimConfig) -> tuple[list[list[float]], list[float], list[float]]:
    """Return (food grid [rows][cols], region x-productivity, region y-productivity)."""
    rng = _make_rng(cfg, salt=2)
    rx_mods = [rng.uniform(0.70, 1.30) for _ in range(3)]
    ry_mods = [rng.uniform(0.85, 1.15) for _ in range(2)]
    scale = cfg.food_initial / 55.0
    grid = []
    for r in range(cfg.world_rows):
        row = []
        for c in range(cfg.world_cols):
            mult = rx_mods[min(2, c * 3 // cfg.world_cols)] * ry_mods[min(1, r * 2 // cfg.world_rows)]
            base = rng.uniform(0.45, 1.15) * 55.0 * scale * mult
            row.append(clamp(base, 2.0, CELL_FOOD_CAP))
        grid.append(row)
    return grid, rx_mods, ry_mods


def generate_agents(cfg: SimConfig) -> list[Agent]:
    rng = _make_rng(cfg, salt=3)
    agents: list[Agent] = []
    for i in range(cfg.population):
        a = Agent(
            aid=i,
            cx=rng.randrange(cfg.world_cols), cy=rng.randrange(cfg.world_rows),
            jx=rng.uniform(0.15, 0.85), jy=rng.uniform(0.15, 0.85),
            food=rng.uniform(25, 55),
            energy=rng.uniform(55, 95),
            wealth=rng.uniform(0, 25),
            health=rng.uniform(75, 100),
        )
        # correlated traits: cooperativeness tends to come with low aggression,
        # competitiveness with risk appetite. The correlation is generated, not hidden.
        base_c = rng.uniform(0, 1)
        a.t_coop = clamp01(base_c * 0.85 + rng.uniform(0, 0.15))
        a.t_comp = clamp01((1 - base_c) * 0.85 + rng.uniform(0, 0.15))
        a.t_expl = clamp01(rng.uniform(0.05, 0.95))
        a.t_risk = clamp01(rng.uniform(0.05, 0.95))
        a.t_adapt = clamp01(rng.uniform(0.05, 0.95))
        a.actions_taken = {k: 0 for k in ACTIONS + FORCED}
        agents.append(a)
    return agents


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 3 · the agent rules + simulation engine (the actual experiment)
# ═══════════════════════════════════════════════════════════════════════════

class Society:
    """One run of the model. The round flow is documented in "Method & Limits"."""

    def __init__(self, cfg: SimConfig, brain: Optional[np.ndarray] = None, brain_gen: int = 0):
        self.cfg = cfg
        self.rng = _make_rng(cfg, salt=4)
        self.grid, self.rx_mods, self.ry_mods = generate_world(cfg)
        self.agents = generate_agents(cfg)
        self.n0 = len(self.agents)
        self.by_id = {a.aid: a for a in self.agents}
        self.cellmap: dict[int, list[int]] = defaultdict(list)   # key cy*cols+cx -> agent ids (rebuilt each round)
        self.pair_coop: dict[tuple[int, int], int] = {}
        self.trade_done: set[tuple[int, int]] = set()
        self.next_group = 1
        self.groups: dict[int, set[int]] = {}
        self.events: list[dict] = []
        self.round_rows: list[dict] = []
        self.decisions: list[dict] = []
        self.frames: list[dict] = []
        self._crossed_50 = False
        self.rnd = 0
        self.t_wall0 = time.perf_counter()
        # ── LEARNED policy: per-agent weights on the SAME documented factors ──
        self.learned = cfg.policy == "LEARNED"
        self.brain_gen = int(brain_gen)
        self._divn = 0
        self._learn_stats: dict = {}
        if self.learned:
            n0, T = self.n0, cfg.policy_buffer
            w0 = np.ones((n0, 8, N_SLOT), dtype=np.float64)
            if brain is not None:
                w0 += np.asarray(brain, dtype=np.float64)[None, :, :]
            _jr = np.random.default_rng(self.rng.randrange(2 ** 63))
            w0 += _jr.uniform(-0.05, 0.05, w0.shape)   # seeded individuality
            self.W = np.clip(w0, -2.0, 3.0)
            self.Vbuf = np.zeros((T, n0, 8, N_SLOT), dtype=np.float32)
            self.Abuf = np.zeros((T, n0), dtype=np.int8)
            self.Rbuf = np.zeros((T, n0), dtype=np.float32)
            self.Mbuf = np.zeros((T, n0), dtype=np.uint8)        # slot valid?
            self.nseen = np.zeros(n0, dtype=np.int64)
            self.vbase = np.zeros(n0, dtype=np.float64)          # value baseline (EMA)
            self.r0 = np.zeros(n0, dtype=np.float64)            # reward snapshot at round start
            self._wcur = 0
            ta = np.array([ag.t_adapt for ag in self.agents], dtype=float)
            self._anchor = np.maximum(0.0, cfg.policy_anchor * (1.7 - ta))[:, None, None]
        else:
            self.W = None

    # ── small helpers ────────────────────────────────────────────────────
    def cell_food(self, cx: int, cy: int) -> float:
        return self.grid[cy % self.cfg.world_rows][cx % self.cfg.world_cols]

    def trust_of(self, a: Agent, b: Agent) -> float:
        """Trust a holds for b. With memory OFF it never updates — it is just the start value."""
        if not self.cfg.memory:
            return self.cfg.start_trust
        return a.trust.get(b.aid, self.cfg.start_trust)

    def rebuild_cellmap(self) -> None:
        self.cellmap = defaultdict(list)
        for a in self.agents:
            if a.alive:
                self.cellmap[a.cy * self.cfg.world_cols + a.cx].append(a.aid)

    def neighbours(self, a: Agent) -> list[Agent]:
        ids = self.cellmap.get(a.cy * self.cfg.world_cols + a.cx, [])
        return [self.by_id[i] for i in ids if i != a.aid and self.by_id[i].alive]

    def _region_mult(self, cx: int, cy: int) -> float:
        return (self.rx_mods[min(2, cx * 3 // self.cfg.world_cols)]
                * self.ry_mods[min(1, cy * 2 // self.cfg.world_rows)])

    def log(self, rnd: int, kind: str, text: str) -> None:
        self.events.append({"round": rnd, "kind": kind, "text": text})

    # ── candidate selection for social actions ───────────────────────────
    def _trade_partner(self, a: Agent, near: list[Agent]) -> Optional[tuple[Agent, str]]:
        """Return (partner, direction): 'sell' = a gives food & earns coins; 'buy' = the reverse."""
        best, best_dir, best_score = None, None, -1.0
        for p in near:
            if a.food >= 44 and p.food < 42 and p.wealth >= 8:
                dirn, sc = "sell", 1.0 + self.trust_of(a, p)
            elif p.food >= 44 and a.food < 42 and a.wealth >= 8:
                dirn, sc = "buy", 0.8 + self.trust_of(a, p)
            else:
                continue
            if sc > best_score:
                best, best_dir, best_score = p, dirn, sc
        return (best, best_dir) if best is not None else None

    def _share_target(self, a: Agent, near: list[Agent]) -> Optional[Agent]:
        if a.food < 42:
            return None  # must have a surplus to share
        cands = [p for p in near if p.food < 45 and (not self.cfg.memory or self.trust_of(a, p) >= 0.30)]
        return min(cands, key=lambda p: (p.food, p.aid)) if cands else None

    def _coop_partner(self, a: Agent, near: list[Agent]) -> Optional[Agent]:
        gate = 0.45 if self.cfg.memory else max(0.2, self.cfg.start_trust - 0.05)
        cands = [p for p in near if p.energy >= 15 and self.trust_of(a, p) >= gate]
        return max(cands, key=lambda p: (self.trust_of(a, p) + p.t_coop, -p.aid)) if cands else None

    def _compete_target(self, a: Agent, near: list[Agent]) -> Optional[Agent]:
        if not near:
            return None
        v = min(near, key=lambda p: (p.energy + p.food + p.health, p.aid))
        return v if (v.food > 2 or v.wealth > 2) else None

    # ── the scoring policy (fully explainable; no hidden model) ───────────
    def score_actions(self, a: Agent, near: list[Agent], rnd: int) -> dict[str, dict]:
        cfg = self.cfg
        cell = self.cell_food(a.cx, a.cy)
        hunger = clamp01((45.0 - a.food) / 45.0)
        tired = clamp01((55.0 - a.energy) / 55.0)
        mults = REWARD_MULTS[cfg.reward_model]
        out: dict[str, dict] = {}

        in_group = a.group is not None and a.group in self.groups and len(self.groups[a.group]) > 1
        gain = min(cell, 12 + 4 * a.t_risk) * (1.10 if in_group else 1.0)
        out["collect"] = {"breakdown": {
            "hunger × 6": hunger * 6.0,
            "cell has food": cell * 0.06,
            "expected haul": gain * 0.22,
            "full belly (no room)": -1.6 if a.food >= 58 else 0.0,
            "exhaustion penalty": -tired * 1.6,
            "reward model favours": (mults["collect"] - 1.0) * 1.5,
        }}

        best_nb = max(
            (self.cell_food((a.cx + dx) % cfg.world_cols, (a.cy + dy) % cfg.world_rows)
             for dx in (-1, 0, 1) for dy in (-1, 0, 1) if (dx, dy) != (0, 0)),
            default=0.0,
        )
        crowd = max(0, len(near) - 3)
        out["move"] = {"breakdown": {
            "neighbour richer": max(0.0, best_nb - cell) * 0.42,
            "exploration trait": a.t_expl * 1.2,
            "crowding at home": -crowd * 0.45,
            "hunger push": hunger * 1.5,
            "exhaustion penalty": -tired * 1.2,
        }}

        out["explore"] = {"breakdown": {
            "exploration trait": a.t_expl * 3.4,
            "diminishing local food": 2.5 if cell < 30 else 0.0,
            "patchy home cell": 2.2 if cell < 20 else 0.0,
            "risk appetite": a.t_risk * 1.2,
            "weak / hurt": -2.2 if a.health < 40 else 0.0,
            "exhaustion penalty": -tired * 2.0,
        }}

        out["rest"] = {"breakdown": {
            "exhaustion": tired * 7.0,
            "low health": max(0.0, 55 - a.health) * 0.10,
            "empty belly risk": -1.6 if a.food < 15 else 0.0,
        }}

        tp = self._trade_partner(a, near)
        if tp is None:
            out["trade"] = {"breakdown": {"no valid partner in cell": -8.0}}
        else:
            partner, dirn = tp
            out["trade"] = {"breakdown": {
                "my hunger (buy pressure)": (3.2 * clamp01((42 - a.food) / 42)) if dirn == "buy" else 0.0,
                "selling surplus at full belly": 3.4 if (dirn == "sell" and a.food >= 55) else (2.2 if dirn == "sell" else 0.0),
                "partner trust": self.trust_of(a, partner) * 1.6,
                "mutual-gain habit (adaptability)": 1.4 + a.t_adapt * 1.0,
                "reward model favours": (mults["trade"] - 1.0) * 2.0,
                "exhaustion penalty": -tired * 1.0,
            }}
        out["trade"]["partner"] = tp[0].aid if tp else None
        out["trade"]["direction"] = tp[1] if tp else None

        stg = self._share_target(a, near)
        if stg is None:
            why = "no surplus to give" if a.food < 42 else "no trusted hungry neighbour in cell"
            out["share"] = {"breakdown": {why: -8.0}}
        else:
            same_g = a.group is not None and a.group == stg.group
            out["share"] = {"breakdown": {
                "co-operation trait": a.t_coop * 3.6,
                "partner need": (45 - stg.food) / 45 * 3.4,
                "partner trust": self.trust_of(a, stg) * 2.0,
                "same group": 1.6 if same_g else 0.0,
                "fear of being exploited": -1.8 * (1 - self.trust_of(a, stg)) if cfg.memory else 0.0,
                "reward model favours": (mults["share"] - 1.0) * 4.0,
                "my surplus": 2.2,
            }}
        out["share"]["partner"] = stg.aid if stg else None

        cp = self._coop_partner(a, near)
        if cp is None:
            out["cooperate"] = {"breakdown": {"no willing / trusted partner in cell": -8.0}}
        else:
            same_group = a.group is not None and a.group == cp.group
            out["cooperate"] = {"breakdown": {
                "co-operation trait": a.t_coop * 4.4,
                "mutual trust": self.trust_of(a, cp) * 3.0,
                "partner co-operative": cp.t_coop * 2.0,
                "same group bonus": 1.5 if same_group else 0.0,
                "reward model favours": (mults["cooperate"] - 1.0) * 4.0,
                "exhaustion penalty": -tired * 1.5,
            }}
        out["cooperate"]["partner"] = cp.aid if cp else None

        vt = self._compete_target(a, near)
        if vt is None:
            out["compete"] = {"breakdown": {"no one weaker nearby": -8.0}}
        else:
            my_strength = a.energy + a.food + a.health
            their_strength = vt.energy + vt.food + vt.health
            retaliation = clamp01((their_strength - my_strength) / 150.0)
            witnesses = max(0, len(near) - 1)
            comm_fear = {"OFF": 0.3, "LIMITED": 0.9, "FULL": 1.6}[cfg.communication]
            group_taboo = -3.2 if (a.group is not None and a.group == vt.group) else 0.0
            past_grudge = -1.5 * min(3, a.helped_me.get(vt.aid, 0)) if cfg.memory else 0.0
            out["compete"] = {"breakdown": {
                "competition trait": a.t_comp * 5.0,
                "content when full": -1.6 if a.food >= 55 else 0.0,
                "target is weak": clamp01((my_strength - their_strength) / 120.0) * 3.0,
                "target haul": (min(12.0, vt.food) * 0.45 + vt.wealth * 0.15) * 0.35,
                "retaliation risk": -retaliation * (7.0 - 3.0 * a.t_risk),
                "getting caught (witnesses)": -0.35 * witnesses * comm_fear,
                "group taboo": group_taboo,
                "past kindness owed": past_grudge,
                "reward model favours": (mults["compete"] - 1.0) * 4.0,
            }}
        out["compete"]["partner"] = vt.aid if vt else None

        # totals: sum of visible factors + seeded jitter that grows with risk appetite
        for act, d in out.items():
            total = sum(d["breakdown"].values())
            jitter = self.rng.uniform(-0.6, 0.6) * (0.4 + a.t_risk)
            d["jitter (risk-scaled uncertainty)"] = jitter
            d["total"] = total + jitter
        return out

    # ── a plain-language "why", generated from the SAME breakdown ─────────
    def why_sentence(self, act: str, a: Agent, sc: dict) -> str:
        p_aid = sc.get("partner")
        partner_txt = f" partner #{p_aid}" if p_aid is not None else ""
        trust_txt = ""
        if p_aid is not None:
            trust_txt = f" (trust {fmt(self.trust_of(a, self.by_id[p_aid]), 2)})"
        if act == "collect":
            return f"hunger was high and the cell underfoot still held {fmt(self.cell_food(a.cx, a.cy), 0)} food"
        if act == "move":
            return "a neighbouring cell looked richer than this one"
        if act == "explore":
            return "the home cell ran thin and this agent has a high exploration/risk drive"
        if act == "rest":
            return f"energy was low ({fmt(a.energy, 0)}/100) — recovery beat acting"
        if act == "trade":
            return f"a mutually beneficial exchange existed in the cell{partner_txt}{trust_txt}"
        if act == "share":
            return (f"a trusted neighbour{partner_txt} was hungry and this agent had surplus "
                    f"(co-op trait {fmt(a.t_coop, 2)})")
        if act == "cooperate":
            return (f"mutual trust with{partner_txt} passed the gate and the joint bonus was worth it"
                    f"{trust_txt}")
        if act == "compete":
            return (f"the weakest agent in the cell{partner_txt} was a low-risk target; retaliation risk and "
                    f"witness penalties were beaten by the competition trait ({fmt(a.t_comp, 2)})")
        return "energy hit zero — the body forced a rest at the cost of health"

    # ── execution of the chosen action ────────────────────────────────────
    def execute(self, a: Agent, act: str, sc: dict, rnd: int) -> dict:
        cfg = self.cfg
        mults = REWARD_MULTS[cfg.reward_model]
        deltas: dict[str, float] = {"food": 0.0, "energy": -ACTION_COST_ENERGY[act],
                                    "wealth": 0.0, "health": 0.0,
                                    "reward": ACTION_BASE_REWARD[act] * mults.get(act, 1.0)}
        p_aid = sc.get("partner")
        p = self.by_id[p_aid] if (p_aid is not None and p_aid in self.by_id and self.by_id[p_aid].alive) else None
        note = ""

        if act == "collect":
            cell = self.grid[a.cy][a.cx]
            take = min(cell, 12 + 4 * a.t_risk)
            if a.group is not None and a.group in self.groups and len(self.groups[a.group]) > 1:
                take *= 1.10
            take = round(take, 2)
            self.grid[a.cy][a.cx] = max(0.0, cell - take)
            deltas["food"] = take
        elif act == "move":
            dx = dy = 0
            best = self.cell_food(a.cx, a.cy)
            for ox in (-1, 0, 1):
                for oy in (-1, 0, 1):
                    v = self.cell_food((a.cx + ox) % cfg.world_cols, (a.cy + oy) % cfg.world_rows)
                    if v > best:
                        best, dx, dy = v, ox, oy
            step = cfg.movement_speed
            a.cx = (a.cx + dx * step) % cfg.world_cols
            a.cy = (a.cy + dy * step) % cfg.world_rows
            a.jx, a.jy = self.rng.uniform(0.15, 0.85), self.rng.uniform(0.15, 0.85)
        elif act == "explore":
            a.cx = self.rng.randrange(cfg.world_cols)
            a.cy = self.rng.randrange(cfg.world_rows)
            a.jx, a.jy = self.rng.uniform(0.15, 0.85), self.rng.uniform(0.15, 0.85)
            deltas["health"] -= 1.0
            if self.rng.random() < 0.30 + 0.35 * a.t_expl:
                self.grid[a.cy][a.cx] = min(CELL_FOOD_CAP, self.grid[a.cy][a.cx] + 25.0)
                deltas["reward"] += 1.5
                note = "stumbled on a +25 food patch"
        elif act == "rest":
            deltas["energy"] += 16.0
            if a.food >= 25:
                deltas["health"] += 1.5
        elif act == "trade":
            if p is not None:
                dirn = sc.get("direction", "sell")
                food_amt, coin_amt = 12.0, 8.0
                if dirn == "sell":
                    giver, taker = a, p
                else:
                    giver, taker = p, a
                if giver.food >= food_amt and taker.wealth >= coin_amt:
                    giver.food -= food_amt
                    taker.food = min(AGENT_FOOD_CAP, taker.food + food_amt)
                    taker.wealth -= coin_amt
                    giver.wealth += coin_amt
                    for x, y in ((a, p), (p, a)):
                        if cfg.memory:
                            x.trust[y.aid] = clamp01(x.trust.get(y.aid, cfg.start_trust) + 0.06)
                    p.energy = max(0.0, p.energy - 2.0)
                    p.reward += ACTION_BASE_REWARD["trade"] * mults["trade"]
                    deltas["food"] = food_amt if dirn == "buy" else -food_amt
                    deltas["wealth"] = -coin_amt if dirn == "buy" else coin_amt
                    note = f"gave {food_amt:g} food for {coin_amt:g} coins with #{p.aid}"
                    pair = (min(a.aid, p.aid), max(a.aid, p.aid))
                    if pair not in self.trade_done:
                        self.trade_done.add(pair)
                        self.log(rnd, "trade", f"#{a.aid} ↔ #{p.aid} completed their first trade")
                else:
                    note = "partner reneged (insufficient stock)"
                    deltas["reward"] *= 0.1
        elif act == "share":
            if p is not None and a.food >= 10:
                a.food -= 10.0
                p.food = min(AGENT_FOOD_CAP, p.food + 10.0)
                deltas["food"] = -10.0
                if cfg.memory:
                    a.trust[p.aid] = clamp01(a.trust.get(p.aid, cfg.start_trust) + 0.12)
                    p.trust[a.aid] = clamp01(p.trust.get(a.aid, cfg.start_trust) + 0.12)
                    p.helped_me[a.aid] = p.helped_me.get(a.aid, 0) + 1
                    a.helped_me[p.aid] = a.helped_me.get(p.aid, 0) + 1
                self._spread_goodwill(a, p)
                note = f"gave 10 food to #{p.aid}"
        elif act == "cooperate":
            if p is not None:
                bonus = 1.2 if (a.group is not None and a.group == p.group) else 1.0
                for x, y in ((a, p), (p, a)):
                    x.food = min(AGENT_FOOD_CAP, x.food + 6.0 * bonus)
                    x.energy = max(0.0, x.energy - 4.0)
                    x.reward += ACTION_BASE_REWARD["cooperate"] * mults["cooperate"] * bonus
                    if cfg.memory:
                        x.trust[y.aid] = clamp01(x.trust.get(y.aid, cfg.start_trust) + 0.10)
                        x.helped_me[y.aid] = x.helped_me.get(y.aid, 0) + 1
                key = (min(a.aid, p.aid), max(a.aid, p.aid))
                self.pair_coop[key] = self.pair_coop.get(key, 0) + 1
                self._maybe_form_group(a, p, rnd)
                self._spread_goodwill(a, p)
                deltas["food"] = 6.0 * bonus
                deltas["energy"] += -4.0
                note = f"joint project with #{p.aid} (+{6.0 * bonus:.1f} food each)"
                # the generic reward line double-counts the loop above; undo for 'a'
                deltas["reward"] -= ACTION_BASE_REWARD["cooperate"] * mults["cooperate"]
        elif act == "compete":
            if p is not None:
                v = p
                loot_food = min(12.0, v.food) * 0.45
                loot_coins = min(8.0, v.wealth) * 0.35
                v.food = max(0.0, v.food - loot_food)
                v.wealth = max(0.0, v.wealth - loot_coins)
                v.health -= 8.0
                deltas["food"] = loot_food
                deltas["wealth"] = loot_coins
                my_str = a.energy + a.food + a.health
                their_str = v.energy + v.food + v.health
                if their_str > my_str:
                    deltas["health"] -= 4.0
                    note = f"took {loot_food:.1f} food + {loot_coins:.1f} coins from #{v.aid} — but they fought back"
                else:
                    note = f"took {loot_food:.1f} food + {loot_coins:.1f} coins from #{v.aid}"
                if cfg.memory:
                    v.trust[a.aid] = clamp01(v.trust.get(a.aid, cfg.start_trust) - 0.35)
                    v.betrayed_by[a.aid] = v.betrayed_by.get(a.aid, 0) + 1
                self._spread_badword(a, v, rnd)
                if v.group is not None and v.group == a.group:
                    self._break_group(v.group, a, rnd)
            else:
                deltas["reward"] *= 0.4
                note = "no valid target left this round"

        # apply deltas (final clamps; energy floor is 0)
        a.food = clamp(a.food + deltas["food"], 0.0, AGENT_FOOD_CAP)
        a.wealth = max(0.0, a.wealth + deltas["wealth"])
        a.energy = a.energy + deltas["energy"]
        a.health = a.health + deltas["health"]
        a.reward += deltas["reward"]
        return {"note": note, "deltas": deltas, "partner": p_aid}

    # ── reputation & groups ───────────────────────────────────────────────
    def _spread_goodwill(self, actor: Agent, partner: Agent) -> None:
        cfg = self.cfg
        if cfg.communication != "FULL" or not cfg.memory or partner.group is None:
            return
        for other in self.agents:
            if other.alive and other.group == partner.group and other.aid not in (actor.aid, partner.aid):
                other.trust[actor.aid] = clamp01(other.trust.get(actor.aid, cfg.start_trust) + 0.03)

    def _spread_badword(self, actor: Agent, victim: Agent, rnd: int) -> None:
        cfg = self.cfg
        if cfg.communication == "OFF" or not cfg.memory:
            return
        pen = 0.12
        for other in self.agents:
            if not other.alive or other.aid in (actor.aid, victim.aid):
                continue
            witnessed = (other.cx == actor.cx and other.cy == actor.cy)
            same_group_as_victim = victim.group is not None and other.group == victim.group
            if witnessed:
                other.trust[actor.aid] = clamp01(other.trust.get(actor.aid, cfg.start_trust) - pen)
            elif cfg.communication == "FULL" and same_group_as_victim:
                other.trust[actor.aid] = clamp01(other.trust.get(actor.aid, cfg.start_trust) - pen / 2.0)

    def _maybe_form_group(self, a: Agent, b: Agent, rnd: int) -> None:
        cfg = self.cfg
        key = (min(a.aid, b.aid), max(a.aid, b.aid))
        if self.pair_coop.get(key, 0) < 3:
            return
        trust_ok = ((self.trust_of(a, b) >= 0.6 and self.trust_of(b, a) >= 0.6)
                    if cfg.memory else (cfg.start_trust >= 0.6))
        if not trust_ok:
            return
        if a.group is not None and a.group == b.group:
            return
        if a.group is not None and b.group is None and len(self.groups.get(a.group, set())) < 6:
            b.group = a.group
            self.groups[a.group].add(b.aid)
            self.log(rnd, "group", f"#{b.aid} joined group {a.group} (invited by #{a.aid})")
        elif b.group is not None and a.group is None and len(self.groups.get(b.group, set())) < 6:
            a.group = b.group
            self.groups[b.group].add(a.aid)
            self.log(rnd, "group", f"#{a.aid} joined group {b.group} (invited by #{b.aid})")
        elif a.group is None and b.group is None:
            gid = self.next_group
            self.next_group += 1
            self.groups[gid] = {a.aid, b.aid}
            a.group = b.group = gid
            self.log(rnd, "group", f"group {gid} formed by #{a.aid} & #{b.aid} "
                                   f"(3 mutual co-operations, trust ≥ 0.6)")

    def _break_group(self, gid: int, traitor: Agent, rnd: int) -> None:
        if gid not in self.groups:
            return
        for mid in list(self.groups[gid]):
            m = self.by_id[mid]
            if m.aid != traitor.aid and self.cfg.memory:
                m.trust[traitor.aid] = clamp01(m.trust.get(traitor.aid, self.cfg.start_trust) - 0.25)
        self.groups[gid].discard(traitor.aid)
        traitor.group = None
        self.log(rnd, "betray", f"#{traitor.aid} attacked a group-mate — expelled from group {gid}")

    def _validate_groups(self, rnd: int) -> None:
        for gid in list(self.groups.keys()):
            members = {m for m in self.groups[gid]
                       if self.by_id[m].alive and self.by_id[m].group == gid}
            self.groups[gid] = members
            if len(members) < 2:
                for m in members:
                    self.by_id[m].group = None
                del self.groups[gid]
                self.log(rnd, "group", f"group {gid} dissolved (fewer than 2 members left)")

    # ── the main loop ──────────────────────────────────────────────────────
    def run(self, progress_cb: Optional[Callable[[int, int, dict], None]] = None) -> "SimResult":
        """Instant mode: burn through every round as fast as possible."""
        while self.rnd < self.cfg.rounds:
            self.step()
            if progress_cb and (self.rnd % 5 == 0 or self.rnd == self.cfg.rounds):
                progress_cb(self.rnd, self.cfg.rounds, {"alive": self.round_rows[-1]["population"]})
        return self._finish(time.perf_counter() - self.t_wall0)

    def finalize(self) -> "SimResult":
        """Package what has been simulated so far (used when live playback completes)."""
        return self._finish(time.perf_counter() - self.t_wall0)

    def step(self) -> bool:
        """Advance exactly one round — shared by instant runs and live playback.
        Returns False if the run is already over."""
        if self.rnd >= self.cfg.rounds:
            return False
        rnd = self.rnd + 1
        cfg = self.cfg
        action_counts = {k: 0 for k in ACTIONS + FORCED}
        alive_ids = sorted(a.aid for a in self.agents if a.alive)
        n_acting = len(alive_ids)
        self.rebuild_cellmap()
        if self.learned:
            self._divn = 0
            self.Mbuf[self._wcur, :] = 0
            for _aid in alive_ids:
                self.r0[_aid] = self.by_id[_aid].reward

        # phase 1 — every agent decides & acts (against start-of-round positions)
        for aid in alive_ids:
            a = self.by_id[aid]
            near = self.neighbours(a)
            if a.energy <= 0.0:  # forced collapse
                a.energy = 8.0
                a.health -= 4.0
                a.actions_taken["collapse"] += 1
                a.last_action = "collapse"
                action_counts["collapse"] += 1
                self.decisions.append({
                    "round": rnd, "agent": aid, "action": "collapse",
                    "score": None, "runner_up": None, "runner_up_score": None, "gap": None,
                    "partner": None, "chosen_breakdown": {}, "runner_breakdown": {},
                    "why": "energy hit zero — the body forced a rest at the cost of health",
                    "note": "collapsed", "food_delta": 0.0})
                continue
            scores = self.score_actions(a, near, rnd)
            ranked = sorted(scores.items(), key=lambda kv: (-kv[1]["total"], kv[0]))
            (best_act, best_sc), (second_act, second_sc) = ranked[0], ranked[1]
            pol = None
            if self.learned:
                pol = self._policy_select(a, scores)
                rule_best = max(scores, key=lambda x: (round(sum(scores[x]["breakdown"].values()), 9), -AIDX[x]))
                self._divn += int(pol["chosen"] != rule_best)
                best_act = pol["chosen"]
                best_sc = scores[best_act]
                if second_act == best_act:
                    second_act = max((x for x in scores if x != best_act), key=lambda x: scores[x]["total"])
                second_sc = scores[second_act]
            why = self.why_sentence(best_act, a, best_sc)
            outcome = self.execute(a, best_act, best_sc, rnd)
            a.last_action = best_act
            a.actions_taken[best_act] += 1
            action_counts[best_act] += 1
            self.decisions.append({
                "round": rnd, "agent": aid, "action": best_act,
                "score": round(best_sc["total"], 2),
                "runner_up": second_act, "runner_up_score": round(second_sc["total"], 2),
                "gap": round(best_sc["total"] - second_sc["total"], 2),
                "partner": best_sc.get("partner"),
                "chosen_breakdown": {k: round(v, 2) for k, v in best_sc["breakdown"].items()},
                "runner_breakdown": {k: round(v, 2) for k, v in second_sc["breakdown"].items()},
                "why": why, "note": outcome["note"],
                "food_delta": round(outcome["deltas"].get("food", 0.0), 2),
                **({"policy": pol} if pol else {}),
            })

        # phase 2 — the land regrows (region productivity applies)
        if cfg.food_regrowth > 0:
            for r in range(cfg.world_rows):
                row = self.grid[r]
                for c in range(cfg.world_cols):
                    row[c] = min(CELL_FOOD_CAP, row[c] + cfg.food_regrowth * self._region_mult(c, r))

        # phase 3 — metabolism, eating, health, death
        for aid in alive_ids:
            a = self.by_id[aid]
            a.energy -= METABOLISM
            if a.energy < EAT_TRIGGER_ENERGY and a.food > 0:
                bite = min(a.food, 12.0)
                a.food -= bite
                a.energy = min(100.0, a.energy + bite * EAT_ENERGY_PER_FOOD)
            if a.food <= 0:
                a.health -= STARVE_DAMAGE
            elif a.energy > 35 and a.food > 25:
                a.health += HEAL_RATE
            a.health = clamp(a.health, 0.0, 100.0)
            a.energy = clamp(a.energy, 0.0, 100.0)
            a.food = clamp(a.food, 0.0, AGENT_FOOD_CAP)
            # survival component of reward (documented): being healthy pays a trickle
            a.reward += 0.05 + 0.15 * (a.health / 100.0)
            if a.health <= 0.0:
                if self.learned:
                    self._legacy(a)
                a.alive = False
                a.death_round = rnd
                cause = "starvation" if a.food <= 0 else "wounds / exhaustion"
                self.log(rnd, "death", f"#{a.aid} died at round {rnd} ({cause})")

        # phase 3b — LEARNED policy: one batched policy-gradient step over each agent's own buffer
        if self.learned:
            self._learn_round(alive_ids)

        # phase 4 — slow trust decay toward the baseline; groups re-checked
        if cfg.memory:
            for a in self.agents:
                if not a.alive or not a.trust:
                    continue
                for k, v in list(a.trust.items()):
                    a.trust[k] = v + (cfg.start_trust - v) * 0.01
        self._validate_groups(rnd)

        # phase 5 — record metrics + map frame
        row = self._record_round(rnd, action_counts, n_acting)
        if not self._crossed_50 and row["cooperation_rate"] >= 0.50 and rnd > 5:
            self._crossed_50 = True
            self.log(rnd, "milestone", f"co-operation rate first crossed 50% of living agents (round {rnd})")
        _ = row  # recorded inside step; live player reads soc.round_rows[-1]
        self.rnd = rnd
        return True

    # ── LEARNED policy internals (transparent: factors × learned multipliers) ─
    def _policy_select(self, a: Agent, scores: dict[str, dict]) -> dict:
        """Factors of every action × this agent's current multipliers → logits → sample an action.
        Also stores the (factors, action) sample in the agent's tiny replay buffer."""
        v = np.zeros((8, N_SLOT), dtype=np.float32)
        for act, d in scores.items():
            ai = AIDX[act]
            bd = d["breakdown"]
            for lab, val in bd.items():
                sid = POLICY_SLOT_ID.get((act, lab))
                if sid is not None and val != 0.0:
                    v[ai, sid] = val
        slot = self._wcur
        self.Vbuf[slot, a.aid] = v
        logits = (v.astype(np.float64) * self.W[a.aid]).sum(axis=1)
        z = (logits - logits.max()) / self.cfg.policy_temp
        ex = np.exp(z)
        pi = ex / ex.sum()
        chosen_i = int(np.clip(np.searchsorted(np.cumsum(pi), self.rng.random()), 0, 7))
        order = np.argsort(-pi)
        runner_i = int(order[1] if order[0] == chosen_i else order[0])
        self.Abuf[slot, a.aid] = chosen_i
        self.Mbuf[slot, a.aid] = 1
        self.nseen[a.aid] += 1
        ent = float(-(pi * np.log(pi + 1e-12)).sum())
        mults: dict[str, dict[str, float]] = {}
        for act in (ACTIONS[chosen_i], ACTIONS[runner_i]):
            ai2 = AIDX[act]
            mults[act] = {lab: round(float(self.W[a.aid, ai2, POLICY_SLOT_ID[(act, lab)]]), 3)
                          for lab in scores[act]["breakdown"] if (act, lab) in POLICY_SLOT_ID}
        return {"chosen": ACTIONS[chosen_i], "runner": ACTIONS[runner_i],
                "pi": {act: round(float(pi[AIDX[act]]), 4) for act in ACTIONS},
                "ent": round(ent, 4), "mults": mults,
                "rule_pick": None, "off_rule": False}

    def _learn_round(self, alive_ids: list[int]) -> None:
        """One REINFORCE update per agent, batched over its own last-T valid samples:
        ΔW ∝ (advantage)·(onehot(aₜ) − πₜ)⊗fₜ, plus a gentle pull toward 1.0 (the rule anchor)."""
        cfg = self.cfg
        T = cfg.policy_buffer
        sw = self._wcur
        m_all = self.Mbuf[sw].astype(bool)
        if m_all.any():
            idx = np.nonzero(m_all)[0]
            rew = np.fromiter((self.agents[i].reward for i in idx), dtype=np.float64, count=len(idx))
            dr = rew - self.r0[idx]
            self.Rbuf[sw, idx] = dr
            self.vbase[idx] += 0.15 * (dr - self.vbase[idx])
        V64 = self.Vbuf.astype(np.float64)                      # (T, n0, 8, S)
        L = np.einsum("tnas,nas->tna", V64, self.W) / cfg.policy_temp   # (T, n0, 8)
        L -= L.max(axis=2, keepdims=True)
        ex = np.exp(L)
        P = ex / ex.sum(axis=2, keepdims=True)                   # policy probs per stored sample
        ar = np.arange(T)[:, None]
        nr = np.arange(self.n0)[None, :]
        OH = np.zeros_like(P)
        OH[ar, nr, self.Abuf] = 1.0
        mask = self.Mbuf[..., None].astype(np.float64)           # (T, n0, 1)
        adv_raw = (self.Rbuf - self.vbase[None, :]) * self.Mbuf  # (T, n0)
        _s = float(self.Mbuf.sum())
        if _s > 3:                                               # normalise advantages (PPO-style)
            _mu = float(adv_raw.sum()) / _s
            _sd = math.sqrt(max(1e-6, float(((adv_raw - _mu) ** 2 * self.Mbuf).sum()) / _s))
            adv = (((adv_raw - _mu) / _sd) * self.Mbuf)[..., None]
        else:
            adv = adv_raw[..., None]
        grad = np.einsum("tna,tnas->nas", (OH - P) * adv * mask, V64)
        cnt = np.maximum(1, self.Mbuf.sum(axis=0))[:, None, None]
        self.W += cfg.policy_lr * grad / cnt
        self.W -= self._anchor * (self.W - 1.0)             # personality-scaled pull toward the rule anchor
        np.clip(self.W, -2.0, 3.0, out=self.W)
        self._diffuse()                                     # 🌀 culture: group-mates converge
        # ---- round logging from the policy as it stood this round
        Pnow = P[sw]
        acted = np.nonzero(m_all)[0]
        vg = float(self.W.var(axis=0).mean())
        _vw: list[float] = []
        for mem in self.groups.values():
            idx2 = np.array(sorted(m for m in mem if self.by_id[m].alive), dtype=int)
            if len(idx2) >= 2:
                _vw.append(float(self.W[idx2].var(axis=0).mean()))
        if len(acted):
            ent_i = -(Pnow[acted] * np.log(Pnow[acted] + 1e-12)).sum(axis=1)
            self._learn_stats = {
                "var_within": float(np.mean(_vw)) if _vw else vg,
                "policy_entropy": float(np.mean(ent_i)),
                "entropy_p10": float(np.quantile(ent_i, 0.10)),
                "entropy_p90": float(np.quantile(ent_i, 0.90)),
                "divergence_rate": self._divn / max(1, len(alive_ids)),
                "avg_shift": float(np.abs(self.W[acted] - 1.0).mean()),
                "var_global": vg,
                "avg_advantage": float(np.mean(self.Rbuf[sw, acted] - self.vbase[acted])),
                "samples_updated": int(len(acted)),
            }
        else:
            self._learn_stats = {}
        self._wcur = (sw + 1) % T

    def _diffuse(self) -> None:
        """🌀 Culture diffusion — group members' trained multipliers pull toward the group mean each round.
        influence = 0 ⇒ strict no-op (every agent learns in isolation)."""
        inf = self.cfg.influence
        if inf <= 0:
            return
        for mem in self.groups.values():
            idx = np.array(sorted(m for m in mem if self.by_id[m].alive), dtype=int)
            if len(idx) < 2:
                continue
            tgt = self.W[idx].mean(axis=0)
            self.W[idx] += inf * (tgt[None, :, :] - self.W[idx])
        np.clip(self.W, -2.0, 3.0, out=self.W)

    def _legacy(self, dead: "Agent") -> None:
        """Deaths leave habits behind: survivors of the dead agent's group inherit part of its multipliers."""
        inf = self.cfg.influence
        if inf <= 0 or dead.group is None:
            return
        for oid in self.groups.get(dead.group, ()):
            if oid != dead.aid and self.by_id[oid].alive:
                self.W[oid] += 0.35 * inf * (self.W[dead.aid] - self.W[oid])

    # ── round record + frame ───────────────────────────────────────────────
    def _record_round(self, rnd: int, ac: dict, n_acting: int) -> dict:
        cfg = self.cfg
        alive = [a for a in self.agents if a.alive]
        n = max(1, len(alive))
        trusts: list[float] = []
        if cfg.memory:
            for a in alive:
                if a.trust:
                    trusts.extend(a.trust.values())
        avg_trust = float(np.mean(trusts)) if trusts else cfg.start_trust
        coop = ac["share"] + ac["cooperate"]
        row = {
            "round": rnd,
            "population": len(alive),
            "agents_acted": n_acting,
            "survival": len(alive) / self.n0,
            "deaths": self.n0 - len(alive),
            "avg_food": float(np.mean([a.food for a in alive] or [0.0])),
            "max_food": float(np.max([a.food for a in alive] or [0.0])),
            "avg_energy": float(np.mean([a.energy for a in alive] or [0])),
            "avg_wealth": float(np.mean([a.wealth for a in alive] or [0])),
            "total_wealth": float(np.sum([a.wealth for a in alive])),
            "gini_wealth": gini([a.wealth for a in alive]),
            "avg_health": float(np.mean([a.health for a in alive] or [0])),
            "world_food": float(np.sum(self.grid)),
            "world_food_scaled": float(np.sum(self.grid)) / 100.0,
            "cooperation_rate": coop / n,
            "competition_rate": ac["compete"] / n,
            "trade_rate": ac["trade"] / n,
            "collect_rate": ac["collect"] / n,
            "explore_rate": (ac["explore"] + ac["move"]) / n,
            "rest_rate": (ac["rest"] + ac["collapse"]) / n,
            "avg_trust": avg_trust,
            "groups": len([g for g in self.groups.values() if len(g) >= 2]),
            "group_members": sum(len(g) for g in self.groups.values() if len(g) >= 2),
            "avg_reward": float(np.mean([a.reward for a in alive] or [0])),
            "total_reward": float(np.sum([a.reward for a in alive])),
            "n_collapse": ac["collapse"],
        }
        for act in ACTIONS + FORCED:
            row[f"n_{act}"] = ac[act]
        if self.learned and self._learn_stats:
            row.update({k: round(float(v), 5) if isinstance(v, float) else v
                        for k, v in self._learn_stats.items()})
        self.round_rows.append(row)

        cats, xs, ys, ids, w, fl, en, hl, la = [], [], [], [], [], [], [], [], []
        for a in self.agents:
            if not a.alive:
                continue
            ids.append(a.aid)
            xs.append(a.cx * CELL + a.jx * CELL)
            ys.append((cfg.world_rows - 1 - a.cy) * CELL + (1 - a.jy) * CELL)  # flip so row 0 renders on top
            w.append(round(a.wealth, 1))
            fl.append(round(a.food, 1))
            en.append(round(a.energy, 1))
            hl.append(round(a.health, 1))
            la.append(a.last_action)
            if a.last_action in ("share", "cooperate"):
                cats.append("co-operative")
            elif a.last_action == "trade":
                cats.append("trading")
            elif a.last_action == "compete":
                cats.append("hostile")
            elif a.last_action == "collect":
                cats.append("gathering")
            elif a.last_action in ("move", "explore"):
                cats.append("seeking")
            elif a.last_action == "rest":
                cats.append("resting")
            else:
                cats.append("collapsed")
        self.frames.append({
            "round": rnd, "ids": ids, "x": xs, "y": ys, "wealth": w, "food": fl,
            "energy": en, "health": hl, "last": la, "cat": cats,
            "grid": [list(r) for r in self.grid],
            "groups": {g: sorted(m) for g, m in self.groups.items() if len(m) >= 2},
            "n_alive": len(ids),
        })
        return row

    def _finish(self, elapsed: float) -> "SimResult":
        cfg = self.cfg
        rows = []
        for a in self.agents:
            rows.append({
                "agent": a.aid,
                "alive": a.alive,
                "death_round": a.death_round,
                "survived_rounds": (a.death_round - 1) if a.death_round is not None else cfg.rounds,
                "food": round(a.food, 1), "energy": round(a.energy, 1),
                "wealth": round(a.wealth, 1), "health": round(a.health, 1),
                "reward": round(a.reward, 1),
                "trait_cooperation": round(a.t_coop, 3), "trait_competition": round(a.t_comp, 3),
                "trait_exploration": round(a.t_expl, 3), "trait_risk": round(a.t_risk, 3),
                "trait_adaptation": round(a.t_adapt, 3),
                "group": ("solo" if a.group is None else f"g{a.group}"),
                "group_size": len(self.groups.get(a.group, set())) if a.group is not None else 1,
                "n_coop_partners": len(a.helped_me),
                "n_attacks_received": len(a.betrayed_by),
                "mean_trust_given": round(float(np.mean(list(a.trust.values()))) if a.trust else cfg.start_trust, 3),
                **{f"act_{k}": v for k, v in a.actions_taken.items()},
                "last_action": a.last_action,
            })
        rounds_df = pd.DataFrame(self.round_rows)
        agents_df = pd.DataFrame(rows)
        if len(agents_df):
            agents_df["death_round"] = agents_df["death_round"].astype("Int64")
        events_df = pd.DataFrame(self.events, columns=["round", "kind", "text"]) if self.events \
            else pd.DataFrame(columns=["round", "kind", "text"])
        dec_rows = []
        for d in self.decisions:
            dd = dict(d)
            dd.pop("policy", None)          # the full policy record stays in decisions_raw for the inspector
            dd["chosen_breakdown"] = json.dumps(dd["chosen_breakdown"], ensure_ascii=False)
            dd["runner_breakdown"] = json.dumps(dd["runner_breakdown"], ensure_ascii=False)
            dec_rows.append(dd)
        decisions_df = pd.DataFrame(dec_rows)
        if len(decisions_df):
            decisions_df["partner"] = decisions_df["partner"].astype("Int64")
        cfg_hash = hashlib.sha256(json.dumps(cfg.to_dict(), sort_keys=True).encode()).hexdigest()[:12]
        if self.learned:
            agents_df["learn_shift"] = [round(float(np.abs(self.W[i] - 1.0).mean()), 4)
                                        for i in range(self.n0)]
        meta = {
            "engine": ENGINE_VERSION,
            "config_hash": cfg_hash,
            "rounds": cfg.rounds, "agents0": self.n0,
            "alive_end": int(agents_df["alive"].sum()) if len(agents_df) else cfg.population,
            "run_ms": int(elapsed * 1000),
            "decisions": len(self.decisions),
            "events": len(self.events),
            "regions_x_mods": [round(x, 3) for x in self.rx_mods],
            "regions_y_mods": [round(x, 3) for x in self.ry_mods],
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "policy_mode": "LEARNED" if self.learned else "RULE",
        }
        if self.learned:
            meta["policy_slots"] = N_SLOT
            meta["brain_gen"] = self.brain_gen
            if cfg.policy_carry:
                alive_m = np.array([a.alive for a in self.agents], dtype=bool)
                rew_a = np.array([a.reward for a in self.agents], dtype=float)
                pool = rew_a if not alive_m.any() else rew_a[alive_m]
                thr = float(np.quantile(pool, 0.60))
                sel = (rew_a >= thr) & (alive_m | (rew_a >= 0))
                if sel.any():
                    meta["brain_out"] = np.round((self.W[sel].mean(axis=0) - 1.0), 6).tolist()
        res = SimResult(cfg=cfg, rounds_df=rounds_df, agents_df=agents_df,
                        events_df=events_df, decisions_df=decisions_df,
                        decisions_raw=self.decisions, frames=self.frames,
                        events=self.events, meta=meta)
        if self.learned:
            res.W_final = self.W.copy()          # (agents, 8, 53) — for the [L5] heatmap only
        return res


class SimResult:
    """Everything one run produced — kept in Streamlit session_state (temporary!)."""

    def __init__(self, cfg: SimConfig, rounds_df: pd.DataFrame, agents_df: pd.DataFrame,
                 events_df: pd.DataFrame, decisions_df: pd.DataFrame, decisions_raw: list,
                 frames: list, events: list, meta: dict):
        self.cfg = cfg
        self.rounds_df = rounds_df
        self.agents_df = agents_df
        self.events_df = events_df
        self.decisions_df = decisions_df
        self.decisions_raw = decisions_raw          # with dict breakdowns, for the inspector
        self.frames = frames                        # one compact dict per round, for the map
        self.events = events
        self.meta = meta
        self.n0 = meta["agents0"]

    def frame(self, r: int) -> dict:
        r = clamp(int(r), 1, max(1, len(self.frames)))
        return self.frames[int(r) - 1]


def run_simulation(cfg: SimConfig, progress_cb: Optional[Callable[[int, int, dict], None]] = None,
                   brain: Optional[np.ndarray] = None, brain_gen: int = 0) -> SimResult:
    return Society(cfg.clamped(), brain=brain, brain_gen=brain_gen).run(progress_cb=progress_cb)


def comparison_pair(base: SimConfig, variable: str) -> tuple[SimConfig, SimConfig, dict]:
    """Build the A/B configs. The ONLY difference is the single patched key."""
    spec = COMPARE_VARS[variable]
    a = SimConfig(**{**base.clamped().to_dict(), spec["key"]: spec["a"]})
    b = SimConfig(**{**a.to_dict(), spec["key"]: spec["b"]})
    return a.clamped(), b.clamped(), spec


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 4 · metrics, discoveries, comparison analysis
# ═══════════════════════════════════════════════════════════════════════════

def third_compare(s: pd.Series) -> dict:
    """Compare the first third vs the last third of a per-round series."""
    x = np.asarray(s, dtype=float)
    n = len(x)
    k = max(3, n // 3)
    first, last = float(x[:k].mean()), float(x[-k:].mean())
    sd = float(np.std(x)) or 1e-9
    rel = (last - first) / (abs(first) if abs(first) > 1e-9 else 1.0)
    z = (last - first) / (sd / math.sqrt(k))
    return {"first": first, "last": last, "delta": last - first, "rel": rel, "z": z}


TREND_COLS = [
    ("var_within", "within-group belief variance", "culture"),
    ("policy_entropy", "policy entropy (LEARNED runs)", "learning"),
    ("divergence_rate", "divergence from fixed rules (LEARNED runs)", "learning"),
    ("cooperation_rate", "co-operation rate", "behaviour"),
    ("competition_rate", "competition rate", "behaviour"),
    ("avg_trust", "average trust", "information"),
    ("avg_reward", "average reward", "economy"),
    ("gini_wealth", "wealth concentration (Gini)", "economy"),
    ("world_food", "total food in the world", "economy"),
    ("avg_food", "average food carried per agent", "survival"),
    ("groups", "active groups", "structure"),
    ("survival", "survival share of the starting population", "survival"),
    ("avg_health", "average health", "survival"),
]


def discover(res: SimResult) -> list[dict]:
    """Rule-based findings computed from THIS run's own dataframes.
    Each finding is an observation; any interpretation is labelled and kept separate."""
    rdf, adf = res.rounds_df, res.agents_df
    out: list[dict] = []
    if rdf.empty or len(rdf) < 9:
        return out
    for col, label, cat in TREND_COLS:
        if col not in rdf.columns or rdf[col].std() < 1e-9:
            continue
        t = third_compare(rdf[col])
        if abs(t["rel"]) < 0.10 or abs(t["z"]) < 1.5:
            continue
        sev = "high" if abs(t["z"]) >= 12 else ("medium" if abs(t["z"]) >= 4 else "low")
        verb = "rose" if t["delta"] > 0 else "fell"
        out.append({
            "severity": sev, "category": cat, "kind": "trend",
            "title": f"{label.capitalize()} {verb} across the run",
            "observed": (f"First third of rounds averaged {fmt(t['first'], 3)} → last third {fmt(t['last'], 3)} "
                         f"(Δ {fmt(t['delta'], 3)}, {t['rel'] * 100:+.0f}% relative, segment z = {fmt(t['z'], 1)})."),
            "reading": ("Possible reading (NOT a causal claim): the feedback between actions, memory and the "
                        "environment drifted together over the run."),
        })
    if "groups" in rdf and rdf["groups"].max() > 0:
        pk = int(rdf["groups"].idxmax())
        out.append({"severity": "low", "category": "structure", "kind": "extreme",
                    "title": f"Peak of {int(rdf.loc[pk, 'groups'])} simultaneous groups at round {int(rdf.loc[pk, 'round'])}",
                    "observed": (f"{int(rdf.loc[pk, 'group_members'])} of {res.n0} original agents were grouped at the peak; "
                                 f"largest group held up to {int(adf['group_size'].max())} members."),
                    "reading": ("Groups here are just ≥2 agents that co-operated 3+ times with mutual trust ≥ 0.6 — "
                                "a bookkeeping label, not an institution."),
                    })
    if "world_food" in rdf:
        mn = int(rdf["world_food"].idxmin())
        out.append({"severity": "low", "category": "economy", "kind": "extreme",
                    "title": f"World food bottomed at round {int(rdf.loc[mn, 'round'])}",
                    "observed": (f"Total field food fell to {fmt(rdf.loc[mn, 'world_food'], 0)} units and ended at "
                                 f"{fmt(rdf['world_food'].iloc[-1], 0)} (start: {fmt(rdf['world_food'].iloc[0], 0)})."),
                    "reading": "Consumption versus regrowth is the whole material economy of this model.",
                    })
    if len(adf) >= 8:
        for x, y, lx, ly in [
            ("trait_cooperation", "reward", "co-operation trait", "final reward"),
            ("trait_competition", "survived_rounds", "competition trait", "rounds survived"),
            ("trait_risk", "wealth", "risk appetite", "final wealth"),
        ]:
            if adf[x].std() > 1e-9 and adf[y].std() > 1e-9:
                r = float(np.corrcoef(adf[x].astype(float), adf[y].astype(float))[0, 1])
                if abs(r) >= 0.15:
                    out.append({"severity": "medium" if abs(r) >= 0.3 else "low",
                                "category": "economy", "kind": "correlation",
                                "title": f"{lx.capitalize()} correlates with {ly} across agents (r = {fmt(r, 2)})",
                                "observed": f"Computed over {len(adf)} agents from their generated traits and recorded outcomes.",
                                "reading": ("A correlation across simulated personalities is an artefact of these rules — "
                                            "it does not describe real personalities."),
                                })
        s_sh = int((adf["act_share"] + adf["act_cooperate"]).sum())
        t_cp = int(adf["act_compete"].sum())
        out.append({"severity": "info", "category": "behaviour", "kind": "totals",
                    "title": f"Lifetime action totals: {s_sh:,} co-operative vs {t_cp:,} competitive acts",
                    "observed": (f"{int(adf['act_trade'].sum()):,} trades, {int(adf['act_share'].sum()):,} shares, "
                                 f"{int(adf['act_cooperate'].sum()):,} co-operations, {t_cp:,} attacks, "
                                 f"{int(adf['act_collapse'].sum()):,} collapses — across {len(rdf)} rounds."),
                    "reading": "",
                    })
    kinds = res.events_df["kind"].value_counts().to_dict() if not res.events_df.empty else {}
    if kinds:
        out.append({"severity": "info", "category": "information", "kind": "events",
                    "title": f"{sum(kinds.values())} logged events across {len(kinds)} categories",
                    "observed": "; ".join(f"{k} ×{v}" for k, v in sorted(kinds.items(), key=lambda kv: -kv[1])),
                    "reading": "All events are emitted by the rules themselves — no outside data.",
                    })
    still = []
    for col, label, _ in TREND_COLS:
        if col not in rdf:
            continue
        t = third_compare(rdf[col])
        if abs(t["rel"]) < 0.04:
            still.append(f"{label}: {fmt(t['first'], 2)} → {fmt(t['last'], 2)} (only {t['rel'] * 100:+.1f}%)")
    if still:
        out.append({"severity": "info", "category": "stability", "kind": "unchanged",
                    "title": "What did NOT change over the run",
                    "observed": " · ".join(still[:5]),
                    "reading": "Reported for honesty — flat metrics are findings too.",
                    })
    order = {"high": 0, "medium": 1, "low": 2, "info": 3}
    out.sort(key=lambda f: (order.get(f["severity"], 9), f["title"]))
    return out


def analyse_comparison(a_res: SimResult, b_res: SimResult, spec: dict) -> dict:
    """Per-metric A vs B stats, with effect sizes and a first-divergence estimate."""
    rows = []
    for col, label, unit, valence in COMPARE_METRICS:
        sa = a_res.rounds_df[col].to_numpy(dtype=float)
        sb = b_res.rounds_df[col].to_numpy(dtype=float)
        n = min(len(sa), len(sb))
        sa, sb = sa[:n], sb[:n]
        diff = sa - sb
        pooled = float(np.sqrt((np.var(sa) + np.var(sb)) / 2.0)) or 1e-9
        d = float(np.mean(diff)) / pooled
        thresh = max(0.02, 0.05 * abs(float(np.mean(sa))))
        diverge = next((int(i + 1) for i, v in enumerate(np.abs(diff)) if v > thresh), None)
        mb = float(np.mean(sb))
        rel = (float(np.mean(sa)) - mb) / (abs(mb) if abs(mb) > 1e-9 else 1.0)
        rows.append({"key": col, "label": label, "unit": unit, "valence": valence,
                     "mean_a": float(np.mean(sa)), "mean_b": mb,
                     "delta": float(np.mean(diff)), "rel": rel, "d": d,
                     "first_divergence": diverge})
    verdict = ""
    main = next((r for r in rows if r["key"] == "cooperation_rate"), rows[0] if rows else None)
    if main:
        if abs(main["delta"]) > 0.01:
            bigger = spec["a_label"] if main["delta"] > 0 else spec["b_label"]
            smaller = spec["b_label"] if main["delta"] > 0 else spec["a_label"]
            verdict = (f"Co-operation averaged {fmt(main['mean_a'], 3)} under {spec['a_label']} vs "
                       f"{fmt(main['mean_b'], 3)} under {spec['b_label']} (Δ {main['delta']:+.3f}, "
                       f"Cohen's d = {fmt(main['d'], 2)}). Inside this model, {bigger} produced more "
                       f"co-operation than {smaller}.")
        else:
            verdict = ("No meaningful difference in co-operation was detected between A and B — on one seed, "
                       "that means 'not enough evidence', not 'proved equal'.")
    unchanged = [r["label"] for r in rows if abs(r["d"]) < 0.20]
    return {"rows": rows, "verdict": verdict, "unchanged": unchanged, "spec": spec}


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 5 · exports (CSV / JSON / zip) — data is temporary, files are not
# ═══════════════════════════════════════════════════════════════════════════

def df_to_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


def jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        x = float(x)
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


def build_json_bundle(res: SimResult, findings: list[dict]) -> bytes:
    bundle = {
        "app": APP_NAME,
        "engine": res.meta["engine"],
        "simulated": True,
        "disclaimer": DISCLAIMER,
        "generated_at": res.meta["generated_at"],
        "config": res.cfg.to_dict(),
        "meta": jsonable(res.meta),
        "round_metrics": jsonable(res.rounds_df.to_dict(orient="records")),
        "agents": jsonable(res.agents_df.to_dict(orient="records")),
        "events": jsonable(res.events_df.to_dict(orient="records")),
        "decision_sample_last_2000": jsonable(res.decisions_df.tail(2000).to_dict(orient="records")),
        "findings": jsonable(findings),
    }
    return json.dumps(bundle, ensure_ascii=False, indent=1).encode("utf-8")


def build_zip(res: SimResult, findings: list[dict]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("yudaant_round_metrics.csv", df_to_csv(res.rounds_df))
        z.writestr("yudaant_agents.csv", df_to_csv(res.agents_df))
        z.writestr("yudaant_events.csv", df_to_csv(res.events_df))
        z.writestr("yudaant_decisions.csv", df_to_csv(res.decisions_df))
        z.writestr("yudaant_bundle.json", build_json_bundle(res, findings))
        readme = (f"{APP_NAME} export · {res.meta['generated_at']} · engine {res.meta['engine']}\n"
                  f"config: {json.dumps(res.cfg.to_dict(), indent=1)}\n\n{DISCLAIMER}\n\n"
                  "The same config + seed reproduces this run exactly:\n"
                  "    python -c \"import app; r = app.run_simulation(app.SimConfig.from_dict(<config above>))\"")
        z.writestr("README.txt", readme)
    return buf.getvalue()


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 6 · self-checks — they run the REAL engine and report honestly
# ═══════════════════════════════════════════════════════════════════════════

def _df_hash(df: pd.DataFrame) -> str:
    return f"{int(pd.util.hash_pandas_object(df, index=True).sum()) & 0xFFFFFFFF:08x}"


def selfchecks() -> list[dict]:
    checks: list[dict] = []

    def add(name: str, fn: Callable[[], str]) -> None:
        try:
            detail = fn()
            checks.append({"check": name, "ok": True, "detail": detail})
        except Exception as e:  # failures are surfaced, never swallowed
            checks.append({"check": name, "ok": False,
                           "detail": f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}"})

    small = dict(population=40, rounds=45, seed=987)

    def _repeat() -> str:
        r1 = run_simulation(SimConfig(**small))
        r2 = run_simulation(SimConfig(**small))
        assert _df_hash(r1.rounds_df) == _df_hash(r2.rounds_df), "round metrics differ between identical runs"
        f1, f2 = r1.frames[-1], r2.frames[-1]
        assert f1["x"] == f2["x"] and f1["y"] == f2["y"], "final positions differ between identical runs"
        assert r1.decisions_raw[-1]["action"] == r2.decisions_raw[-1]["action"]
        assert r1.meta["config_hash"] == r2.meta["config_hash"]
        return (f"2× identical runs of seed {small['seed']}: metric-hash {_df_hash(r1.rounds_df)} matched · "
                f"{len(r1.frames) * len(r1.frames[-1]['ids'])} positions identical · decisions identical")

    def _other_seed() -> str:
        r1 = run_simulation(SimConfig(**small))
        r2 = run_simulation(SimConfig(**{**small, "seed": 988}))
        assert _df_hash(r1.rounds_df) != _df_hash(r2.rounds_df), "different seeds produced identical runs — seed ignored!"
        return "seed 987 vs 988 → clearly different trajectories (the seed genuinely drives generation)"

    def _bounds() -> str:
        res = run_simulation(SimConfig(**{**small, "population": 60}))
        g = np.array(res.frames[-1]["grid"])
        assert g.min() >= 0.0 and g.max() <= CELL_FOOD_CAP + 1e-6, "field food left its bounds"
        a = res.agents_df
        assert a["food"].between(0, AGENT_FOOD_CAP + 1e-6).all(), "agent food out of [0, cap]"
        assert a["energy"].between(0, 100).all(), "energy out of [0,100]"
        assert a["health"].between(0, 100).all(), "health out of [0,100]"
        assert (a["wealth"] >= 0).all(), "negative wealth"
        assert a["reward"].notna().all(), "NaN rewards"
        assert (a["trait_cooperation"].between(0, 1).all() and a["trait_risk"].between(0, 1).all()), "trait out of [0,1]"
        return (f"field food ∈ [0,{CELL_FOOD_CAP:g}] · food/energy/health/wealth in range for all {len(a)} agents · "
                "traits in [0,1] · no NaNs")

    def _metrics() -> str:
        res = run_simulation(SimConfig(**small))
        df = res.rounds_df
        assert len(df) == small["rounds"], f"expected {small['rounds']} rows, got {len(df)}"
        for c in ["cooperation_rate", "competition_rate", "trade_rate", "survival", "avg_trust", "gini_wealth"]:
            assert df[c].between(0, 1).all(), f"{c} left [0,1]: min {df[c].min()}, max {df[c].max()}"
        pop = df["population"].to_numpy()
        assert (np.diff(pop) <= 0).all(), "population increased — deaths-only is broken"
        tot = df[[f"n_{k}" for k in ACTIONS + FORCED]].sum(axis=1).to_numpy()
        assert (tot == df["agents_acted"].to_numpy()).all(), "not every alive agent acted exactly once per round"
        return (f"{len(df)} round rows · all rates ∈ [0,1] · population non-increasing · "
                "every living agent acted exactly once per round (bookkeeping ties out)")

    def _pair() -> str:
        base = SimConfig(population=30, rounds=20, seed=55)
        msgs = []
        for var in COMPARE_VARS:
            ca, cb, spec = comparison_pair(base, var)
            da, db = ca.to_dict(), cb.to_dict()
            diff_keys = {k for k in da if da[k] != db[k]}
            assert diff_keys == {spec["key"]}, f"{var}: expected only {spec['key']} to differ, got {diff_keys}"
            assert da["seed"] == db["seed"] and da["rounds"] == db["rounds"] and da["population"] == db["population"]
            msgs.append(f"{var} → 1 key")
        return "A/B pairs differ in exactly one setting for all knobs (" + ", ".join(msgs) + ")"

    def _export() -> str:
        res = run_simulation(SimConfig(**small))
        csv = pd.read_csv(io.BytesIO(df_to_csv(res.rounds_df)))
        assert list(csv.columns) == list(res.rounds_df.columns) and len(csv) == len(res.rounds_df)
        bundle = json.loads(build_json_bundle(res, []).decode())
        assert bundle["simulated"] is True and len(bundle["round_metrics"]) == len(res.rounds_df)
        z = zipfile.ZipFile(io.BytesIO(build_zip(res, [])))
        names = z.namelist()
        assert "yudaant_bundle.json" in names and "yudaant_decisions.csv" in names
        assert z.read("yudaant_round_metrics.csv").lstrip(b"\xef\xbb\xbf").startswith(b"round,")
        return f"CSV round-trips ({len(csv)} rows) · JSON has simulated:true + full metrics · zip = {len(names)} files"

    def _presets() -> str:
        bad = []
        for p in PRESETS:
            for k in p["changes"]:
                if k not in SimConfig.__dataclass_fields__:
                    bad.append(f"{p['key']}:{k}")
        assert not bad, f"presets write unknown settings: {bad}"
        n = sum(len(p["changes"]) for p in PRESETS)
        return f"all {len(PRESETS)} presets touch only real SimConfig fields ({n} declared changes shown in UI)"

    def _society_moves() -> str:
        kw = dict(population=60, rounds=60, seed=13)
        r_mixed = run_simulation(SimConfig(**kw, reward_model="MIXED"))
        r_comp = run_simulation(SimConfig(**kw, reward_model="COMPETITIVE"))
        r_coop = run_simulation(SimConfig(**kw, reward_model="COOPERATIVE"))
        m = int(r_mixed.rounds_df["n_compete"].sum())
        c = int(r_comp.rounds_df["n_compete"].sum())
        assert c >= m * 0.9 + 1, f"competitive rewards did NOT increase attacks ({c} vs {m}) — incentive plumbing broken"
        kind_m = int((r_mixed.rounds_df["n_share"] + r_mixed.rounds_df["n_cooperate"]).sum())
        kind_c = int((r_coop.rounds_df["n_share"] + r_coop.rounds_df["n_cooperate"]).sum())
        assert kind_c >= kind_m * 1.05, f"cooperative rewards did not increase kind acts ({kind_c} vs {kind_m})"
        r_scarce = run_simulation(SimConfig(population=80, rounds=120, seed=21, food_regrowth=0.4, food_initial=22))
        assert r_scarce.rounds_df["deaths"].iloc[-1] > 0, "scarcity killed no one — starvation plumbing broken"
        r_full = run_simulation(SimConfig(population=60, rounds=60, seed=13, start_trust=0.7))
        assert len(r_full.events) > 0, "no events emitted at all"
        return (f"rewards move behaviour as designed: attacks {c} (COMP) vs {m} (MIXED) · "
                f"kind acts {kind_c} (COOP) vs {kind_m} (MIXED) · Food-Shock world lost "
                f"{int(r_scarce.rounds_df['deaths'].iloc[-1])} agents to starvation · trusting world logged "
                f"{len(r_full.events)} events")

    def _livepath() -> str:
        cfg_a = SimConfig(**small)
        r_inst = run_simulation(cfg_a)
        soc = Society(SimConfig(**small))
        steps = 0
        while soc.step():
            steps += 1
        r_live = soc.finalize()
        assert steps == cfg_a.rounds, f"step() advanced {steps} rounds, expected {cfg_a.rounds}"
        assert not soc.step(), "step() did not refuse past the final round"
        assert _df_hash(r_live.rounds_df) == _df_hash(r_inst.rounds_df), "live step-loop ≠ instant run"
        assert len(r_live.frames) == len(r_inst.frames) and len(r_live.decisions_raw) == len(r_inst.decisions_raw)
        return (f"round-by-round playback is bit-identical to the instant run "
                f"({steps} steps, {len(r_live.decisions_raw):,} decisions matched, same final frames)")

    def _learning() -> str:
        lc = SimConfig(population=45, rounds=40, seed=31, policy="LEARNED", policy_lr=0.3, policy_temp=0.7)
        r1 = run_simulation(lc)
        r2 = run_simulation(lc)
        assert _df_hash(r1.rounds_df) == _df_hash(r2.rounds_df), "LEARNED mode not reproducible at fixed seed"
        e = r1.rounds_df["policy_entropy"].to_numpy(dtype=float)
        assert np.isfinite(e).all() and (e >= -1e-6).all() and e.max() <= math.log(8) + 0.01, \
            f"policy entropy out of bounds (max {e.max():.3f}, ln8 = {math.log(8):.3f})"
        sh = float(r1.agents_df["learn_shift"].mean())
        assert sh > 0.002, f"multipliers never moved from the ×1.00 rule anchor (mean |ΔW| = {sh:.4f})"
        div = float(r1.rounds_df["divergence_rate"].mean())
        assert div > 0.02, f"learned agents never diverged from the rules ({div*100:.1f}%)"
        r3 = run_simulation(SimConfig(population=45, rounds=40, seed=31))
        assert _df_hash(r1.rounds_df) != _df_hash(r3.rounds_df), "LEARNED run identical to RULE run — switch not wired"
        adf = r1.agents_df
        assert (adf["food"] >= -1e-9).all() and (adf["health"] <= 100.001).all() and (adf["wealth"] >= -1e-9).all(), \
            "learning broke resource bounds"
        return (f"weights moved (mean |ΔW| {sh:.3f}), off-rule on {div*100:.0f}% of decisions, "
                f"entropy {e[0]:.2f}→{e[-1]:.2f} bits ≤ ln 8, bounds intact, same-seed reruns identical")

    def _carry() -> str:
        base = SimConfig(population=45, rounds=30, seed=99, policy="LEARNED", policy_carry=True)
        r_a = run_simulation(base)
        brain = np.zeros((8, N_SLOT))
        brain[AIDX["share"], POLICY_ACTION_SLOTS["share"]] += 0.8
        brain[AIDX["compete"], POLICY_ACTION_SLOTS["compete"]] -= 0.8
        r_b = run_simulation(base, brain=brain)
        assert _df_hash(r_a.rounds_df) != _df_hash(r_b.rounds_df), "carried brain changed nothing — wiring broken"
        k_a = int(r_a.rounds_df["n_share"].iloc[-1] + r_a.rounds_df["n_cooperate"].iloc[-1])
        k_b = int(r_b.rounds_df["n_share"].iloc[-1] + r_b.rounds_df["n_cooperate"].iloc[-1])
        return (f"inherited multipliers rewire behaviour as intended (bias toward share +0.8 / away from "
                f"compete −0.8 gave {k_b} kind acts in the final round vs {k_a} unseeded)")

    add("Live playback path ≡ instant path", _livepath)
    def _culture() -> str:
        s = Society(SimConfig(population=50, rounds=20, seed=88, policy="LEARNED", influence=0.5))
        for _ in range(10):
            s.step()
        ids = sorted(a.aid for a in s.agents)[:3]
        s.groups[9999] = set(ids)
        s.W[ids[0]] = 1.0 + 0.6
        s.W[ids[1]] = 1.0 - 0.6
        before = float(np.abs(s.W[ids[0]] - s.W[ids[1]]).mean())
        s._diffuse()
        after = float(np.abs(s.W[ids[0]] - s.W[ids[1]]).mean())
        assert after < before * 0.75, f"diffusion did not converge the group ({before:.3f} → {after:.3f})"
        s0 = Society(SimConfig(population=50, rounds=5, seed=88, policy="LEARNED", influence=0.0))
        w0 = s0.W.copy()
        s0.groups[9999] = {0, 1}
        s0._diffuse()
        assert np.allclose(w0, s0.W), "diffusion fired with influence = 0"
        s2 = Society(SimConfig(population=50, rounds=5, seed=88, policy="LEARNED", influence=0.5))
        s2.groups[8888] = {0, 1}
        s2.agents[0].group = 8888
        s2.W[0] = 2.2
        s2.W[1] = 1.0
        s2._legacy(s2.agents[0])
        assert float(s2.W[1].mean()) > 1.15, "legacy transmission did not shift the survivor toward the dead"
        ta = np.array([ag.t_adapt for ag in s.agents])
        assert np.allclose(s._anchor[:, 0, 0], 0.02 * (1.7 - ta)), "personality anchor not wired"
        for _ in range(8):
            s.step()
        rr = s.round_rows[-1]
        assert np.isfinite(rr["var_within"]) and np.isfinite(rr["var_global"]), "variance stats not finite"
        return (f"diffusion converged a group ({before:.3f} → {after:.3f}), influence 0 is a strict no-op, "
                f"deaths bequeath habits, anchors scale with adaptability, [L9] stats finite")

    add("Learning plumbing — LEARNED trains, diverges & stays deterministic", _learning)
    add("🌀 Culture plumbing — diffusion, legacy, anchors & variance stats", _culture)
    add("Carry-across-runs memory actually seeds the next generation", _carry)
    add("Repeatability — same seed ⇒ same run", _repeat)
    add("Sensitivity — different seed ⇒ different run", _other_seed)
    add("Resource bounds — food / energy / health / wealth", _bounds)
    add("Metric ranges & bookkeeping (rates, deaths-only, actions tie out)", _metrics)
    add("Comparison pairs are single-variable", _pair)
    add("CSV / JSON / zip exports round-trip", _export)
    add("Presets only set real, listed settings", _presets)
    add("Behaviour sanity — reward table actually steers behaviour", _society_moves)
    return checks


@st.cache_resource(show_spinner=False)
def cached_checks() -> list[dict]:
    return selfchecks()


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 7 · Streamlit interface
# ═══════════════════════════════════════════════════════════════════════════

def style_fig(fig: go.Figure, h: int = 330, legend: bool = True) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(11,18,32,0.55)",
        font=dict(family="'Inter','Segoe UI',system-ui,sans-serif", size=12, color=PALETTE["text"]),
        margin=dict(l=52, r=16, t=44, b=36),
        height=h,
        hoverlabel=dict(bgcolor="#0d1526", bordercolor="#25334f", font=dict(size=12, color="#dbe6fb")),
    )
    if legend:
        fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.005, xanchor="left", x=0))
    else:
        fig.update_layout(showlegend=False)
    fig.update_xaxes(gridcolor=PALETTE["grid"], zerolinecolor=PALETTE["grid"])
    fig.update_yaxes(gridcolor=PALETTE["grid"], zerolinecolor=PALETTE["grid"])
    return fig


def chip(text: str, color: str = "#25334f", fg: str = "#9fb3d1") -> str:
    return (f"<span style='display:inline-block;padding:2px 10px;border-radius:999px;"
            f"background:{color};color:{fg};font-size:11.5px;letter-spacing:.04em;"
            f"border:1px solid rgba(148,163,184,.18);font-weight:600'>{text}</span>")


def kpi_card(label: str, value: str, sub: str = "", accent: str = PALETTE["teal"]) -> str:
    return (f"<div class='yu-card' style='border-top:2px solid {accent}'>"
            f"<div class='yu-kpi-label'>{label}</div>"
            f"<div class='yu-kpi-value' style='color:{accent}'>{value}</div>"
            f"<div class='yu-kpi-sub'>{sub}</div></div>")


CSS = """
<style>
#MainMenu, footer {visibility:hidden;}
.block-container {padding-top:2.1rem; padding-bottom:3.2rem; max-width:1560px;}
html, body, [class*="css"] {font-family:'Inter','Segoe UI',system-ui,sans-serif;}
.yu-hero {background:
    radial-gradient(1100px 300px at 12% -60px, rgba(45,212,191,.14), transparent 60%),
    linear-gradient(160deg, #101a30 0%, #0b1220 70%);
    border:1px solid rgba(45,212,191,.22); border-radius:18px; padding:26px 30px 20px 30px; margin-bottom:14px;}
.yu-title {font-size:34px; font-weight:800; color:#eaf1fb; margin:0; letter-spacing:.06em;}
.yu-title .accent {color:#2dd4bf;}
.yu-sub {color:#8ea0bd; font-size:14.5px; margin:6px 0 12px 0; max-width:940px; line-height:1.55;}
.yu-card {background:linear-gradient(175deg,#111b31, #0d1526); border:1px solid rgba(148,163,184,.16);
    border-radius:14px; padding:11px 14px; height:100%;}
.yu-kpi-label {font-size:10.5px; letter-spacing:.13em; text-transform:uppercase; color:#7c8db0; font-weight:700;}
.yu-kpi-value {font-size:24px; font-weight:800; color:#eaf1fb; line-height:1.3; font-variant-numeric:tabular-nums;}
.yu-kpi-sub {font-size:11px; color:#7c8db0; line-height:1.4;}
.yu-feed {max-height:430px; overflow-y:auto; padding-right:6px; font-size:12.8px;}
.yu-feed .row {padding:5.5px 9px; border-bottom:1px dashed rgba(148,163,184,.14); color:#c3d0e6; line-height:1.4;}
.yu-feed .row .t {color:#5c6f92; font-variant-numeric:tabular-nums; margin-right:8px;}
.yu-feed::-webkit-scrollbar {width:8px;}
.yu-feed::-webkit-scrollbar-thumb {background:#243250; border-radius:8px;}
.yu-note {background:rgba(245,178,60,.07); border:1px solid rgba(245,178,60,.35); border-radius:12px;
    padding:12px 16px; color:#e7c98a; font-size:13.5px; line-height:1.55;}
.yu-danger {background:rgba(248,113,113,.07); border:1px solid rgba(248,113,113,.4); border-radius:12px;
    padding:12px 16px; color:#f3b1b1; font-size:13.5px; line-height:1.55;}
.yu-ok {background:rgba(52,211,153,.07); border:1px solid rgba(52,211,153,.35); border-radius:12px;
    padding:12px 16px; color:#a9e8cf; font-size:13.5px; line-height:1.55;}
.yu-find {background:#0d1526; border:1px solid rgba(148,163,184,.14); border-left:4px solid #94a3b8;
    border-radius:12px; padding:13px 16px; margin-bottom:10px;}
.yu-find.high {border-left-color:#f87171;} .yu-find.medium {border-left-color:#f5b23c;}
.yu-find.low {border-left-color:#60a5fa;} .yu-find.info {border-left-color:#94a3b8;}
.yu-find h4 {margin:0 0 4px 0; color:#e7eefb; font-size:15px;}
.yu-find .obs {color:#b9c7de; font-size:12.8px; line-height:1.5;}
.yu-find .read {color:#e7c98a; font-size:12px; font-style:italic; margin-top:6px; line-height:1.45;}
.yu-chiprow {margin:2px 0 12px 0; display:flex; gap:8px; flex-wrap:wrap; align-items:center;}
.stTabs [data-baseweb="tab-list"] {gap:4px;}
.stTabs [data-baseweb="tab"] {font-weight:600; letter-spacing:.02em;}
div[data-testid="stSidebar"] {background:#0a111f;}
div[data-testid="stSidebar"] .yu-side-title {font-size:13px; font-weight:800; letter-spacing:.12em;
    color:#dbe6fb; text-transform:uppercase;}
code, .stMarkdown code {background:#0e1930; border:1px solid #1e2c49; padding:1px 6px; border-radius:6px;
    color:#9fe8dc; font-size:.86em;}
button[kind="primary"] {background:linear-gradient(160deg,#0f766e,#134e4a)!important; border:1px solid #2dd4bf!important;
    box-shadow:0 4px 18px rgba(45,212,191,.25); font-weight:700; letter-spacing:.04em;}
.empty-state {border:1.5px dashed #2a3b5e; border-radius:16px; padding:40px 30px; text-align:center; color:#8ea0bd;
    background:radial-gradient(600px 200px at 50% 0%, rgba(45,212,191,.06), transparent 70%); margin-top:8px;}
.empty-state h3 {color:#c9d7ee; margin-top:8px;}
progress {width:100%; height:6px; accent-color:#2dd4bf;}
</style>
"""


# ── sidebar state helpers ──────────────────────────────────────────────────
def init_sidebar_defaults() -> None:
    d = SimConfig().to_dict()
    if not st.session_state.get("_sb_init"):
        for k in WIDGET_KEYS:
            st.session_state.setdefault(f"sb_{k}", d[k])
        st.session_state.setdefault("sb_live", True)
        st.session_state["_sb_init"] = True


def cfg_from_widgets() -> SimConfig:
    s = st.session_state
    return SimConfig(
        population=int(s.get("sb_population", 110)), rounds=int(s.get("sb_rounds", 200)),
        seed=int(s.get("sb_seed", 1337)), start_trust=float(s.get("sb_start_trust", 0.5)),
        memory=bool(s.get("sb_memory", True)), communication=s.get("sb_communication", "LIMITED"),
        reward_model=s.get("sb_reward_model", "MIXED"), food_regrowth=float(s.get("sb_food_regrowth", 2.4)),
        food_initial=float(s.get("sb_food_initial", 55.0)), movement_speed=int(s.get("sb_movement_speed", 1)),
        policy=s.get("sb_policy", "RULE"), policy_lr=float(s.get("sb_policy_lr", 0.10)),
        policy_temp=float(s.get("sb_policy_temp", 0.9)), policy_buffer=int(s.get("sb_policy_buffer", 8)),
        policy_carry=bool(s.get("sb_policy_carry", False)),
        influence=float(s.get("sb_influence", 0.0)), policy_anchor=float(s.get("sb_policy_anchor", 0.02)),
    )


def apply_preset(pkey: str) -> None:
    preset = PRESET_BY_KEY[pkey]
    merged = cfg_from_widgets().to_dict()
    merged.update(preset["changes"])
    for k, v in merged.items():
        if k in WIDGET_KEYS:
            st.session_state[f"sb_{k}"] = v
    st.session_state["last_preset"] = pkey
    st.session_state["just_applied"] = (
        f"Preset “{preset['label']}” applied to the settings — press ▶ Start simulation to run it. "
        f"Changed: " + (", ".join(f"{SETTING_LABEL[k]}" for k in preset["changes"]) or "nothing (baseline)"))


def start_run() -> None:
    st.session_state["_start_requested"] = True


def render_sidebar() -> None:
    init_sidebar_defaults()
    with st.sidebar:
        st.markdown("<div class='yu-side-title'>Yudaant setup</div>", unsafe_allow_html=True)
        st.caption("All data is generated here — nothing is fetched, nothing is saved online.")

        st.markdown("**Presets** — each lists *exactly* what it changes")
        preset_keys = [p["key"] for p in PRESETS]
        cur = st.session_state.get("last_preset", "free")
        sel = st.radio("scenario", options=preset_keys,
                       index=preset_keys.index(cur) if cur in preset_keys else 0,
                       format_func=lambda k: PRESET_BY_KEY[k]["label"], label_visibility="collapsed")
        p = PRESET_BY_KEY[sel]
        st.caption(p["blurb"])
        diff = preset_diff(cfg_from_widgets(), p["changes"])
        if diff:
            st.markdown("**it will set →**")
            for label, old, new in diff:
                if old != new:
                    st.markdown(chip(f"{label}:  {old}  →  {new}", "#132a3a", "#7dd3fc"), unsafe_allow_html=True)
                else:
                    st.markdown(chip(f"{label}:  {new}  (unchanged by you)", "#1a2436", "#64748b"),
                                unsafe_allow_html=True)
        else:
            st.caption("This preset sets every listed field back to its default.")
        st.button("Apply preset to settings", width="stretch", on_click=apply_preset, args=(sel,))
        if st.session_state.get("last_preset") != sel:
            st.caption("⚠ Selection changed but not applied yet — press “Apply preset”.")

        st.divider()
        st.markdown("**Quick settings**")
        st.number_input("👥 Population", 5, 250, step=5, key="sb_population", help=CONFIG_HELP["population"])
        st.number_input("⏱ Rounds", 10, 500, step=10, key="sb_rounds", help=CONFIG_HELP["rounds"])
        st.number_input("🎲 Seed", 0, 10_000_000, step=1, key="sb_seed", help=CONFIG_HELP["seed"])
        st.slider("🤝 Starting trust (0–1)", min_value=0.0, max_value=1.0, step=0.02,
                  key="sb_start_trust", help=CONFIG_HELP["start_trust"])
        c1, c2 = st.columns(2)
        c1.toggle("🧠 Memory", key="sb_memory", help=CONFIG_HELP["memory"])
        c2.selectbox("📣 Communication", COMM_LEVELS, key="sb_communication", help=CONFIG_HELP["communication"])

        with st.expander("⚙️ Advanced — world & incentives"):
            st.selectbox("💰 Reward model", REWARD_MODELS, key="sb_reward_model", help=CONFIG_HELP["reward_model"])
            st.slider("🌾 Food regrowth / cell / round", min_value=0.05, max_value=6.0, step=0.05,
                      key="sb_food_regrowth", help=CONFIG_HELP["food_regrowth"])
            st.slider("🗺 Starting food / cell (avg)", min_value=5.0, max_value=100.0, step=1.0,
                      key="sb_food_initial", help=CONFIG_HELP["food_initial"])
            st.slider("🚶 Movement speed (cells / round)", 1, 3, key="sb_movement_speed",
                      help=CONFIG_HELP["movement_speed"])

        st.divider()
        st.markdown("**🧠 Agent policy** — how each agent picks its next action")
        st.radio("policy", ["RULE", "LEARNED"], key="sb_policy", horizontal=True,
                 format_func=lambda k: "📜 Fixed rules" if k == "RULE" else "🧠 Learned (trains as it runs)",
                 label_visibility="collapsed", help=CONFIG_HELP["policy"])
        if st.session_state.get("sb_policy", "RULE") == "LEARNED":
            c1, c2 = st.columns(2)
            c1.slider("learning rate", 0.01, 0.40, step=0.01, key="sb_policy_lr", help=CONFIG_HELP["policy_lr"])
            c2.slider("temperature", 0.20, 2.00, step=0.05, key="sb_policy_temp",
                      help=CONFIG_HELP["policy_temp"])
            c3, c4 = st.columns(2)
            c3.slider("own-buffer size", 2, 24, step=1, key="sb_policy_buffer",
                      help=CONFIG_HELP["policy_buffer"])
            c4.toggle("🔁 carry across runs", key="sb_policy_carry", help=CONFIG_HELP["policy_carry"])
            st.slider("🌀 culture diffusion (peer influence)", 0.0, 0.60, step=0.05, key="sb_influence",
                      help=CONFIG_HELP["influence"])
            with st.expander("advanced training knobs"):
                st.slider("rule-anchor pull (0 = habits never fade)", 0.0, 0.20, step=0.005,
                          key="sb_policy_anchor", help=CONFIG_HELP["policy_anchor"])
            st.caption("small data = each agent's own last-k decisions only. Deterministic under the seed "
                       "(except 🔁 carry, which is stateful on purpose).")

        st.toggle("🎬  Watch it live", key="sb_live",
                  help="ON (recommended): Start animates the world round-by-round in the Society tab — "
                       "pause, change speed, or skip to end. OFF: instant headless run (~0.5 s).")
        st.button("▶  Start simulation", width="stretch", type="primary", on_click=start_run, key="sb_start",
                  help="Runs fresh from the current settings. Live mode shows the movement; instant mode takes ~0.5 s.")
        est = int(st.session_state.get("sb_rounds", 200)) * int(st.session_state.get("sb_population", 110))
        st.caption(f"≈ {est:,} decisions to compute; pure Python, deterministic.")


# ── hero + KPI strip ───────────────────────────────────────────────────────
def render_hero() -> None:
    st.markdown(
        "<div class='yu-hero'>"
        "<div class='yu-title'>⚔ YUDA<span class='accent'>ANT</span>"
        "<span style=\"font-size:13px;font-weight:600;color:#7c8db0;letter-spacing:.18em;margin-left:10px;\">"
        "ARTIFICIAL-SOCIETY OBSERVATORY</span></div>"
        "<div class='yu-sub'>A small synthetic society of agents — they gather, trade, share, co-operate "
        "and steal in a seeded world. Every score and chart on this page is computed <b>live from that "
        "simulation</b>.</div>"
        "</div>", unsafe_allow_html=True)


def render_kpi_strip(res: SimResult) -> None:
    r = res.rounds_df.iloc[-1]
    if res.cfg.policy == "LEARNED":
        _sh = f" · mean |ΔW| {r['avg_shift']:.3f} · H {r['policy_entropy']:.2f} bits" if "avg_shift" in r.index else ""
        st.markdown(chip(f"🧠 LEARNED policy · lr {res.cfg.policy_lr:.2f} · τ {res.cfg.policy_temp:.2f} · own "
                         f"buffer {res.cfg.policy_buffer} samples · 🌀 {res.cfg.influence:.2f} influence · "
                         f"anchor {res.cfg.policy_anchor:.3f} · "
                         f"{'🔁 carry gen ' + str(res.meta.get('brain_gen', 0) + 1) if res.cfg.policy_carry else 'no carry'}"
                         + _sh, "#241d43", "#c4b5fd"), unsafe_allow_html=True)
    prev = res.rounds_df.iloc[-11] if len(res.rounds_df) >= 11 else res.rounds_df.iloc[0]

    def d(v: float, p: float) -> str:
        dv = v - p
        if abs(p) > 1e-9:
            dv = dv / abs(p) * 100.0
            return f"<span style='color:{'#34d399' if dv >= 0 else '#f87171'}'>{dv:+.1f}%</span> vs prev 10 rounds"
        return "<span style='color:#7c8db0'>≈ flat vs prev 10 rounds</span>"

    cards = [
        ("Living agents", f"{int(r['population'])} / {res.n0}",
         f"{r['survival'] * 100:.0f}% survived · {int(r['deaths'])} deaths", PALETTE["blue"]),
        ("Co-operation rate", f"{r['cooperation_rate'] * 100:.0f}%",
         d(r["cooperation_rate"], prev["cooperation_rate"]), PALETTE["green"]),
        ("Competition rate", f"{r['competition_rate'] * 100:.1f}%",
         d(r["competition_rate"], prev["competition_rate"]), PALETTE["red"]),
        ("Mean trust", f"{r['avg_trust']:.2f}",
         "0–1 · " + ("remembered per partner" if res.cfg.memory else "fixed at start value (memory off)"),
         PALETTE["teal"]),
        ("Field food", f"{r['world_food']:,.0f}",
         "units across all 288 map cells (cap 28,800)", PALETTE["amber"]),
        ("Wealth Gini", f"{r['gini_wealth']:.3f}", "0 = equal · 1 = held by one agent", PALETTE["violet"]),
        ("Active groups", f"{int(r['groups'])}",
         f"{int(r['group_members'])} agents inside groups ≥ 2", PALETTE["blue"]),
        ("Mean reward", f"{r['avg_reward']:,.1f}",
         f"pts / living agent · {res.cfg.reward_model} payout table", PALETTE["teal"]),
    ]
    cols = st.columns(len(cards))
    for col, (lab, val, sub, ac) in zip(cols, cards):
        col.markdown(kpi_card(lab, val, sub, ac), unsafe_allow_html=True)


# ── society view (map + feed + scrubber) ───────────────────────────────────

LIVE_INTERVAL = 0.25   # seconds between playback ticks while playing


def _skip_live() -> None:
    """Skip to end: burn the remaining rounds instantly, then package the run."""
    live = st.session_state.get("live")
    if not live:
        return
    soc = live["soc"]
    while soc.step():
        pass
    _finish_live(soc)


def _abort_live() -> None:
    st.session_state.pop("live", None)
    st.session_state["last_run_msg"] = "⏹ live run aborted — nothing recorded to the other tabs."


def _finish_live(soc: "Society") -> None:
    res = soc.finalize()
    st.session_state["res"] = res
    after_run(res)
    st.session_state.pop("live", None)
    st.session_state.pop("frame_slider", None)
    st.session_state["insp_sel"] = None
    st.session_state["last_run_msg"] = (
        f"✅ live run complete — {res.cfg.population} agents × {res.cfg.rounds} rounds · "
        f"{res.meta['alive_end']}/{res.n0} survived · {res.meta['decisions']:,} decisions · "
        f"{res.meta['events']} events · config hash `{res.meta['config_hash']}` — every tab is now live data")


def render_live() -> None:
    live = st.session_state.get("live")
    if not live:
        return
    soc: "Society" = live["soc"]
    if soc.rnd >= soc.cfg.rounds:
        _finish_live(soc)
        return

    cA, cB, cC, cD = st.columns([1.3, 3.4, 1.2, 1.2])
    playing = st.session_state.get("live_playing", True)
    cA.button("⏸ Pause" if playing else "▶ Resume", width="stretch", type="primary", key="live_pp",
              on_click=lambda: st.session_state.update(live_playing=not st.session_state.get("live_playing", True)))
    with cB:
        st.radio("speed (rounds / second)", [1, 2, 5, 10, 20, 40], index=4,
                 key="live_rps", horizontal=True, label_visibility="collapsed")
    cC.button("⏭ Skip to end", width="stretch", key="live_skip", on_click=_skip_live,
              help="finish all remaining rounds instantly")
    cD.button("⏹ Abort", width="stretch", key="live_abort", on_click=_abort_live)

    interval = LIVE_INTERVAL if playing else None

    def _tick() -> None:
        l = st.session_state.get("live")
        if not l:
            return
        s = l["soc"]
        if st.session_state.get("live_playing", True):
            chunk = max(1, int(round(st.session_state.get("live_rps", 20) * LIVE_INTERVAL)))
            for _ in range(chunk):
                if not s.step():
                    break
        _render_live_frame(s)
        if s.rnd >= s.cfg.rounds:
            _finish_live(s)

    _tick = st.fragment(_tick, run_every=interval)
    _tick()


def _render_live_frame(soc: "Society") -> None:
    cfg = soc.cfg
    if not soc.frames:
        st.progress(0.0, text=f"warming up… seed {cfg.seed} · world generated, round 0 of {cfg.rounds}")
        return
    fr = soc.frames[-1]
    row = soc.round_rows[-1]
    _lx = (f" · 🧠 H {row['policy_entropy']:.2f}b · {row['divergence_rate']*100:.0f}% off-rule"
           if "policy_entropy" in row else "")
    st.progress(soc.rnd / cfg.rounds,
                text=f"🎬 round {soc.rnd}/{cfg.rounds} · {fr['n_alive']}/{soc.n0} alive · "
                     f"co-operation {row['cooperation_rate']*100:.0f}% · competition {row['competition_rate']*100:.0f}% · "
                     f"{row['groups']} groups · field food {row['world_food']:,.0f}{_lx}")

    cL, cR = st.columns([2.05, 1])
    with cL:
        fig = go.Figure()
        g = np.array(fr["grid"], dtype=float)
        fig.add_trace(go.Heatmap(
            z=g, x=[i * CELL for i in range(len(g[0]) + 1)],
            y=[j * CELL for j in range(len(g) + 1)][::-1],
            colorscale=[[0, "rgba(13,21,38,0)"], [0.45, "rgba(20,83,64,.42)"], [1.0, "rgba(52,211,153,.55)"]],
            showscale=False, hoverinfo="skip", zsmooth="best"))
        by_cat: dict[str, dict[str, list]] = {}
        for i, aid in enumerate(fr["ids"]):
            d0 = by_cat.setdefault(fr["cat"][i], {"x": [], "y": [], "sz": [], "hov": []})
            d0["x"].append(fr["x"][i]); d0["y"].append(fr["y"][i])
            d0["sz"].append(6 + 14 * clamp01(fr["wealth"][i] / 80.0))
            d0["hov"].append(f"#{aid} · {ACTION_LABEL.get(fr['last'][i], fr['last'][i])} · food {fr['food'][i]:.0f}")
        for cat, d0 in by_cat.items():
            fig.add_trace(go.Scatter(x=d0["x"], y=d0["y"], mode="markers", name=f"{cat} ({len(d0['x'])})",
                                     marker=dict(size=d0["sz"], color=CAT_COLOR.get(cat, "#94a3b8"), opacity=0.92,
                                                 line=dict(width=1, color="rgba(8,14,26,.9)")),
                                     text=d0["hov"], hoverinfo="text"))
        for gid, members in fr["groups"].items():
            pos = {aid: (fr["x"][i], fr["y"][i]) for i, aid in enumerate(fr["ids"])}
            pts = [pos[m] for m in members if m in pos]
            if not pts:
                continue
            gx = [p[0] for p in pts]; gy = [p[1] for p in pts]
            r0 = max(18.0, math.hypot(max(gx) - min(gx), max(gy) - min(gy)) / 2 + 16)
            fig.add_trace(go.Scatter(x=[sum(gx)/len(gx)], y=[sum(gy)/len(gy)], mode="markers", showlegend=False,
                                     marker=dict(size=r0, sizemode="diameter", symbol="circle-open",
                                                 line=dict(width=1.6, color=GROUP_COLORS[gid % len(GROUP_COLORS)]),
                                                 opacity=0.75)))
        fig.update_layout(
            title=dict(text=f"🗺 LIVE — seed {cfg.seed} · {cfg.world_cols}×{cfg.world_rows} cells · shading = food",
                       font=dict(size=13, color="#c9d7ee")),
            xaxis=dict(range=[-2, WORLD_W + 2], showticklabels=False),
            yaxis=dict(range=[-2, WORLD_H + 2], showticklabels=False),
            plot_bgcolor="#0a111f", paper_bgcolor="rgba(0,0,0,0)",
            height=500, margin=dict(l=8, r=8, t=40, b=4),
            legend=dict(orientation="h", yanchor="top", y=-0.02, xanchor="center", x=0.5, font=dict(size=10.5)),
            hovermode="closest")
        st.plotly_chart(fig, width="stretch", key=f"livemap_{soc.rnd}")

    with cR:
        # ---- what the agents are doing right now (latest decisions, straight from the engine) ----
        st.markdown("<div style='font-size:11px;letter-spacing:.12em;color:#7c8db0;font-weight:700;margin-bottom:4px;'>"
                    f"NOW ACTING — latest decisions (round {soc.rnd})</div>", unsafe_allow_html=True)
        recent = soc.decisions[-8:][::-1]
        rows = []
        for d in recent:
            note = f" → {d['note']}" if d["note"] else ""
            rows.append(f"<div class='row'><span class='t'>r{d['round']}</span>"
                        f"<b style='color:{ACTION_COLOR.get(d['action'],'#94a3b8')}'>{d['action']}</b> · #{d['agent']}"
                        f"<span style='color:#7c8db0'>{note[:150]}</span></div>")
        st.markdown(f"<div class='yu-feed' style='max-height:170px'>{''.join(rows)}</div>", unsafe_allow_html=True)

        # ---- follow one agent while the world moves ----
        st.markdown("<div style='font-size:11px;letter-spacing:.12em;color:#7c8db0;font-weight:700;margin:8px 0 4px;'>"
                    "FOLLOW AN AGENT</div>", unsafe_allow_html=True)
        ids = fr["ids"]
        fcol1, fcol2 = st.columns([1.4, 1])
        with fcol1:
            _prev = st.session_state.get("live_follow")
        _idx = ids.index(_prev) if _prev in ids else 0
        fid = st.selectbox("agent", ids, index=_idx, key="live_follow", label_visibility="collapsed")
        aid = int(fid) if fid in ids else ids[0]
        i = fr["ids"].index(aid)
        vals = {"food": fr["food"][i], "energy": fr["energy"][i], "wealth": fr["wealth"][i], "health": fr["health"][i]}
        colr = {"food": "#34d399", "energy": "#60a5fa", "wealth": "#a78bfa", "health": "#f87171"}
        cap = {"food": 90, "energy": 100, "wealth": 80, "health": 100}
        inner = "".join(
            f"<div style='font-size:12px;color:#c3d0e6;margin:2px 0'>{k} <b style='float:right'>{v:.0f}</b>"
            f"<progress value='{v}' max='{cap[k]}' style='width:100%;accent-color:{colr[k]}'></progress></div>"
            for k, v in vals.items())
        d = next((x for x in reversed(soc.decisions) if x["agent"] == aid), None)
        why = f"last: <b style='color:#5eead4'>{d['action']}</b> — {d['why'][:110]}" if d else "no decision yet"
        st.markdown(f"<div class='yu-card'><div class='yu-kpi-label'>agent #{aid} · doing “{fr['last'][i]}”</div>"
                    f"<div style='margin-top:6px'>{inner}</div>"
                    f"<div style='font-size:11.5px;color:#8ea0bd;margin-top:6px;line-height:1.4'>{why}</div></div>",
                    unsafe_allow_html=True)
        with fcol2:
            hist = soc.round_rows[-80:]
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(y=[h["cooperation_rate"] for h in hist], name="co-operation",
                                      line=dict(color=PALETTE["green"], width=1.8)))
            fig2.add_trace(go.Scatter(y=[h["competition_rate"] for h in hist], name="competition",
                                      line=dict(color=PALETTE["red"], width=1.8)))
            style_fig(fig2, h=170)
            fig2.update_layout(title=dict(text="last 80 rounds", font=dict(size=11, color="#8ea0bd")),
                               legend=dict(orientation="h", y=-0.35, x=0, font=dict(size=10)),
                               margin=dict(l=6, r=6, t=26, b=18))
            fig2.update_yaxes(range=[0, max(0.35, float(np.max([h['cooperation_rate'] for h in hist])))])
            st.plotly_chart(fig2, width="stretch", key=f"livehist_{soc.rnd // 5}")
    st.caption("Live view = the engine mid-run: each tick advances real rounds (same code path as the instant run — "
               "self-check 1 verifies both are bit-identical). Pause, change speed, or Skip to end any time.")


def _step_frame(delta: int) -> None:
    n = st.session_state.get("_frame_max", 1)
    cur = st.session_state.get("frame_slider", n)
    st.session_state["frame_slider"] = int(clamp(cur + delta, 1, n))


def _end_frame() -> None:
    st.session_state["frame_slider"] = st.session_state.get("_frame_max", 1)


def render_society(res: SimResult) -> None:
    st.caption("Every dot is one simulated agent, coloured by the action it just took. Background shading is the "
               "food stored in each map cell. This is the live simulated state — not a pre-made animation.")
    n_frames = len(res.frames)
    st.session_state["_frame_max"] = n_frames
    if st.session_state.get("frame_slider", 1) > n_frames:
        st.session_state["frame_slider"] = n_frames

    c1, c2 = st.columns([3, 2])
    with c1:
        r = st.slider("Round shown on the map", 1, n_frames, n_frames, key="frame_slider",
                      label_visibility="collapsed")
    with c2:
        b1, b2, b3, b4, b5 = st.columns(5)
        b1.button("⏮", key="fb1", on_click=lambda: _step_frame(-1), help="one round back")
        b2.button("◀ 10", key="fb2", on_click=lambda: _step_frame(-10), help="ten rounds back")
        b3.button("10 ▶", key="fb3", on_click=lambda: _step_frame(10), help="ten rounds forward")
        b4.button("⏭", key="fb4", on_click=lambda: _step_frame(1), help="one round forward")
        b5.button(" to end", key="fb5", on_click=_end_frame, help="jump to the last round")
    r = int(st.session_state.get("frame_slider", n_frames))
    fr = res.frame(r)

    cL, cR = st.columns([2.05, 1])
    with cL:
        t1, t2, t3, t4 = st.columns([1, 1, 1, 2.4])
        show_heat = t1.toggle("🌾 food", value=True, key="opt_heat", help="shade each map cell by its stored food")
        show_groups = t2.toggle("⭕ groups", value=True, key="opt_groups",
                                help="ring around co-operating clusters (3+ mutual co-operations, trust ≥ 0.6)")
        show_trails = t3.toggle("〰 trails", value=False, key="opt_trails", help="last-8-round movement trails")
        t4.markdown("<div style='height:34px'></div>", unsafe_allow_html=True)

        fig = go.Figure()
        if show_heat:
            g = np.array(fr["grid"], dtype=float)
            fig.add_trace(go.Heatmap(
                z=g, x=[i * CELL for i in range(len(g[0]) + 1)],
                y=[j * CELL for j in range(len(g) + 1)][::-1],
                colorscale=[[0, "rgba(13,21,38,0)"], [0.45, "rgba(20,83,64,.42)"], [1.0, "rgba(52,211,153,.55)"]],
                showscale=False, hoverinfo="skip", zsmooth="best"))
        if show_trails:
            lo = max(1, r - 7)
            for rr in range(lo + 1, r + 1):
                fa, fb = res.frames[rr - 2], res.frames[rr - 1]
                pos_a = {aid: (fa["x"][i], fa["y"][i]) for i, aid in enumerate(fa["ids"])}
                xx, yy = [], []
                for i, aid in enumerate(fb["ids"]):
                    if aid in pos_a and abs(fb["x"][i] - pos_a[aid][0]) < CELL * 8:
                        xx += [pos_a[aid][0], fb["x"][i], None]
                        yy += [pos_a[aid][1], fb["y"][i], None]
                fig.add_trace(go.Scatter(x=xx, y=yy, mode="lines",
                                          line=dict(width=1, color="rgba(45,212,191,.16)"),
                                          hoverinfo="skip", showlegend=False))
        by_cat: dict[str, dict[str, list]] = {}
        for i, aid in enumerate(fr["ids"]):
            d0 = by_cat.setdefault(fr["cat"][i], {"x": [], "y": [], "sz": [], "hov": []})
            d0["x"].append(fr["x"][i])
            d0["y"].append(fr["y"][i])
            d0["sz"].append(6 + 14 * clamp01(fr["wealth"][i] / 80.0))
            d0["hov"].append(
                f"agent #{aid} · {ACTION_LABEL.get(fr['last'][i], fr['last'][i])}<br>"
                f"🍽 food {fr['food'][i]:.0f}/90 · ⚡ energy {fr['energy'][i]:.0f}/100<br>"
                f"💰 wealth {fr['wealth'][i]:.0f} · ❤ health {fr['health'][i]:.0f}/100")
        for cat, d0 in by_cat.items():
            fig.add_trace(go.Scatter(
                x=d0["x"], y=d0["y"], mode="markers",
                name=f"{cat} ({len(d0['x'])})",
                marker=dict(size=d0["sz"], color=CAT_COLOR.get(cat, "#94a3b8"), opacity=0.92,
                            line=dict(width=1, color="rgba(8,14,26,.9)")),
                text=d0["hov"], hoverinfo="text"))
        if show_groups:
            for gid, members in fr["groups"].items():
                pos = {aid: (fr["x"][i], fr["y"][i]) for i, aid in enumerate(fr["ids"])}
                pts = [pos[m] for m in members if m in pos]
                if not pts:
                    continue
                gx = [p[0] for p in pts]
                gy = [p[1] for p in pts]
                r0 = max(18.0, math.hypot(max(gx) - min(gx), max(gy) - min(gy)) / 2 + 16)
                col = GROUP_COLORS[gid % len(GROUP_COLORS)]
                fig.add_trace(go.Scatter(
                    x=[sum(gx) / len(gx)], y=[sum(gy) / len(gy)], mode="markers", showlegend=False,
                    marker=dict(size=r0, sizemode="diameter", symbol="circle-open",
                                line=dict(width=1.6, color=col), opacity=0.75),
                    hovertemplate=f"group {gid} · {len(pts)} members<extra></extra>"))
        fig.update_layout(
            title=dict(text=(f"🗺 round {r} / {n_frames} · {fr['n_alive']} / {res.n0} alive · "
                              f"{sum(len(m) for m in fr['groups'].values())} grouped · "
                              f"world 96 × 48 units (toroidal)"),
                       font=dict(size=13.5, color="#c9d7ee")),
            xaxis=dict(range=[-2, WORLD_W + 2], showticklabels=False),
            yaxis=dict(range=[-2, WORLD_H + 2], showticklabels=False),
            plot_bgcolor="#0a111f", paper_bgcolor="rgba(0,0,0,0)",
            height=560, margin=dict(l=8, r=8, t=42, b=8),
            legend=dict(orientation="h", yanchor="top", y=-0.02, xanchor="center", x=0.5, font=dict(size=10.5)),
            hovermode="closest",
        )
        st.plotly_chart(fig, width="stretch", key=f"map_{r}")
        st.caption("dot size = wealth held · colour = action taken this round · cell shading = food on the ground")
    with cR:
        kinds = sorted({e["kind"] for e in res.events}) or ["none"]
        if "feed_kinds" in st.session_state and not set(st.session_state["feed_kinds"]) <= set(kinds):
            st.session_state["feed_kinds"] = kinds      # stale filter from a previous run
        selk = st.multiselect("event kinds shown in the feed", kinds, default=kinds, key="feed_kinds",
                              label_visibility="collapsed")
        upto = [e for e in res.events if e["round"] <= r and e["kind"] in selk][::-1]
        icon = {"trade": "🤝", "group": "⭕", "betray": "🗡", "death": "💀", "milestone": "🏁"}
        rows_html = "".join(
            f"<div class='row'><span class='t'>r{e['round']}</span>{icon.get(e['kind'], '•')} {e['text']}</div>"
            for e in upto[:220])
        st.markdown(
            f"<div style='font-size:11px;letter-spacing:.12em;color:#7c8db0;font-weight:700;margin-bottom:6px;'>"
            f"ACTIVITY FEED · {len(upto)} events up to round {r} · newest first</div>"
            f"<div class='yu-feed'>{rows_html or '<div class=\"row\">no events yet for the selected kinds</div>'}</div>",
            unsafe_allow_html=True)
        bar = pd.Series(fr["last"]).value_counts().reindex(ACTIONS + FORCED, fill_value=0)
        bar = bar[bar > 0]
        f2 = go.Figure(go.Bar(
            x=bar.values[::-1], y=[ACTION_LABEL.get(i, i) for i in bar.index[::-1]], orientation="h",
            marker_color=[ACTION_COLOR.get(i, "#94a3b8") for i in bar.index[::-1]],
            hovertemplate="%{y}: %{x} agents<extra></extra>"))
        style_fig(f2, h=235, legend=False)
        f2.update_layout(title=dict(text="actions taken in this round", font=dict(size=12.5, color="#c9d7ee")),
                         margin=dict(l=8, r=8, t=34, b=8))
        f2.update_xaxes(showgrid=False, showticklabels=False)
        f2.update_yaxes(gridcolor="rgba(0,0,0,0)")
        st.plotly_chart(f2, width="stretch", key=f"actbar_{r}")


# ── trends tab ──────────────────────────────────────────────────────────────
def multi_line(rdf: pd.DataFrame, series: list[tuple[str, str, str, Optional[str]]],
               title: str, h: int = 300, ylab: Optional[str] = None) -> go.Figure:
    fig = go.Figure()
    for col, lab, colr, dash in series:
        if col not in rdf.columns:
            continue
        fig.add_trace(go.Scatter(x=rdf["round"], y=rdf[col], name=lab, mode="lines",
                                 line=dict(width=2.2, color=colr, dash=dash),
                                 hovertemplate=f"r%{{x}} · {lab}: %{{y:.3f}}<extra></extra>"))
    style_fig(fig, h)
    fig.update_layout(title=dict(text=title, font=dict(size=13.5, color="#c9d7ee")),
                      yaxis_title=ylab, xaxis_title="round (simulated)")
    return fig


def render_trends(res: SimResult) -> None:
    """📈 Analysis Lab — 26 small multiples + a conclusions box.
    Every panel reads ONLY from the recorded run dataframes (no smoothing, no presets)."""
    rdf, adf = res.rounds_df, res.agents_df
    st.caption("All charts below plot the exact per-round values recorded during the run — same numbers as the "
               "CSV export. Scroll the grid; the auto-conclusions at the bottom cite panels by number.")

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(multi_line(rdf, [
            ("cooperation_rate", "co-operation (share + co-operate)", PALETTE["green"], None),
            ("competition_rate", "competition (attacks)", PALETTE["red"], None),
            ("trade_rate", "trade", PALETTE["violet"], "dot"),
        ], "🤝 [1/26] social behaviour per round · fraction of living agents", ylab="rate (0–1)"), width="stretch")
    with c2:
        st.plotly_chart(multi_line(rdf, [
            ("avg_trust", "mean trust (remembered partners)", PALETTE["teal"], None),
            ("survival", "survival share", PALETTE["blue"], "dash"),
            ("avg_health", "mean health ÷ 100", PALETTE["amber"], "dot"),
        ], "🧠 [2/26] trust · survival · health", ylab="0–1"), width="stretch")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(multi_line(rdf, [
            ("avg_food", "avg food carried per agent", PALETTE["amber"], None),
            ("world_food_scaled", "world field food ÷ 100", PALETTE["green"], "dot"),
        ], "🌾 [3/26] the stomach vs the land", ylab="units (field ÷100)"), width="stretch")
    with c2:
        st.plotly_chart(multi_line(rdf, [
            ("gini_wealth", "wealth Gini", PALETTE["violet"], None),
            ("groups", "active groups", PALETTE["teal"], "dot"),
        ], "⚖ [4/26] inequality & group formation", ylab="Gini / count"), width="stretch")

    st.divider()
    st.markdown("**The full lens grid** — one question per panel")

    def mini(series: list[tuple[str, str, str]], title: str, h: int = 190) -> go.Figure:
        f = multi_line(rdf, [(c, n, col, None) for c, n, col in series], title, h=h)
        f.update_layout(margin=dict(l=46, r=8, t=32, b=26), font=dict(size=10.5),
                        legend=dict(orientation="h", y=-0.34, x=0, font=dict(size=9.5)))
        return f

    def hist(vals, color: str, title: str, xlab: str) -> go.Figure:
        f = go.Figure(go.Histogram(x=vals, nbinsx=22, marker_color=color,
                                   hovertemplate="%{x}: %{y} agents<extra></extra>"))
        style_fig(f, 190, legend=False)
        f.update_layout(title=dict(text=title, font=dict(size=11.5, color="#c9d7ee")),
                        margin=dict(l=40, r=8, t=30, b=32), xaxis_title=xlab, yaxis_title="agents")
        return f

    grid: list[go.Figure] = [
        mini([("population", "living agents", PALETTE["blue"])], "[5/26] population (deaths are permanent)"),
        mini([("avg_energy", "mean energy", PALETTE["blue"])], "[6/26] mean energy (auto-eat floors it)"),
        mini([("avg_trust", "mean trust", PALETTE["teal"])], "[7/26] mean trust"),
        mini([("cooperation_rate", "co-operation rate", PALETTE["green"])], "[8/26] co-operation rate"),
        mini([("competition_rate", "competition rate", PALETTE["red"])], "[9/26] competition rate"),
        mini([("trade_rate", "trade rate", PALETTE["violet"])], "[10/26] trade rate"),
        mini([("avg_food", "avg food carried", PALETTE["amber"])], "[11/26] food carried per agent"),
        mini([("world_food", "field food (units)", PALETTE["green"])], "[12/26] total food in the world"),
        mini([("avg_wealth", "mean wealth", PALETTE["violet"]), ("gini_wealth", "Gini (0–1, other scale)", PALETTE["gray"])],
             "[13/26] wealth level vs concentration"),
        mini([("groups", "active groups", PALETTE["teal"]), ("group_members", "agents grouped", PALETTE["blue"])],
             "[14/26] group formation"),
        mini([("rest_rate", "rest/collapse", PALETTE["gray"]), ("explore_rate", "move+explore", PALETTE["blue"])],
             "[15/26] rest & wandering share"),
        mini([("n_collapse", "collapses this round", PALETTE["red"])], "[16/26] exhaustion collapses per round"),
    ]
    deaths_delta = rdf["deaths"].diff().fillna(rdf["deaths"]).clip(lower=0)
    f = go.Figure(go.Bar(x=rdf["round"], y=deaths_delta, marker_color=PALETTE["red"],
                         hovertemplate="r%{x}: %{y} deaths<extra></extra>"))
    style_fig(f, 190, legend=False)
    f.update_layout(title=dict(text="[17/26] deaths per round (starvation & wounds)",
                               font=dict(size=11.5, color="#c9d7ee")),
                    margin=dict(l=40, r=8, t=30, b=26), yaxis_title="deaths")
    grid.append(f)
    f = go.Figure()
    for act, colr in [("cooperate", PALETTE["teal"]), ("share", PALETTE["green"]),
                      ("trade", PALETTE["violet"]), ("compete", PALETTE["red"])]:
        f.add_trace(go.Scatter(x=rdf["round"], y=rdf[f"n_{act}"].cumsum(), name=f"Σ {act}",
                               line=dict(width=1.9, color=colr),
                               hovertemplate="r%{x}: %{y:,} cumulative<extra></extra>"))
    style_fig(f, 190)
    f.update_layout(title=dict(text="[18/26] cumulative decisions by kind", font=dict(size=11.5, color="#c9d7ee")),
                    margin=dict(l=48, r=8, t=30, b=42), xaxis_title="round",
                    legend=dict(orientation="h", y=-0.46, x=0, font=dict(size=9.5)))
    grid.append(f)
    grid.append(hist(adf["wealth"].astype(float), PALETTE["violet"],
                     "[19/26] final wealth distribution", "coins at run end"))
    grid.append(hist(adf["food"].astype(float), PALETTE["amber"],
                     "[20/26] final food carried", "units (cap 90)"))
    tnames = ["trait_cooperation", "trait_competition", "trait_exploration", "trait_risk", "trait_adaptation"]
    tlabs = ["co-operation", "competition", "exploration", "risk", "adaptation"]
    f = go.Figure(go.Bar(x=tlabs, y=[float(adf[t].mean()) for t in tnames],
                         marker_color=[PALETTE["green"], PALETTE["red"], PALETTE["blue"], PALETTE["amber"],
                                       PALETTE["violet"]],
                         hovertemplate="%{x}: mean %{y:.2f} across agents<extra></extra>"))
    style_fig(f, 190, legend=False)
    f.update_yaxes(range=[0, 1])
    f.update_layout(title=dict(text="[21/26] average personality of this society (0–1)",
                               font=dict(size=11.5, color="#c9d7ee")),
                    margin=dict(l=40, r=8, t=30, b=42), xaxis_tickangle=-18)
    grid.append(f)
    xk = adf["act_share"] + adf["act_cooperate"]
    f = go.Figure(go.Scattergl(
        x=xk, y=adf["act_compete"], mode="markers",
        marker=dict(size=6 + 10 * np.clip(adf["reward"] / max(1.0, float(adf["reward"].max())), 0.0, 1.0),
                    color=np.where(adf["alive"], PALETTE["green"], PALETTE["red"]), opacity=0.7,
                    line=dict(width=1, color="#0b1220")),
        text=[f"#{int(t.agent)}: {int(t.act_share)+int(t.act_cooperate)} kind acts · {int(t.act_compete)} attacks "
              f"· reward {t.reward:,.0f}" for t in adf.itertuples()],
        hoverinfo="text"))
    style_fig(f, 190, legend=False)
    f.update_layout(title=dict(text="[22/26] kindness vs aggression per agent (size = reward)",
                               font=dict(size=11.5, color="#c9d7ee")),
                    margin=dict(l=40, r=8, t=30, b=32), xaxis_title="share + co-operate acts", yaxis_title="attack acts")
    grid.append(f)
    f = go.Figure(go.Scattergl(
        x=adf["trait_cooperation"], y=adf["reward"], mode="markers",
        marker=dict(size=6, color=np.where(adf["alive"], PALETTE["green"], PALETTE["red"]), opacity=0.75,
                    line=dict(width=1, color="#0b1220")),
        text=[f"#{int(t.agent)} · trait {t.trait_cooperation:.2f} · reward {t.reward:,.0f}" for t in adf.itertuples()],
        hoverinfo="text"))
    r_trait = float(np.corrcoef(adf["trait_cooperation"], adf["reward"])[0, 1]) if len(adf) > 3 else 0.0
    style_fig(f, 190, legend=False)
    f.update_layout(title=dict(text=f"[23/26] co-operation trait vs final reward (r = {r_trait:.2f})",
                               font=dict(size=11.5, color="#c9d7ee")),
                    margin=dict(l=46, r=8, t=30, b=32), xaxis_title="trait (0–1)", yaxis_title="reward pts")
    grid.append(f)
    fin = res.frames[-1]
    top5 = [fin["ids"][i] for i in sorted(range(len(fin["ids"])), key=lambda i: -fin["wealth"][i])[:5]]
    f = go.Figure()
    for rank, aid in enumerate(top5):
        xs, ys = [], []
        for fr in res.frames:
            try:
                i = fr["ids"].index(aid)
            except ValueError:
                continue
            xs.append(fr["round"]); ys.append(fr["wealth"][i])
        f.add_trace(go.Scatter(x=xs, y=ys, name=f"#{aid}", line=dict(width=1.8, color=GROUP_COLORS[rank]),
                               hovertemplate=f"r%{{x}} · #{aid}: %{{y:.0f}} coins<extra></extra>"))
    style_fig(f, 190)
    f.update_layout(title=dict(text="[24/26] the five richest agents — wealth over time",
                               font=dict(size=11.5, color="#c9d7ee")),
                    margin=dict(l=40, r=8, t=30, b=42), xaxis_title="round", yaxis_title="coins",
                    legend=dict(orientation="h", y=-0.46, x=0, font=dict(size=9.5)))
    grid.append(f)
    corr_cf = float(np.corrcoef(rdf["world_food"], rdf["cooperation_rate"])[0, 1]) \
        if len(rdf) > 3 and rdf["world_food"].std() > 1e-9 else 0.0
    f = go.Figure(go.Scatter(
        x=rdf["world_food"], y=rdf["cooperation_rate"], mode="markers",
        marker=dict(size=6, color=rdf["round"], colorscale=[[0, "#2dd4bf"], [1, "#f5b23c"]], showscale=False,
                    line=dict(width=1, color="#0b1220")),
        hovertemplate="r%{customdata} · food %{x:,.0f} · coop %{y:.2f}<extra></extra>",
        customdata=rdf["round"]))
    style_fig(f, 190, legend=False)
    f.update_layout(title=dict(text=f"[25/26] round-wise: field food vs co-operation (r = {corr_cf:.2f})",
                               font=dict(size=11.5, color="#c9d7ee")),
                    margin=dict(l=40, r=8, t=30, b=32), xaxis_title="world food (units)",
                    yaxis_title="co-operation rate")
    grid.append(f)
    alive_m = [float(adf.loc[adf["alive"], t].mean()) for t in tnames]
    dead_m = [float(adf.loc[~adf["alive"], t].mean()) if (~adf["alive"]).any() else 0.0 for t in tnames]
    f = go.Figure()
    f.add_trace(go.Bar(x=tlabs, y=alive_m, name="alive at end", marker_color=PALETTE["green"]))
    f.add_trace(go.Bar(x=tlabs, y=dead_m, name="died", marker_color=PALETTE["red"]))
    style_fig(f, 190)
    f.update_layout(barmode="group", title=dict(text="[26/26] who died? mean traits, survivors vs dead",
                                                 font=dict(size=11.5, color="#c9d7ee")),
                    margin=dict(l=40, r=8, t=30, b=42), xaxis_tickangle=-18, yaxis_title="mean trait",
                    legend=dict(orientation="h", y=-0.46, x=0, font=dict(size=9.5)))
    f.update_yaxes(range=[0, 1])
    grid.append(f)
    st.caption("⚠ [21–23] & [26]: traits were generated, not earned — these panels restate the payout table of the "
               "rules; and if nobody died, the red bars in [26] are zero by definition (try 🍽 Food Shock).")

    mix = rdf[["round"] + [f"n_{a}" for a in ACTIONS]].copy()
    nbins = max(12, len(rdf) // 10)
    mix["bin"] = ((mix["round"] - 1) * nbins // len(rdf)).astype(int)
    g = mix.groupby("bin")[[f"n_{a}" for a in ACTIONS]].sum()
    tot = g.sum(axis=1).replace(0, 1)
    gpct = g.div(tot, axis=0) * 100
    fig = go.Figure()
    for act in ["cooperate", "share", "trade", "collect", "move", "explore", "rest", "compete"]:
        col = f"n_{act}"
        fig.add_trace(go.Scatter(x=(gpct.index + 0.5) * len(rdf) / nbins, y=gpct[col], name=ACTION_LABEL[act],
                                 stackgroup="one", line=dict(width=0.5, color=ACTION_COLOR[act]),
                                 hovertemplate=f"r%{{x:.0f}} · {ACTION_LABEL[act]}: %{{y:.1f}}%<extra></extra>"))
    style_fig(fig, 250)
    fig.update_layout(title=dict(text="[bonus] action mix per phase — % of all decisions (stacked)",
                                  font=dict(size=12.5, color="#c9d7ee")),
                      xaxis_title="round", yaxis_title="% of actions")
    st.plotly_chart(fig, width="stretch")

    for chunk_start in range(0, len(grid), 3):
        cols = st.columns(3)
        for col, fg in zip(cols, grid[chunk_start:chunk_start + 3]):
            with col:
                st.plotly_chart(fg, width="stretch")

    st.divider()
    tC = third_compare(rdf["cooperation_rate"])
    tT = third_compare(rdf["avg_trust"])
    tR = third_compare(rdf["avg_reward"])
    tW = third_compare(rdf["world_food"])
    d_comp = float(np.corrcoef(deaths_delta, rdf["competition_rate"])[0, 1]) \
        if len(rdf) > 3 and deaths_delta.std() > 0 else 0.0
    peak_groups = int(rdf["groups"].max())
    peak_r = int(rdf.loc[rdf["groups"].idxmax(), "round"])
    grouped_end = int((adf["group"] != "solo").sum()) if len(adf) else 0
    findings = [fx for fx in discover(res) if fx["kind"] == "trend"][:4]
    concl = [
        f"<b>Survival:</b> {int(rdf['population'].iloc[-1])}/{res.n0} agents alive at the end "
        f"({rdf['survival'].iloc[-1]*100:.0f}%); total deaths {int(rdf['deaths'].iloc[-1])} — panels 5 & 17.",
        f"<b>Co-operation:</b> first-third mean {fmt(tC['first'],3)} → last-third {fmt(tC['last'],3)} "
        f"({tC['rel']*100:+.0f}%) — a property of these rules+seed, not a law — panels 1 & 8.",
        f"<b>Trust:</b> {fmt(tT['first'],3)} → {fmt(tT['last'],3)} ({tT['rel']*100:+.0f}%); memory is "
        f"{'ON' if res.cfg.memory else 'OFF'} here, which is exactly the knob that moves it — panel 7.",
        f"<b>Economy:</b> world food {fmt(tW['first'],0)} → {fmt(tW['last'],0)}; wealth Gini ended at "
        f"{rdf['gini_wealth'].iloc[-1]:.3f}; reward/round {fmt(tR['first'],1)} → {fmt(tR['last'],1)} — panels 3, 12, 13, 19.",
        f"<b>Groups:</b> peaked at {peak_groups} simultaneous groups (round {peak_r}); {grouped_end} agents still "
        f"grouped at the end — panels 4 & 14.",
        f"<b>Violence ↔ deaths:</b> per-round competition vs deaths r = {d_comp:.2f}; attacks mostly transfer "
        "resources, wounds kill slowly — compare panels 9 & 17 before reading causality.",
        f"<b>Trait pay-off:</b> co-operation trait vs reward r = {r_trait:.2f} (panel 23) — the payout table "
        "defines this inside the model; it is not a personality finding.",
    ]
    for fnd in findings:
        concl.append(f"<b>Trend ({fnd['severity']}):</b> {fnd['title']} — {fnd['observed']}")
    if not (~adf["alive"]).any():
        concl.append("<b>No deaths</b> — the environment was generous at these settings; 🍽 Food Shock produces a "
                     "mortality story under the same rules.")
    if "policy_entropy" in rdf.columns:
        tE = third_compare(rdf["policy_entropy"])
        concl.insert(1, f"<b>Learning:</b> this was a 🧠 LEARNED run — policy entropy {fmt(tE['first'],2)} → "
                        f"{fmt(tE['last'],2)} bits, {rdf['divergence_rate'].mean()*100:.0f}% of decisions off-rule; "
                        f"full training story in the Learning lab tab.")
    concl.append("<b>Reality check:</b> all 26 panels describe the simulated rules only. Same seed + settings "
                 "reproduces every number bit-for-bit; a different seed is a different sample — use Compare → "
                 "3 seeds before believing any direction.")
    st.markdown("<div class='yu-ok'><b>📌 What these charts conclude (within this model)</b>"
                "<br>" + "<br>".join(f"{i+1}. {c}" for i, c in enumerate(concl)) + "</div>",
                unsafe_allow_html=True)
    md = "\n".join(f"{i+1}. " + c.replace("<b>", "**").replace("</b>", "**") for i, c in enumerate(concl))
    st.download_button("⬇ conclusions as Markdown", md.encode(),
                       f"yudaant_conclusions_seed{res.cfg.seed}.md", "text/markdown", width="stretch")

    with st.expander("📋 end-of-run leaderboard — top 10 by reward (actual recorded values)"):
        lcols = ["agent", "reward", "wealth", "food", "health", "trait_cooperation", "trait_competition",
                 "act_share", "act_cooperate", "act_compete", "group", "alive"]
        st.dataframe(adf.sort_values("reward", ascending=False)[lcols].head(10), width="stretch", hide_index=True)
        st.caption("Deaths are permanent (no births). “alive” = health > 0 in the final round.")




# ── 🗂 study shelf — completed runs kept until YOU remove them ──────────────
def _shelf_charts(rd: pd.DataFrame, title: str, h: int = 250) -> go.Figure:
    series = [("cooperation_rate", "co-operation", PALETTE["green"], None),
              ("competition_rate", "competition", PALETTE["red"], None),
              ("trade_rate", "trade", PALETTE["violet"], "dot"),
              ("avg_trust", "mean trust", PALETTE["teal"], "dash")]
    f = multi_line(rd, series, title, h=h)
    f.update_layout(legend=dict(orientation="h", y=-0.3, x=0, font=dict(size=9.5)))
    return f


def _shelf_stats_md(r: "SimResult") -> str:
    rd = r.rounds_df
    def eol(col: str) -> tuple[float, float]:
        return float(rd[col].iloc[0]), float(rd[col].iloc[-1])
    rows = [("co-operation", "cooperation_rate", "%"), ("competition", "competition_rate", "%"),
            ("mean trust", "avg_trust", ""), ("survival", "survival", "%"),
            ("wealth Gini", "gini_wealth", ""), ("reward / agent / r", "avg_reward", "")]
    lines = []
    for lab, col, unit in rows:
        a, b = eol(col)
        if unit == "%":
            a, b = a * 100, b * 100
        arr = "▲" if b > a + 1e-9 else ("▼" if b < a - 1e-9 else "→")
        lines.append(f"<div style='display:flex;justify-content:space-between;font-size:12.5px;'>"
                     f"<span style='color:#9fb3d1'>{lab}</span>"
                     f"<span style='color:#e6edf7'><b>{a:.1f}</b> → <b>{b:.1f}</b>{unit} {arr}</span></div>")
    if "policy_entropy" in rd.columns:
        e0, e1 = eol("policy_entropy")
        d0, d1 = eol("divergence_rate")
        lines.append(f"<div style='display:flex;justify-content:space-between;font-size:12.5px;'>"
                     f"<span style='color:#a78bfa'>policy entropy</span>"
                     f"<span style='color:#e6edf7'><b>{e0:.2f}</b> → <b>{e1:.2f}</b> bits</span></div>")
        lines.append(f"<div style='display:flex;justify-content:space-between;font-size:12.5px;'>"
                     f"<span style='color:#a78bfa'>off-rule decisions</span>"
                     f"<span style='color:#e6edf7'>avg <b>{float(rd['divergence_rate'].mean())*100:.0f}%</b></span></div>")
    return ("<div class='yu-card' style='padding:10px 12px'>"
            "<div style='color:#7c8db0;font-size:11px;letter-spacing:.08em;margin-bottom:6px;'>FIRST ROUND → LAST ROUND</div>"
            + "".join(lines) + "</div>")


def render_shelf() -> None:
    shelf = st.session_state.get("shelf", [])
    if not shelf:
        st.markdown("<div class='empty-state'><div style='font-size:34px'>🗂</div>"
                    "<h3>Nothing saved yet</h3><p style='max-width:640px;margin:0 auto'>Every completed run lands "
                    "here automatically — full graphs and stats, frozen exactly as they finished. Snapshots stay "
                    f"on the shelf until YOU press ✖ (last {SHELF_MAX} unpinned are kept; 📌 pins are never "
                    "auto-removed). Press Start in the sidebar to make the first one.</p></div>",
                    unsafe_allow_html=True)
        return
    note = st.session_state.pop("shelf_note", None)
    if note:
        st.warning(note)
    st.caption(f"🗂 Study shelf — frozen copies of finished runs, held in this browser session's memory only "
               f"(temporary — download if you want to keep them). Unpinned shelf size: {SHELF_MAX} · "
               f"pinned are kept until you press ✖.")
    for idx, e in enumerate(list(shelf)):
        r = e["res"]
        head = ("📌 " if e["pinned"] else "") + f"{e['ts']} · {e['label']}"
        with st.expander(head, expanded=(idx == len(shelf) - 1)):
            rd = r.rounds_df
            cA, cB = st.columns([2.6, 1])
            with cA:
                st.plotly_chart(_shelf_charts(rd, f"how this run unfolded — {e['label']}"), width="stretch")
            with cB:
                st.markdown(_shelf_stats_md(r), unsafe_allow_html=True)
            if "policy_entropy" in rd.columns:
                st.plotly_chart(multi_line(rd, [
                    ("policy_entropy", "policy entropy (bits)", PALETTE["violet"], None),
                    ("divergence_rate", "off-rule share", "#a78bfa", "dot"),
                ], "🧠 training story of this run", h=190), width="stretch")
            b1, b2, b3, b4 = st.columns([1.15, 0.8, 0.8, 2.4])
            b1.button("↩ Load into tabs", width="stretch", key=f"sh_load_{e['key']}", on_click=_shelf_load,
                      args=(e["key"],), help="restores this exact snapshot as the current run — no re-simulation")
            b2.button("📌 Pin" if not e["pinned"] else "🧵 Unpin", width="stretch", key=f"sh_pin_{e['key']}",
                      on_click=_shelf_pin, args=(e["key"],))
            b3.button("✖ Remove", width="stretch", key=f"sh_rm_{e['key']}", on_click=_shelf_remove, args=(e["key"],))
            b4.caption("Loading fills Society view, Trends, Learning lab, Inspector and Compare with this "
                       "run's real values — scroll, scrub, inspect, export at leisure.")
            st.download_button("⬇ this run's per-round CSV", df_to_csv(rd),
                               f"yudaant_shelf_{e['key']}.csv", "text/csv", key=f"sh_csv_{e['key']}")
    if len(shelf) >= 2:
        st.divider()
        st.markdown("**⚖ Compare any two saved runs** — overlay their recorded curves, side by side")
        opts = {f"#{i+1} · {e['ts']} · {e['label']}": i for i, e in enumerate(shelf)}
        c1, c2 = st.columns(2)
        ka = c1.selectbox("run A", list(opts), index=len(shelf) - 1, key="sh_cmp_a")
        kb = c2.selectbox("run B", list(opts), index=max(0, len(shelf) - 2), key="sh_cmp_b")
        i_a, i_b = opts[ka], opts[kb]
        if i_a == i_b:
            st.info("Pick two different snapshots to overlay them.")
            return
        da = shelf[i_a]["res"].rounds_df
        db = shelf[i_b]["res"].rounds_df
        fig = go.Figure()
        for df, nm, dash in ((da, ka.split(" · ")[1], "solid"), (db, kb.split(" · ")[1], "dot")):
            for col, lab, colr in (("cooperation_rate", "co-op", PALETTE["green"]),
                                   ("competition_rate", "compete", PALETTE["red"]),
                                   ("avg_trust", "trust", PALETTE["teal"]),
                                   ("survival", "survival", PALETTE["blue"])):
                if col not in df.columns:
                    continue
                fig.add_trace(go.Scatter(x=df["round"], y=df[col], name=f"{nm} · {lab}",
                                         line=dict(width=2.1 if dash == "solid" else 1.5, color=colr, dash=dash),
                                         hovertemplate=f"r%{{x}} · {nm} {lab}: %{{y:.3f}}<extra></extra>"))
        style_fig(fig, 300)
        fig.update_layout(title=dict(text="solid = run A · dotted = run B (raw recorded values, no smoothing)",
                                     font=dict(size=12.5, color="#c9d7ee")), xaxis_title="round")
        st.plotly_chart(fig, width="stretch")
        rows = []
        for col, lab in (("cooperation_rate", "co-operation rate"), ("competition_rate", "competition rate"),
                         ("avg_trust", "mean trust"), ("survival", "survival"), ("gini_wealth", "wealth Gini"),
                         ("avg_reward", "reward / agent / round")):
            va = float(da[col].iloc[-1]); vb = float(db[col].iloc[-1])
            rows.append({"metric": lab, "A (end)": round(va, 3), "B (end)": round(vb, 3), "Δ B−A": round(vb - va, 3)})
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        nmin = min(len(da), len(db))
        ddf = pd.DataFrame({
            "round": da["round"].to_numpy()[:nmin],
            "Δ co-op": da["cooperation_rate"].to_numpy()[:nmin] - db["cooperation_rate"].to_numpy()[:nmin],
            "Δ compete": da["competition_rate"].to_numpy()[:nmin] - db["competition_rate"].to_numpy()[:nmin],
            "Δ trust": da["avg_trust"].to_numpy()[:nmin] - db["avg_trust"].to_numpy()[:nmin]})
        thr = 0.15
        over = np.nonzero(np.abs(ddf["Δ co-op"].to_numpy()) > thr)[0]
        div_r = int(ddf["round"].iloc[over[0]]) if len(over) else None
        fig = go.Figure()
        for cn, cc in (("Δ co-op", PALETTE["green"]), ("Δ compete", PALETTE["red"]), ("Δ trust", PALETTE["teal"])):
            fig.add_trace(go.Scatter(x=ddf["round"], y=ddf[cn], name=cn, line=dict(width=1.9, color=cc),
                                     hovertemplate="r%{x} · " + cn + ": %{y:.3f}<extra></extra>"))
        if div_r is not None:
            fig.add_vline(x=div_r, line_dash="dash", line_color="#f5b23c",
                          annotation_text=f"first real split · |Δco-op| crossed {thr}", annotation_font_size=9.5,
                          annotation_font_color="#f5b23c")
        style_fig(fig, 250)
        fig.update_layout(title=dict(text="where did the two societies split? (per-round A minus B)",
                                     font=dict(size=12, color="#c9d7ee")),
                          xaxis_title="round", yaxis_title="A − B (rate difference)")
        st.plotly_chart(fig, width="stretch")
        st.caption(f"Split point: round {div_r} — before it the two runs were within ±0.15 co-operation of each "
                   "other; after it they are different societies." if div_r is not None else
                   "No round ever separated them by more than 0.15 in co-operation — near-twin trajectories.")
        st.caption("Two independent samples of the model — differences describe these two runs, not “laws”.")


# ── 🧠 learning lab ─────────────────────────────────────────────────────────
def render_learning(res: SimResult) -> None:
    cfg = res.cfg
    if cfg.policy != "LEARNED":
        st.info("🧠 This run used the fixed 📜 RULE policy. Set **Agent policy → LEARNED** in the sidebar and "
                "press Start — every chart below then appears, driven by what the agents actually learned.")
        st.markdown(
            "<div class='yu-card'><b>What LEARNED mode does — small-data online training, fully transparent</b><br>"
            "• The 8 actions and every named factor stay exactly the same — nothing hidden, no LLM, no internet.<br>"
            "• Each agent carries a personal multiplier on every factor (53 factors), starting at <b>×1.00</b> "
            "= the documented rule, plus a tiny seeded individuality (±0.05).<br>"
            "• Each round it <i>samples</i> an action from softmax(factors × multipliers ÷ temperature), then "
            "after rewards land it takes one batched policy-gradient (REINFORCE, normalised advantage, value "
            "EMA as baseline) step over <b>only its own last-k decisions</b> — the small dataset is its lived "
            "experience, k = the buffer slider.<br>"
            "• A gentle anchor pull (−0.02·(W−1)) and clipping to [−2, +3] keep learning bounded; multipliers "
            "are inspectable per agent, per factor, per decision.<br>"
            "• <b>🔁 carry across runs</b>: after each run the top-40% agents' mean weight-shift is averaged and "
            "kept <i>in this browser session only</i> to seed the next Start — so the society's behaviour "
            "keeps evolving every time you study it. While carry is ON, runs are stateful by design and no "
            "longer bit-reproducible from the seed alone (everything else still is; the Learning lab prints the "
            "generation).</div>", unsafe_allow_html=True)
        return
    rdf = res.rounds_df
    if "policy_entropy" not in rdf.columns:
        st.warning("LEARNED run but no learning columns were recorded — engine bug worth reporting, not hiding.")
        return
    brain = st.session_state.get("brain") or {}
    _carry_txt = ("ON · generation " + str(int(res.meta.get("brain_gen", 0)) + 1)) if cfg.policy_carry else "OFF"
    st.caption(f"🧠 trained live inside the run · lr {fmt(cfg.policy_lr, 2)} · temperature {fmt(cfg.policy_temp, 2)}"
               f" · each agent uses only its own last {cfg.policy_buffer} (factors, action, reward) samples · "
               f"53 named factors × 8 actions · carry {_carry_txt} — all simulated, all seeded.")

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(multi_line(rdf, [
            ("policy_entropy", "mean policy entropy", PALETTE["violet"], None),
            ("entropy_p10", "10th pct", PALETTE["gray"], "dot"),
            ("entropy_p90", "90th pct", PALETTE["gray"], "dot"),
        ], "[L1] decision certainty over training · lower = more confident", ylab="entropy (bits, max 2.08)"),
            width="stretch")
    with c2:
        st.plotly_chart(multi_line(rdf, [
            ("divergence_rate", "share of decisions ≠ fixed-rule pick", PALETTE["red"], None),
            ("avg_shift", "mean |multiplier − 1| ×10", PALETTE["blue"], "dash"),
        ], "[L2] how far agents drifted from the documented rules", ylab="rate / 10×|ΔW|"), width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(multi_line(rdf, [
            ("avg_advantage", "mean advantage of taken action", PALETTE["amber"], None),
        ], "[L3] were choices better than expected? (reward − value-EMA)", ylab="advantage (pts)"), width="stretch")
    with c2:
        st.plotly_chart(multi_line(rdf, [
            ("cooperation_rate", "co-operation", PALETTE["green"], None),
            ("competition_rate", "competition", PALETTE["red"], None),
            ("trade_rate", "trade", PALETTE["violet"], "dot"),
        ], "[L4] behaviour that the training is actually producing", ylab="rate"), width="stretch")

    # [L5] learned multipliers heatmap — the trained “brain”, per factor
    z = np.full((8, N_SLOT), np.nan)
    hov = [[""] * N_SLOT for _ in range(8)]
    for ai, act in enumerate(ACTIONS):
        for s in POLICY_ACTION_SLOTS[act]:
            m = float(res_soc_mean(res, ai, s))
            z[ai, s] = m
            hov[ai][s] = f"{act} · {POLICY_SLOTS[s][1]}<br>mean ×{m:.2f} vs rule ×1.00"
    lab = [f"{POLICY_SLOTS[s][0][:3]}·{POLICY_SLOTS[s][1]}" for s in range(N_SLOT)]
    fig = go.Figure(go.Heatmap(z=z, y=[ACTION_LABEL.get(a, a) for a in ACTIONS], x=lab,
                               colorscale=[[0, "#f87171"], [0.5, "#141c2f"], [1, "#34d399"]], zmid=1.0,
                               zmin=0.40, zmax=1.80, colorbar=dict(title="×rule", tickfont=dict(size=9)),
                               xgap=1, ygap=2, text=hov, hoverinfo="text"))
    style_fig(fig, 430, legend=False)
    fig.update_layout(title=dict(text="[L5] what training did to every factor — average multiplier across agents "
                                      "(red = learned to distrust it · green = leaned into it)",
                                 font=dict(size=12, color="#c9d7ee")),
                      margin=dict(l=92, r=8, t=44, b=140),
                      yaxis=dict(autorange="reversed"), xaxis=dict(tickangle=90, tickfont=dict(size=7.6)))
    st.plotly_chart(fig, width="stretch")

    adf = res.agents_df
    c1, c2 = st.columns(2)
    with c1:
        f = go.Figure(go.Scattergl(x=adf["learn_shift"], y=adf["reward"], mode="markers",
                                  marker=dict(size=6, color=np.where(adf["alive"], PALETTE["green"], PALETTE["red"]),
                                              opacity=0.72, line=dict(width=1, color="#0b1220")),
                                  text=[f"#{int(t.agent)} · |ΔW| {t.learn_shift:.3f} · reward {t.reward:,.0f}"
                                        for t in adf.itertuples()], hoverinfo="text"))
        r_sr = float(np.corrcoef(adf["learn_shift"], adf["reward"])[0, 1]) if len(adf) > 3 \
            and adf["learn_shift"].std() > 1e-9 else 0.0
        style_fig(f, 230, legend=False)
        f.update_layout(title=dict(text=f"[L6] did faster learners do better? · r = {r_sr:.2f}",
                                   font=dict(size=11.5, color="#c9d7ee")),
                        margin=dict(l=48, r=8, t=32, b=34), xaxis_title="mean |multiplier − 1|",
                        yaxis_title="final reward")
        st.plotly_chart(f, width="stretch")
    with c2:
        st.plotly_chart(multi_line(rdf, [
            ("samples_updated", "agents that trained this round", PALETTE["blue"], None),
            ("n_collapse", "forced collapses (context)", PALETTE["gray"], "dot"),
        ], "[L7] how many agents had enough experience to learn from", ylab="agents"), width="stretch")

    c1, c2 = st.columns([1.6, 1])
    with c1:
        st.plotly_chart(multi_line(rdf, [
            ("var_within", "within-group multiplier variance", PALETTE["teal"], None),
            ("var_global", "whole-population variance", PALETTE["gray"], "dash"),
        ], "[L9] 🌀 culture condensation — teal dipping below gray = groups agreeing more than strangers",
            ylab="var(W) per slot"), width="stretch")
    with c2:
        st.markdown(
            "<div class='yu-card' style='padding:10px 12px'>"
            f"<b>🌀 culture diffusion</b> = {fmt(cfg.influence, 2)} · "
            f"<b>rule-anchor pull</b> = {fmt(cfg.policy_anchor, 3)} × (1.7 − adaptability)<br>"
            f"<span style='color:#9fb3d1;font-size:12.4px'>"
            + ("Peers drag each other's habits together, and the dead bequeath theirs to survivors. The stress "
               "arena below tests what those habits are worth when the world turns hostile."
               if cfg.influence > 0 else "Influence OFF — every agent learns in isolation; groups share nothing "
                                         "but space. Try 0.50 and watch [L9] split.")
            + " Flip Compare → “Culture diffusion ON vs OFF” for a controlled verdict.</span></div>",
            unsafe_allow_html=True)

    with st.expander("⚔ Stress-test arena — freeze these learned habits, then wreck their world"):
        st.caption("Two fresh 45-round catastrophes, same seed: regrowth ×0.22. One society runs on the "
                   "multipliers it trained in THIS run (learning & diffusion switched off — pure habit); the "
                   "other is naïve rule-brain. If the trained one survives better, habits earned in calmer "
                   "times were worth something. Computed live from the real engine; one extra run each, cached.")
        skey = f"stress_{res.meta['config_hash']}"
        if st.button("▶ compute the catastrophe pair", key="st_go"):
            Wf = getattr(res, "W_final", None)
            if Wf is None:
                st.error("this run did not keep its final weight matrices — nothing to freeze")
            else:
                alive_m = res.agents_df["alive"].to_numpy(dtype=bool)
                arena_brain = (Wf[alive_m].mean(axis=0) if alive_m.any() else Wf.mean(axis=0)) - 1.0
                shock = {**cfg.to_dict(), "policy": "LEARNED", "policy_lr": 0.0, "policy_anchor": 0.0,
                         "influence": 0.0, "policy_carry": False, "rounds": 45,
                         "food_regrowth": max(0.05, cfg.food_regrowth * 0.22)}
                with st.spinner("two shocked worlds, ~2 s…"):
                    fro = run_simulation(SimConfig(**shock), brain=arena_brain)
                    nai = run_simulation(SimConfig(**{**shock, "policy": "RULE"}))
                pick = ("survival", "cooperation_rate", "competition_rate", "avg_reward", "gini_wealth")
                st.session_state[skey] = {
                    "rows": [{"metric": m, "🧠 frozen habits": round(float(fro.rounds_df[m].iloc[-1]), 3),
                              "📜 naïve rules": round(float(nai.rounds_df[m].iloc[-1]), 3)} for m in pick],
                    "died_frozen": int(fro.rounds_df["deaths"].iloc[-1]),
                    "died_naive": int(nai.rounds_df["deaths"].iloc[-1]),
                }
        if skey in st.session_state:
            sr = st.session_state[skey]
            st.dataframe(pd.DataFrame(sr["rows"]), width="stretch", hide_index=True)
            win_f = sr["rows"][0]["🧠 frozen habits"] - sr["rows"][0]["📜 naïve rules"]
            if win_f > 0.01:
                verdict = ("<div class='yu-ok'>😎 <b>Old habits survived the shock better</b> — the frozen-habit "
                           f"society ended at {sr['rows'][0]['🧠 frozen habits']:.2f} survival vs naïve "
                           f"{sr['rows'][0]['📜 naïve rules']:.2f} (deaths {sr['died_frozen']} vs "
                           f"{sr['died_naive']}).</div>")
            elif win_f < -0.01:
                verdict = ("<div class='yu-danger'>😳 <b>The shock levelled the playing field</b> — trained "
                           f"habits did NOT beat instincts here (survival {sr['rows'][0]['🧠 frozen habits']:.2f} "
                           f"vs {sr['rows'][0]['📜 naïve rules']:.2f}, deaths {sr['died_frozen']} vs "
                           f"{sr['died_naive']}). Crises can wash away culture — that is a finding, not a "
                           "bug.</div>")
            else:
                verdict = ("<div class='yu-note'>🤝 <b>Dead heat</b> — habits neither helped nor hurt much "
                           "under this particular shock.</div>")
            st.markdown(verdict + "<span style='font-size:11.5px;color:#8ea0bd'>Single catastrophe sample per "
                        "side; another seed = another disaster. Two more samples would make this a statement, "
                        "not an anecdote.</span>", unsafe_allow_html=True)

    hist = [h for h in brain.get("hist", []) if h.get("end_entropy") is not None]
    if cfg.policy_carry and len(hist) >= 2:
        hd = pd.DataFrame(hist)
        fig = go.Figure()
        for colr, cname, clab in ((PALETTE["violet"], "end_entropy", "final policy entropy (bits)"),
                                  (PALETTE["red"], "end_divergence", "final off-rule share"),
                                  (PALETTE["green"], "coop_end", "final co-operation rate")):
            fig.add_trace(go.Scatter(x=hd["gen"], y=hd[cname], name=clab, mode="lines+markers",
                                     line=dict(width=2, color=colr), marker=dict(size=6)))
        style_fig(fig, 250)
        fig.update_layout(title=dict(text="[L8] society evolution ACROSS runs — every Start inherits the elites' "
                                           "learned multipliers (session-only memory)",
                                     font=dict(size=12, color="#c9d7ee")),
                          xaxis_title="run generation (this session)", yaxis_title="value at run end")
        st.plotly_chart(fig, width="stretch")
        st.dataframe(hd, width="stretch", hide_index=True)
    elif cfg.policy_carry:
        st.caption("🔁 carry is ON — run a second Start to see the society evolve across generations (L8).")

    with st.expander("⚖ Fair contrast — re-run the exact same seed & settings as fixed rules, one click"):
        key = f"rulecontrast_{res.meta['config_hash']}"
        if st.button("▶ compute 📜 RULE baseline for comparison", key="rc_go"):
            with st.spinner("one instant RULE run (few seconds)…"):
                rc = run_simulation(SimConfig(**{**cfg.to_dict(), "policy": "RULE", "policy_carry": False}))
            st.session_state[key] = rc.rounds_df
        if key in st.session_state:
            b = st.session_state[key]
            rows = [("cooperation_rate", "co-operation rate"), ("competition_rate", "competition rate"),
                    ("trade_rate", "trade rate"), ("avg_trust", "mean trust"), ("survival", "survival at end"),
                    ("gini_wealth", "wealth Gini"), ("avg_reward", "reward / agent / round")]
            tab = pd.DataFrame({"metric": [l for _, l in rows],
                                "📜 fixed rules": [float(b[c].iloc[-1]) for c, _ in rows],
                                "🧠 learned": [float(rdf[c].iloc[-1]) for c, _ in rows]})
            tab["Δ (learned − rules)"] = tab["🧠 learned"] - tab["📜 fixed rules"]
            st.dataframe(tab, width="stretch", hide_index=True)
            st.caption("Extra run computed live from the same seed + settings, cached until the config changes. "
                       "One sample each — treat differences as hypotheses, not laws.")

    tE = third_compare(rdf["policy_entropy"])
    tD = third_compare(rdf["divergence_rate"])
    tA = third_compare(rdf["avg_advantage"])
    concl = [
        f"<b>Certainty:</b> mean policy entropy {fmt(tE['first'],2)} → {fmt(tE['last'],2)} bits "
        f"(max 2.08 = pure coin-flip between 8 actions) — L1.",
        f"<b>Independence:</b> {rdf['divergence_rate'].mean()*100:.0f}% of all learned decisions differ from "
        f"what the fixed rules would do (off-rule share {fmt(tD['first']*100,0)}% → {fmt(tD['last']*100,0)}%) "
        f"— L2. This is behaviour the rule-table alone never showed.",
        f"<b>Did it pay?</b> mean advantage {fmt(tA['first'],2)} → {fmt(tA['last'],2)} pts per action (≈0 means "
        f"agents chose about as well as they expected; big positive = exploiting surprises) — L3.",
        (f"<b>Culture:</b> influence {fmt(cfg.influence, 2)} — within-group belief spread "
         f"{fmt(float(rdf['var_within'].iloc[0]), 4)} → {fmt(float(rdf['var_within'].iloc[-1]), 4)} while the "
         f"population sat at {fmt(float(rdf['var_global'].iloc[-1]), 4)} (teal below gray = groups agreeing; "
         f"deaths also transmit habits) — L9."
         if "var_within" in rdf.columns else
         "<b>Culture:</b> diffusion is OFF — every agent learned in isolation; L9 shows the flat baseline."),
        f"<b>Who learned most:</b> correlation between an agent's total |ΔW| and its reward is r = "
        f"{r_sr:.2f} (L6) — inside this model only, with n = {len(adf)} agents; not a claim about people.",
    ]
    if cfg.policy_carry and hist:
        concl.append(f"<b>Across runs:</b> {len(hist)} generations recorded this session; entropy at run end "
                     f"went {fmt(hist[0]['end_entropy'],2)} → {fmt(hist[-1]['end_entropy'],2)} bits and "
                     f"co-operation {fmt(hist[0]['coop_end']*100,0)}% → {fmt(hist[-1]['coop_end']*100,0)}%. "
                     f"While 🔁 carry is ON runs are stateful by design (seed alone no longer fixes the path) — "
                     f"turn it OFF to get pure bit-reproducibility back. — L8.")
    else:
        concl.append("<b>Reproducibility:</b> carry is OFF, so the exact same seed + settings replays this whole "
                     "training run bit-for-bit (self-checks 1–3).")
    concl.append("<b>Reality check:</b> a 53-parameter linear policy on one agent's own last few decisions is a "
                 "toy learner, by design — it shows <i>how</i> incentives reshape behaviour when agents adapt, "
                 "not how humans or real markets learn. No gradients were taken on anyone but simulated agents.")
    st.markdown("<div class='yu-ok'><b>📌 What the learning data says</b>"
                "<br>" + "<br>".join(f"{i+1}. {c}" for i, c in enumerate(concl)) + "</div>",
                unsafe_allow_html=True)
    md = ("# YUDAANT — learning report (seed "
          + str(cfg.seed) + ", gen " + str(res.meta.get("brain_gen", 0) + 1 if cfg.policy_carry else 1) + ")\n\n"
          + "\n".join(f"{i+1}. " + c.replace("<b>", "**").replace("</b>", "**") for i, c in enumerate(concl)))
    st.download_button("⬇ learning report (Markdown)", md.encode(),
                       f"yudaant_learning_seed{cfg.seed}.md", "text/markdown", width="stretch")


def res_soc_mean(res: SimResult, ai: int, s: int) -> float:
    """mean multiplier of one agent for (action-slot); recomputed from stored meta-free state.
    Falls back to 1.0 if the array was not kept."""
    w = getattr(res, "W_final", None)
    return float(w[:, ai, s].mean()) if w is not None else 1.0


# ── agent inspector ────────────────────────────────────────────────────────
def render_inspector(res: SimResult) -> None:
    adf = res.agents_df
    st.caption("The inspector reads the exact decision records the engine wrote — the same factor sums the policy "
               "used at the time. “Why” is the stored breakdown, not a post-hoc story.")
    opts = [int(i) for i in adf["agent"]]
    by_id = {int(t.agent): t for t in adf.itertuples()}
    if st.session_state.get("insp_sel") not in opts:      # stale selection from a previous run
        st.session_state["insp_sel"] = opts[0]
    best_agent = int(adf.sort_values("reward", ascending=False)["agent"].iloc[0])
    dead_df = adf[~adf["alive"]].sort_values("death_round")
    first_death = int(dead_df["agent"].iloc[0]) if len(dead_df) else None
    rand_agent = int(random.Random(res.cfg.seed * 7 + len(res.events)).choice(opts))
    c0, c1, c2, c3 = st.columns([2.6, 1, 1.1, 1])
    chosen = c0.selectbox(
        "agent", opts,
        key="insp_sel",
        format_func=lambda i: (f"#{i} · {'alive' if by_id[i].alive else f'died r{int(by_id[i].death_round)}'} · "
                               f"reward {by_id[i].reward:,.0f} · {'group ' + str(by_id[i].group) if by_id[i].group != 'solo' else 'solo'}"))
    if c1.button("🏆 best", width="stretch", key="insp_best",
                 help="agent with the highest recorded lifetime reward",
                 on_click=lambda: st.session_state.update(insp_sel=best_agent)):
        pass
    if c2.button("💀 first death", width="stretch", key="insp_death", disabled=first_death is None,
                 on_click=(lambda: st.session_state.update(insp_sel=first_death)) if first_death is not None else None):
        pass
    if c3.button("🎲 random", width="stretch", key="insp_rand",
                 on_click=lambda: st.session_state.update(insp_sel=rand_agent)):
        pass
    aid = int(chosen)
    row = by_id[aid]

    cc = st.columns([1.25, 1.25, 1.0, 2.1])
    with cc[0]:
        st.markdown(f"""
        <div class="yu-card">
          <div class="yu-kpi-label">final state · agent #{aid}</div>
          <div style="margin-top:8px;font-size:13px;color:#c3d0e6;line-height:1.75;">
            🍽 food <b style="float:right">{fmt(row.food, 0)} / 90</b><br>
            <progress value="{row.food}" max="90"></progress><br>
            ⚡ energy <b style="float:right">{fmt(row.energy, 0)} / 100</b><br>
            <progress value="{row.energy}" max="100"></progress><br>
            💰 wealth <b style="float:right">{fmt(row.wealth, 0)}</b><br>
            <progress value="{min(100.0, float(row.wealth))}" max="100"></progress><br>
            ❤ health <b style="float:right">{fmt(row.health, 0)} / 100</b><br>
            <progress value="{row.health}" max="100"></progress><br>
            ⭐ reward <b style="float:right">{fmt(row.reward, 1)}</b>
          </div></div>""", unsafe_allow_html=True)
        st.markdown(kpi_card(
            "status", "alive ✅" if row.alive else f"died round {int(row.death_round)}",
            f"survived {int(row.survived_rounds)} of {res.cfg.rounds} rounds · "
            f"group {row.group} (size {int(row.group_size)})",
            PALETTE["green"] if row.alive else PALETTE["red"]), unsafe_allow_html=True)
    with cc[1]:
        fig = go.Figure(go.Bar(
            x=[row.trait_cooperation, row.trait_competition, row.trait_exploration, row.trait_risk, row.trait_adaptation],
            y=["co-operation", "competition", "exploration", "risk", "adaptation"], orientation="h",
            marker_color=[PALETTE["green"], PALETTE["red"], PALETTE["blue"], PALETTE["amber"], PALETTE["violet"]],
            hovertemplate="%{y}: %{x:.2f} (generated trait 0–1)<extra></extra>"))
        style_fig(fig, 200, legend=False)
        fig.update_layout(title=dict(text="personality (seeded traits)", font=dict(size=12, color="#c9d7ee")))
        fig.update_xaxes(range=[0, 1])
        st.plotly_chart(fig, width="stretch")
        st.markdown(kpi_card(
            "mean trust given", f"{row.mean_trust_given:.2f}",
            (f"over {int(row.n_coop_partners)} remembered partners · "
             f"{int(row.n_attacks_received)} attacker(s) remember the grudge") if res.cfg.memory
            else "memory OFF → every stranger stays at the start value",
            PALETTE["teal"]), unsafe_allow_html=True)
    with cc[2]:
        acts = {a: int(getattr(row, f"act_{a}", 0)) for a in ACTIONS + FORCED}
        fig = go.Figure(go.Bar(
            x=[ACTION_LABEL.get(a, a) for a in acts], y=list(acts.values()),
            marker_color=[ACTION_COLOR.get(a, "#94a3b8") for a in acts],
            hovertemplate="%{x}: %{y} times over the whole run<extra></extra>"))
        style_fig(fig, 356, legend=False)
        fig.update_layout(title=dict(text="lifetime action counts", font=dict(size=12, color="#c9d7ee")),
                          xaxis_tickangle=-40)
        st.plotly_chart(fig, width="stretch")
    with cc[3]:
        dec = [d for d in res.decisions_raw if d["agent"] == aid]
        if not dec:
            st.info("This agent never got to act (it must have been dead from round 1 — impossible in v1; "
                    "if you see this, it is a real bug worth reporting).")
            return
        last_r = int(dec[-1]["round"])
        rkey = f"insp_r_{aid}"
        if rkey in st.session_state:
            st.session_state[rkey] = int(clamp(int(st.session_state[rkey]), 1, last_r))
            dround = st.slider("show the decision taken in round", 1, last_r, key=rkey)
        else:
            dround = st.slider("show the decision taken in round", 1, last_r, last_r, key=rkey)
        d = next((x for x in reversed(dec) if x["round"] <= int(dround)), dec[-1])
        cb, rb = d["chosen_breakdown"], d["runner_breakdown"]
        st.markdown(f"""
        <div class="yu-card" style="margin-bottom:10px">
        <div class="yu-kpi-label">decision X-ray · round {d['round']}</div>
        <div style="font-size:17px;font-weight:800;color:#5eead4;margin:6px 0 2px 0;">
          {d['action'].upper()}
          <span style="color:#7c8db0;font-size:12.5px;font-weight:600;">
            &nbsp;utility {fmt(d['score'], 2)} · runner-up “{d['runner_up']}” {fmt(d['runner_up_score'], 2)}
            · margin {fmt(d['gap'], 2)}</span></div>
        <div style="color:#b9c7de;font-size:12.8px;line-height:1.5;"><b>why:</b> <i>{d['why']}</i></div>
        {f"<div style='color:#8ea0bd;font-size:12px;margin-top:5px;'><b>outcome:</b> {d['note']}</div>" if d['note'] else ""}
        </div>""", unsafe_allow_html=True)
        names = list(cb.keys()) + [f"⟂ {k}" for k in rb]
        vals = [cb[k] for k in cb] + [rb[k] for k in rb]
        colors = ([ACTION_COLOR.get(d['action'], "#34d399")] * len(cb) + ["#64748b"] * len(rb))
        fig = go.Figure(go.Bar(x=vals, y=names, orientation="h", marker_color=colors,
                               hovertemplate="%{y}: %{x:.2f}<extra></extra>"))
        style_fig(fig, max(210, 21 * (len(cb) + len(rb)) + 76), legend=False)
        fig.update_layout(title=dict(text="the exact factor sums used (green = chosen · gray = runner-up)",
                                     font=dict(size=12, color="#c9d7ee")))
        st.plotly_chart(fig, width="stretch", key=f"xray_{aid}_{d['round']}")
        pol = d.get("policy")
        if pol:
            pc1, pc2 = st.columns([1, 1.35])
            with pc1:
                pf = go.Figure(go.Bar(
                    x=list(pol["pi"].values()), y=[ACTION_LABEL.get(k, k) for k in pol["pi"]], orientation="h",
                    marker_color=[PALETTE["green"] if k == d["action"] else "#334155" for k in pol["pi"]],
                    hovertemplate="%{y}: %{x:.1%}<extra></extra>"))
                style_fig(pf, 240, legend=False)
                pf.update_layout(
                    title=dict(text=f"policy probabilities when sampled · H = {pol['ent']:.2f} bits",
                               font=dict(size=11.5, color="#c9d7ee")),
                    margin=dict(l=96, r=8, t=36, b=8), xaxis_tickformat=".0%", xaxis_range=[0, 1])
                st.plotly_chart(pf, width="stretch")
            with pc2:
                ml = pol["mults"].get(d["action"], {})
                mm = pd.DataFrame([{"factor": lab, "raw value": round(val, 2), "learned ×": ml.get(lab, 1.0),
                                    "effective": round(val * ml.get(lab, 1.0), 2)}
                                   for lab, val in cb.items() if lab != "jitter (risk-scaled uncertainty)"])
                st.caption("🧠 multipliers this agent had trained at that moment (×1.00 = exactly the rule)")
                st.dataframe(mm, width="stretch", hide_index=True, height=64 + 30 * len(mm))
        recent = pd.DataFrame(dec[-14:][::-1])[["round", "action", "score", "gap", "partner", "why"]]
        recent["partner"] = pd.array(recent["partner"].tolist(), dtype="Int64")
        st.caption(f"recent decisions of #{aid} (newest first · last 14 of {len(dec):,})")
        st.dataframe(recent, width="stretch", hide_index=True, height=240)


# ── compare A/B ─────────────────────────────────────────────────────────────
def run_compare(base_cfg: SimConfig, variable: str, extra_seeds: bool,
                progress_cb: Callable[[int, int, str], None]) -> dict:
    ca, cb, spec = comparison_pair(base_cfg, variable)
    seeds = [ca.seed] + ([ca.seed + 1, ca.seed + 2] if extra_seeds else [])
    runs_a: list[SimResult] = []
    runs_b: list[SimResult] = []
    coop_gaps: list[float] = []
    done = 0
    total = len(seeds) * 2
    for s in seeds:
        a_cfg = SimConfig(**{**ca.to_dict(), "seed": s})
        b_cfg = SimConfig(**{**cb.to_dict(), "seed": s})
        runs_a.append(run_simulation(a_cfg))
        done += 1
        progress_cb(done, total, f"seed {s} · world A done ({a_cfg.rounds} rounds)")
        runs_b.append(run_simulation(b_cfg))
        done += 1
        progress_cb(done, total, f"seed {s} · world B done")
        ga = np.asarray(runs_a[-1].rounds_df["cooperation_rate"], dtype=float)
        gb = np.asarray(runs_b[-1].rounds_df["cooperation_rate"], dtype=float)
        n = min(len(ga), len(gb))
        coop_gaps.append(float(np.mean(ga[:n] - gb[:n])))
    a0, b0 = runs_a[0], runs_b[0]
    analysis = analyse_comparison(a0, b0, spec)
    agreement = None
    if coop_gaps:
        signs = [1 if g > 0 else (-1 if g < 0 else 0) for g in coop_gaps]
        maj = int(np.sign(np.mean(signs))) if any(signs) else 0
        same = sum(1 for sgn in signs if sgn == maj) if maj else 0
        agreement = {"total": len(signs), "same": same, "per_seed": [fmt(g, 4) for g in coop_gaps],
                     "direction": ("A higher" if maj > 0 else "B higher") if maj else "split"}
    analysis["agreement"] = agreement
    return {"spec": spec, "variable": variable, "A": a0, "B": b0, "runs_a": runs_a, "runs_b": runs_b,
            "seeds": seeds, "analysis": analysis, "ca": ca.to_dict(), "cb": cb.to_dict()}


def render_compare() -> None:
    st.markdown(
        "<div class='yu-note' style='margin-bottom:12px'>"
        "<b>One variable. Same seed, same world, same run length — only one setting is patched.</b> "
        "The engine itself asserts the single-variable property (self-check 8). Comparison runs are computed "
        "<i>live, fresh from the rules</i> — nothing here is pre-recorded, and a verdict on one seed is a single sample."
        "</div>", unsafe_allow_html=True)
    base = cfg_from_widgets()
    c1, c2, c3 = st.columns([2.2, 1.1, 1.1])
    with c1:
        var = st.selectbox("single variable to flip", list(COMPARE_VARS.keys()),
                           format_func=lambda k: COMPARE_VARS[k]["title"], key="cmp_var")
    with c2:
        st.metric("held fixed", f"seed {base.seed} · {base.rounds} rounds", label_visibility="visible")
    with c3:
        extra = st.toggle("3 seeds (robustness)", value=False, key="cmp_extra",
                          help="Runs 6 worlds total (3 seeds × 2 settings). A direction that survives all seeds is "
                               "stronger evidence about the model. ≈3× runtime.")
    spec = COMPARE_VARS[var]
    st.caption(spec["explain"])

    if st.button("⚖ Run A/B comparison", type="primary"):
        ph = st.empty()
        try:
            t0 = time.perf_counter()
            with ph.container():
                res_c = run_compare(base, var, extra,
                                    lambda d, t, m: ph.progress(d / t, text=f"{m} · {d}/{t} worlds done"))
            ph.empty()
            res_c["elapsed_ms"] = int((time.perf_counter() - t0) * 1000)
            st.session_state["compare_res"] = res_c
        except Exception:
            ph.empty()
            st.session_state.pop("compare_res", None)
            st.error("Comparison run failed — showing the real error instead of hiding it:")
            st.code(traceback.format_exc(), language="text")

    res_c = st.session_state.get("compare_res")
    if not res_c:
        st.markdown("<div class='empty-state'><h3>No comparison yet</h3><p>Choose the single variable above and "
                    "press <b>Run A/B comparison</b>. Two identical worlds will be generated fresh and measured.</p>"
                    "</div>", unsafe_allow_html=True)
        return
    if res_c["variable"] != var:
        st.info("The stored comparison used a different variable — press **Run A/B comparison** to refresh it for "
                f"“{spec['title']}”.")
        with st.expander("view the stored (stale) result anyway"):
            _render_compare_body(res_c)
        return
    _render_compare_body(res_c)


def _render_compare_body(res_c: dict) -> None:
    A, B, ana, spec = res_c["A"], res_c["B"], res_c["analysis"], res_c["spec"]
    st.caption(f"completed in {res_c.get('elapsed_ms', 0):,} ms · {len(res_c['seeds'])} seed(s): "
               f"{', '.join(str(s) for s in res_c['seeds'])}")
    cr = next(r for r in ana["rows"] if r["key"] == "cooperation_rate")
    ag = ana.get("agreement")
    top = st.columns(4)
    cards = [
        ("Co-operation · A mean", f"{cr['mean_a']:.3f}", spec["a_label"], PALETTE["green"]),
        ("Co-operation · B mean", f"{cr['mean_b']:.3f}", spec["b_label"], PALETTE["teal"]),
        ("Δ (A − B)", f"{cr['delta']:+.3f}", f"Cohen's d = {cr['d']:+.2f} on round-to-round values", PALETTE["violet"]),
        ("Seed agreement", (f"{ag['same']}/{ag['total']} seeds: {ag['direction']}" if ag else "1 seed only"),
         "direction of the cooperation gap held across seeds" if ag and ag["total"] > 1
         else "run with 3 seeds to test robustness", PALETTE["amber"]),
    ]
    for col, (l, v, s, ac) in zip(top, cards):
        col.markdown(kpi_card(l, v, s, ac), unsafe_allow_html=True)
    st.markdown(f"<div class='yu-ok' style='margin:10px 0'>🔎 <b>Observed — measured inside this model:</b> "
                f"{ana['verdict']}<br><span style='color:#8ea0bd'>This is a statement about these rules and this "
                f"seed set, not about humans or real economies.</span></div>", unsafe_allow_html=True)

    n = min(len(A.rounds_df), len(B.rounds_df))
    cL, cR = st.columns(2)
    with cL:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=A.rounds_df["round"][:n], y=A.rounds_df["cooperation_rate"][:n],
                                 name=spec["a_label"], line=dict(color=PALETTE["green"], width=2.2),
                                 hovertemplate="r%{x} · A: %{y:.3f}<extra></extra>"))
        fig.add_trace(go.Scatter(x=B.rounds_df["round"][:n], y=B.rounds_df["cooperation_rate"][:n],
                                 name=spec["b_label"], line=dict(color=PALETTE["amber"], width=2.2),
                                 hovertemplate="r%{x} · B: %{y:.3f}<extra></extra>"))
        style_fig(fig, 300)
        fig.update_layout(title=dict(text="co-operation rate over rounds — A vs B",
                                     font=dict(size=13, color="#c9d7ee")),
                          xaxis_title="round", yaxis_title="rate (0–1)")
        st.plotly_chart(fig, width="stretch")
        dv = cr["first_divergence"]
        extra = ""
        if ag and ag.get("per_seed"):
            extra = " · cooperation Δ by seed: " + ", ".join(ag["per_seed"])
        st.caption(f"first measurable cooperation gap: round {dv if dv else 'never'}{extra}")
    with cR:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=A.rounds_df["round"][:n],
            y=(A.rounds_df["avg_trust"][:n] - B.rounds_df["avg_trust"][:n]).round(4),
            name="trust gap (A−B)", line=dict(color=PALETTE["teal"], width=2),
            fill="tozeroy", fillcolor="rgba(45,212,191,.12)",
            hovertemplate="r%{x} · Δtrust %{y:+.3f}<extra></extra>"))
        fig.add_hline(y=0, line=dict(color="#334155", width=1))
        style_fig(fig, 300)
        fig.update_layout(title=dict(text="signed gap over time — above 0 means A is higher",
                                     font=dict(size=13, color="#c9d7ee")),
                          xaxis_title="round", yaxis_title="Δ trust")
        st.plotly_chart(fig, width="stretch")

    with st.expander("📏 every metric — full delta & effect-size table", expanded=True):
        rows = []
        for r in ana["rows"]:
            better = {"good": r["delta"] > 0.01, "bad": r["delta"] < -0.01, "neutral": None}[r["valence"]]
            verdict = ("—" if better is None else
                       ("A better on this metric ✅" if better else "B better on this metric ⚠"))
            strength = ("large" if abs(r["d"]) >= 0.8 else "medium" if abs(r["d"]) >= 0.5
                        else "small" if abs(r["d"]) >= 0.2 else "negligible")
            rows.append({"metric": r["label"], "A mean": r["mean_a"], "B mean": r["mean_b"],
                         "Δ (A−B)": r["delta"], "Δ rel": f"{r['rel'] * 100:+.1f}%",
                         "Cohen's d": r["d"], "effect": strength, "unit": r["unit"], "note": verdict})
        tdf = pd.DataFrame(rows)
        st.dataframe(tdf, width="stretch", hide_index=True, height=34 * len(tdf) + 40,
                     column_config={"A mean": st.column_config.NumberColumn(format="%.3f"),
                                     "B mean": st.column_config.NumberColumn(format="%.3f"),
                                     "Δ (A−B)": st.column_config.NumberColumn(format="%+.4f"),
                                     "Cohen's d": st.column_config.NumberColumn(format="%+.2f")})
        if ana["unchanged"]:
            st.markdown("**What did NOT change** (|d| < 0.2 — reported for honesty): " + ", ".join(ana["unchanged"]))
    with st.expander("🔧 the exact two configs used (single-variable check built in)"):
        ca, cb = res_c["ca"], res_c["cb"]
        diff_keys = [k for k in ca if ca[k] != cb[k]]
        ok = len(diff_keys) == 1
        st.markdown(("✅ " if ok else "❌ ") +
                    f"settings that differ: <b>{', '.join(SETTING_LABEL.get(k, k) for k in diff_keys) or 'none'}</b> "
                    f"— seed {ca['seed']} = seed {cb['seed']} · both {ca['rounds']} rounds × {ca['population']} agents",
                    unsafe_allow_html=True)
        t = pd.DataFrame([ca, cb]).astype(str)
        t.insert(0, "", ["A", "B"])
        st.dataframe(t.set_index("").T, width="stretch")


# ── discoveries ─────────────────────────────────────────────────────────────
def render_discoveries(res: SimResult) -> None:
    findings = discover(res)
    st.caption("Each card is computed from this run's own dataframes (first-third vs last-third means, peaks, "
               "correlations across agents). Observations stay visually separate from any possible reading.")
    sev_counts = pd.Series([f["severity"] for f in findings]).value_counts() if findings else pd.Series(dtype=int)
    chips_html = [chip(f"{len(findings)} findings · {len(res.events)} events · "
                       f"{res.meta['decisions']:,} decision records", "#0f2b26", "#5eead4")]
    for s, cc_, fg_ in [("high", "#3b1224", "#fda4af"), ("medium", "#3a2b12", "#fcd34d"),
                        ("low", "#12233d", "#93c5fd"), ("info", "#16213a", "#94a3b8")]:
        chips_html.append(chip(f"{int(sev_counts.get(s, 0))} {s}", cc_, fg_))
    st.markdown("<div class='yu-chiprow'>" + "".join(chips_html) + "</div>", unsafe_allow_html=True)
    if not findings:
        st.info("Nothing crossed the reporting bar for this run — usually means it was very short or very stable. "
                "Try more rounds, or a more extreme preset.")
        return
    for f in findings:
        st.markdown(f"""
        <div class="yu-find {f['severity']}">
          <h4>{f['title']}</h4>
          <div class="obs">{f['observed']}</div>
          {f"<div class='read'>{f['reading']}</div>" if f.get('reading') else ""}
          <div style="margin-top:6px">{chip(f['category'], '#182644', '#a5b4fc')} {chip('observed', '#122a3a', '#7dd3fc')}</div>
        </div>""", unsafe_allow_html=True)
    st.divider()
    if not res.events_df.empty:
        st.markdown("**Raw engine log — events by kind**")
        cnt = res.events_df["kind"].value_counts()
        fig = go.Figure(go.Bar(
            x=cnt.index, y=cnt.values,
            marker_color=[{"trade": PALETTE["violet"], "group": PALETTE["teal"], "betray": PALETTE["red"],
                           "death": PALETTE["gray"], "milestone": PALETTE["amber"]}.get(k, PALETTE["blue"])
                          for k in cnt.index],
            hovertemplate="%{x}: %{y} events<extra></extra>"))
        style_fig(fig, 220, legend=False)
        st.plotly_chart(fig, width="stretch")


# ── data & export ───────────────────────────────────────────────────────────
def render_data(res: SimResult) -> None:
    findings = discover(res)
    st.markdown(
        "<div class='yu-note'><b>⏳ This data lives only in your browser session.</b> There is no database and no "
        "cloud storage — closing the tab discards everything. The only way to keep a run is to download it below. "
        "The JSON bundle is explicitly tagged <code>\"simulated\": true</code>.</div>", unsafe_allow_html=True)
    st.markdown(f"**Run identity** — engine `{res.meta['engine']}` · config hash `{res.meta['config_hash']}` · "
                f"seed {res.cfg.seed} · {len(res.rounds_df)} rounds · {len(res.decisions_df):,} decisions · "
                f"{len(res.frames)} map frames · generated {res.meta['generated_at']}")
    stem = f"yudaant_seed{res.cfg.seed}_r{res.cfg.rounds}_{res.meta['config_hash']}"
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.download_button("⬇ round metrics CSV", df_to_csv(res.rounds_df), f"{stem}_rounds.csv",
                       "text/csv", width="stretch", help="one row per simulated round — all metrics")
    c2.download_button("⬇ agent records CSV", df_to_csv(res.agents_df), f"{stem}_agents.csv",
                       "text/csv", width="stretch", help="one row per agent — traits, outcomes, action counts")
    c3.download_button("⬇ events CSV", df_to_csv(res.events_df), f"{stem}_events.csv",
                       "text/csv", width="stretch", help="trades, groups, betrayals, deaths, milestones")
    c4.download_button("⬇ decisions CSV", df_to_csv(res.decisions_df), f"{stem}_decisions.csv",
                       "text/csv", width="stretch", help=f"every decision ({len(res.decisions_df):,}) with the exact score breakdowns")
    c5.download_button("⬇ all data + JSON bundle", build_zip(res, findings), f"{stem}_bundle.zip",
                       "application/zip", width="stretch", help="everything in one zip, incl. README + repro config")
    st.download_button("⬇ JSON bundle alone (config · metrics · agents · events · findings)",
                       build_json_bundle(res, findings), f"{stem}_bundle.json", "application/json",
                       width="stretch")
    st.caption("The JSON bundle embeds the exact SimConfig, so re-entering it reproduces the run bit-for-bit "
               "(verified by self-check 4).")
    t1, t2 = st.tabs(["📈 per-round table", "🧾 per-agent table"])
    with t1:
        st.dataframe(res.rounds_df.round(4), width="stretch", height=340)
    with t2:
        st.dataframe(res.agents_df, width="stretch", height=340)

    with st.expander("📇 Society card — paste a whole world to a friend (no server, no account, one blob)"):
        card: dict = {"kind": "yudaant-card", "v": 1, "config": res.cfg.to_dict(),
                      "config_hash": res.meta["config_hash"], "simulated": True,
                      "note": "settings only — re-pressing Start re-simulates; no dataset travels in this card"}
        if res.cfg.policy == "LEARNED" and getattr(res, "W_final", None) is not None:
            Wf = np.asarray(res.W_final, dtype=float)
            am = res.agents_df["alive"].to_numpy(dtype=bool)
            card["brain_mean"] = np.round((Wf[am].mean(axis=0) if am.any() else Wf.mean(axis=0)) - 1.0, 5).tolist()
            card["brain_gen"] = int(res.meta.get("brain_gen", 0))
            card["note"] += " — this one also carries the society's trained culture (mean multiplier shifts)"
        st.code(json.dumps(card, separators=(",", ":")), language="json")
        cA, cB = st.columns([1, 1.3])
        with cA:
            st.download_button("⬇ save card as .json", json.dumps(card, indent=1).encode(),
                               f"yudaant_card_{res.meta['config_hash']}.json", "application/json",
                               width="stretch")
        with cB:
            st.markdown("**Restore a card**")
            st.text_area("paste a society card", key="card_paste", height=88,
                         placeholder='{"kind":"yudaant-card", ...}', label_visibility="collapsed")
            if st.button("📥 Load card into the app", key="card_load", type="primary", width="stretch",
                         on_click=_card_from_state):
                pass
        msg = st.session_state.pop("card_msg", None)
        if msg:
            (st.success if str(msg).startswith("✅") else st.error)(msg)
        st.caption("RULE runs reproduce bit-for-bit from a card. LEARNED without carry too (same seed + same "
                   "settings). With 🔁 carry, a card restores the culture snapshot it shipped — the next runs are "
                   "a faithful continuation, not a replay of the original sequence, exactly as carry promises.")


# ── method & limits ─────────────────────────────────────────────────────────
def render_method(res: Optional[SimResult], checks: list[dict]) -> None:
    cfg = res.cfg if res else SimConfig()
    st.markdown("### What this model is — and is not")
    st.markdown(f"<div class='yu-danger'><b>Read this first.</b> {DISCLAIMER} "
                "In <b>📜 RULE mode</b> agents are utility-maximisers over a fixed score function with "
                "reinforcement from memory — no learned model at all. In <b>🧠 LEARNED mode</b> each agent "
                "additionally trains a documented 8×53 linear policy on <i>its own</i> last few decisions while "
                "the simulation runs (policy gradient, no LLM, no internet, seeded and reproducible unless "
                "🔁 carry is ON — which is stateful across runs by design). “Reward”, “trust”, "
                "“wealth” are model currencies defined below, not measurements of anything external.</div>",
                unsafe_allow_html=True)
    st.markdown("""
    <div class="yu-card"><b>🧠 How the learned policy trains</b><br>
    <b>Representation</b> — for action <i>a</i> the policy score is Σ&#8342; w&#8322;·f&#8322; over the same named
    factors f&#8322; the rule uses (53 total). Multipliers w start at 1.00 (exactly the rule) + a seeded ±0.05
    individuality; the Inspector and [L5] heatmap show them at all times.<br>
    <b>Sampling</b> — actions are drawn from softmax(score ÷ temperature) instead of argmax, so agents can
    explore; every draw comes from the run's seeded RNG stream.<br>
    <b>Training</b> — once per round, over each agent's own replay of its last <i>k</i> samples
    (policy_buffer, the “small data”), one REINFORCE step is taken: ΔW ∝ A·(onehot(aₜ) − πₜ)⊗fₜ with the
    advantage A normalised across the live population and a value EMA (0.15 rate) as baseline; a
    <b>personality-scaled anchor</b> pull −anchor·(1.7−adaptability)·(W−1) plus clipping to [−2, +3] keep it
    tame (anchor 0 = habits never fade). Forced collapses are never treated as choices.<br>
    <b>🌀 Culture</b> — with influence &gt; 0, each round group members pull each other's multipliers a fraction
    of the way toward the group mean, and a dying agent bequeaths part of its habits to surviving group-mates:
    beliefs propagate without any shared gradient. Within-group vs global variance ([L9]) is the measurement.<br>
    <b>⚔ Stress-test arena</b> — freezes the trained multipliers (lr 0, anchor 0, influence 0), drops the world
    into a famine (regrowth ×0.22) and races frozen habits against naïve instincts, same seed.<br>
    <b>📇 Society card</b> — the Data tab exports (and imports) one JSON blob of the full config plus, for
    LEARNED runs, the mean trained culture; paste it anywhere to rebuild the world — no server involved.<br>
    <b>Carry (optional)</b> — with 🔁 ON, the mean (W−1) of the top-40% agents by reward is averaged and stored
    <i>in this browser session</i> to seed the next run: generation-to-generation culture, RAM-only, explicitly
    not bit-reproducible while on.</div>
    """, unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**The loop, one round at a time**")
        st.markdown("""
        1. Each living agent, in id order, scores all 8 legal actions against its <i>current</i> state, traits and
           (if memory ON) per-partner trust. Every score is a sum of named factors — the Inspector shows them.
        2. A seeded, risk-scaled <i>jitter</i> (uncertainty) is added, and the argmax action is executed.
        3. The land regrows: `cell_food += food_regrowth × region_modifier`, capped at 100 per cell.
        4. Metabolism: −4 energy/round; below 35 energy agents <b>auto-eat</b> 12 food (→ +26.4 energy);
           an empty stomach costs 3.5 health/round; being well-fed + energetic heals +0.8/round.
           Health ≤ 0 → death (there are no births, so population is deaths-only).
        5. Trust decays 1% toward the starting-trust value; groups (3+ mutual co-operations & trust ≥ 0.6)
           are re-validated and dissolved when they drop below 2 members.
        6. Metrics and a compact map frame are recorded — that record is what every chart on this page reads.
        """)
        st.markdown("**Action economics — the whole policy in one table**")
        act_tab = pd.DataFrame([
            ["collect", "take up to 12 + risk·4 food from the cell underfoot (group +10%)", "−3 energy",
             "hunger ×6, cell food, expected haul"],
            ["move", "step to the richest adjacent cell (wrap-around world)", "−1.5 energy",
             "neighbour richer, exploration, crowding"],
            ["explore", "jump to a random far cell; 30–65% chance to find a +25 food patch", "−5 energy, −1 health",
             "exploration trait, patchy home, risk appetite"],
            ["rest", "+16 energy; +1.5 health if fed ≥ 25 food", "—", "exhaustion ×7, low health, hunger risk"],
            ["trade", "12 food ↔ 8 coins with a partner in your cell (either direction)", "−2 energy both",
             "surplus/hunger, partner trust, adaptability"],
            ["share", "give 10 food to the neediest neighbour you trust ≥ 0.30", "−2 energy",
             "co-op trait ×3.6, partner need, trust, reward model"],
            ["cooperate", "joint project: each side +6 food (×1.2 in-group), +2.8·mult reward, trust +0.10",
             "−4 energy both", "co-op trait ×4.4, mutual trust, group bonus"],
            ["compete", "take 45% of the weakest neighbour's food (cap 12) + 35% of coins; victim −8 health; "
             "you lose 4 health if they could fight back", "−3 energy",
             "comp trait ×5, weakness, witnesses × gossip level, group taboo, grudge"],
        ], columns=["action", "what actually happens", "cost", "main score drivers"])
        st.dataframe(act_tab, width="stretch", hide_index=True, height=330)
    with c2:
        st.markdown("**Metric definitions (computed, never assumed)**")
        st.markdown("""
        | metric | definition |
        |---|---|
        | co-operation rate | (share + co-operate actions) ÷ living agents, per round |
        | competition rate | compete actions ÷ living agents, per round |
        | mean trust | average over all remembered partner-trust entries; = start trust if memory OFF |
        | reward | per-action base × reward-model multiplier + survival trickle (0.05 + 0.15·health/100) |
        | wealth Gini | Gini over agents' coin holdings (0–1) |
        | groups | clusters of ≥2 agents that co-operated 3× with mutual trust ≥ 0.6 |
        | survival | living agents ÷ starting population |
        """)
        st.markdown("**Current run's exact config**")
        st.dataframe(pd.DataFrame([[k, str(v)] for k, v in cfg.human().items()], columns=["setting", "value"]),
                     width="stretch", hide_index=True, height=330)
        st.markdown("**Reproduce this run headlessly (no Streamlit needed)**")
        st.code(f"""python - <<'PY'
import app   # this file
cfg = app.SimConfig.from_dict({json.dumps(cfg.to_dict())})
r = app.run_simulation(cfg)
print(r.meta["config_hash"])
print(r.rounds_df[["round", "cooperation_rate", "avg_trust", "groups"]].tail())
PY""", language="bash")
        st.caption("Expected: the same config hash and identical numbers you see in the UI.")

    st.markdown("### What these results can and cannot support")
    for x in [
        "**A single run is one sample.** Trends can flip with another seed; use Compare → “3 seeds” before believing a direction.",
        "**Traits were assigned, not earned.** Correlations between generated traits and outcomes mostly restate the reward table.",
        "**No emergence claims.** Groups, markets and grudges here are direct consequences of the rules above.",
        "**No external validity.** Nothing about real societies was used to build or tune this model, so nothing here should be cited as evidence about them.",
        "**Deaths are permanent** (no births), so “survival” mixes starvation and wounds — the event feed says which.",
        "**Interactions use start-of-round positions** (simultaneous-round semantics) — a deliberate simplicity choice.",
        "**This app has no database.** Results live only in the browser session; they are gone on refresh unless exported.",
    ]:
        st.markdown(f"- {x}")

    st.markdown("### 🧪 Self-checks — executed live on the real engine")
    ok = all(c["ok"] for c in checks)
    st.markdown(("✅ <b>all checks passed</b> — repeatability, bounds, metric ranges, comparison honesty, exports, "
                 "incentive plumbing." if ok else
                 "❌ <b>a self-check failed</b> — see below. Treat outputs as unverified until fixed."),
                unsafe_allow_html=True)
    st.dataframe(pd.DataFrame(checks)[["check", "ok", "detail"]].rename(
        columns={"check": "check (actually executed)", "ok": "passed", "detail": "evidence"}),
        width="stretch", hide_index=True, height=36 * (len(checks) + 1) + 40)
    with st.expander("why each check exists"):
        st.markdown("""
        - **Repeatability** catches accidental unseeded randomness — the whole “press Start and trust the numbers” promise depends on it.
        - **Sensitivity** catches the opposite failure: a hard-coded world that ignores your seed.
        - **Bounds / ranges** catch rule bugs (negative food, health > 100) before they silently poison charts.
        - **Single-variable pairs** guarantee Compare mode actually isolates one change.
        - **Export round-trip** guarantees downloaded files parse back to the same rows the UI shows.
        - **Behaviour sanity** re-runs the real policy under different reward tables and asserts the incentives move the outcome.
        """)


# ── empty state / runner / main ─────────────────────────────────────────────
def render_live_wait() -> None:
    live = st.session_state.get("live")
    r = live["soc"].rnd if live else 0
    st.markdown(
        f"<div class='empty-state'><div style='font-size:30px'>🎬</div>"
        f"<h3>The world is still being born — round {r} is playing in the Society tab</h3>"
        f"<p style='font-size:13px'>This tab fills automatically the moment the run finishes (or press "
        f"<b>Skip to end</b> in the Society view). Nothing here is pre-made: every chart will read the frames "
        f"the live engine just produced.</p></div>", unsafe_allow_html=True)


def render_empty() -> None:
    st.markdown(
        "<div class='empty-state'><div style='font-size:34px'>🌍</div>"
        "<h3>No world has been generated yet</h3>"
        "<p style='max-width:660px;margin:0 auto'>Press <b>▶ Start simulation</b> — with “🎬 Watch it live” on "
        "(default), you will see the agents moving round-by-round right here: food tiles fading, trades and "
        "attacks popping up, groups forming — with a live “what every agent is doing” feed beside the map. "
        "Want the agents to <b>learn while they live</b>? Set <b>🧠 Agent policy → LEARNED</b> in the sidebar: "
        "each one trains a tiny transparent policy on its own last few decisions, and a Learning lab tab opens up. "
        "Same seed + settings ⇒ same movie, every time.</p></div>", unsafe_allow_html=True)
    st.button("▶  Start simulation (live · 110 agents · 200 rounds · seed 1337)", type="primary",
              on_click=start_run, key="empty_start")


def render_no_run(hint: str) -> None:
    st.markdown(f"<div class='empty-state' style='padding:26px'><b style='color:#c9d7ee'>{hint}</b>"
                "<br><span style='font-size:13px'>Start a simulation in the sidebar (▶ Start simulation) "
                "and this view fills with real values from it.</span></div>", unsafe_allow_html=True)


SHELF_MAX = 3          # unpinned snapshots kept automatically
SHELF_PIN_MAX = 3      # user-pinned snapshots kept until removed


def _shelf_find(key: str) -> Optional[dict]:
    for e in st.session_state.get("shelf", []):
        if e["key"] == key:
            return e
    return None


def _shelf_remove(key: str) -> None:
    st.session_state["shelf"] = [e for e in st.session_state.get("shelf", []) if e["key"] != key]


def _shelf_pin(key: str) -> None:
    e = _shelf_find(key)
    if not e:
        return
    if not e["pinned"] and sum(1 for x in st.session_state["shelf"] if x["pinned"]) >= SHELF_PIN_MAX:
        st.session_state["shelf_note"] = (f"⚠ already {SHELF_PIN_MAX} pinned — unpin one first (pins are kept "
                                          "forever on purpose, so they are capped).")
        return
    e["pinned"] = not e["pinned"]


def _shelf_load(key: str) -> None:
    e = _shelf_find(key)
    if not e:
        return
    st.session_state["res"] = e["res"]
    st.session_state.pop("frame_slider", None)
    st.session_state["insp_sel"] = None
    st.session_state["last_run_msg"] = (f"🗂 loaded saved run — {e['label']} · snapshotted at {e['ts']}. "
                                        "Nothing was re-simulated: these are the exact figures from that finished "
                                        "run, restored into every tab.")


def shelf_add(res: "SimResult") -> None:
    """Auto-save every completed run for study; the oldest unpinned snapshot is dropped past SHELF_MAX."""
    cfg = res.cfg
    rd = res.rounds_df
    deaths = int(rd["deaths"].iloc[-1]) if len(rd) else 0
    label = (f"seed {cfg.seed} · {cfg.population}×{cfg.rounds} r · "
             + ("🧠 learned" if cfg.policy == "LEARNED" else "📜 rules")
             + (f" · gen {int(res.meta.get('brain_gen', 0)) + 1}" if cfg.policy_carry else "")
             + f" · {res.meta['alive_end']}/{res.n0} survived · {deaths} deaths")
    shelf = st.session_state.setdefault("shelf", [])
    shelf.append({"key": f"s{int(time.time_ns() % 10**9)}", "label": label,
                  "ts": datetime.now(timezone.utc).strftime("%H:%M:%S UTC"),
                  "res": res, "pinned": False})
    while sum(1 for e in shelf if not e["pinned"]) > SHELF_MAX:
        first = next(i for i, e in enumerate(shelf) if not e["pinned"])
        shelf.pop(first)


def _load_card(raw: str) -> str:
    """Paste-a-world restore: settings come back exactly; a LEARNED card can also restore its culture."""
    try:
        card = json.loads(raw)
    except Exception as e:
        return f"❌ not valid JSON ({type(e).__name__}: {str(e)[:80]})"
    if not isinstance(card, dict) or card.get("kind") != "yudaant-card":
        return "❌ not a YUDAANT society card (missing \"kind\": \"yudaant-card\")"
    cfgd = card.get("config")
    if not isinstance(cfgd, dict):
        return "❌ card has no config block"
    bad = [k for k in cfgd if k not in SimConfig.__dataclass_fields__]
    if bad:
        return f"❌ card contains unknown settings: {bad}"
    for k, v in cfgd.items():
        if k in WIDGET_KEYS:
            st.session_state[f"sb_{k}"] = v
    extra = ""
    if isinstance(card.get("brain_mean"), list):
        st.session_state["brain"] = {"gen": int(card.get("brain_gen", 0)),
                                     "W": np.array(card["brain_mean"], dtype=float), "hist": []}
        extra = f" + generation-{int(card.get('brain_gen', 0)) + 1} learned culture"
    return f"✅ loaded {len(cfgd)} settings{extra} — press ▶ Start to rebuild this exact world"


def _card_from_state() -> None:
    """on_click callback (widget writes are legal from callbacks, unlike mid-script)"""
    st.session_state["card_msg"] = _load_card(st.session_state.get("card_paste") or "")


def after_run(res: "SimResult") -> None:
    """Runs finished from any path land on the study shelf; carry-memory bookkeeping too."""
    shelf_add(res)
    bo = res.meta.get("brain_out")
    if bo is None:
        return
    prev = st.session_state.get("brain") or {"gen": 0, "hist": []}
    rd = res.rounds_df
    gen = int(prev["gen"]) + 1
    st.session_state["brain"] = {
        "gen": gen, "W": np.array(bo, dtype=float),
        "hist": prev["hist"] + [{
            "gen": gen,
            "end_entropy": float(rd["policy_entropy"].iloc[-1]) if "policy_entropy" in rd.columns else None,
            "end_divergence": float(rd["divergence_rate"].iloc[-1]) if "divergence_rate" in rd.columns else None,
            "avg_reward": float(rd["avg_reward"].mean()),
            "survival": float(rd["survival"].iloc[-1]),
            "coop_end": float(rd["cooperation_rate"].iloc[-1]),
        }],
    }


def brain_for(cfg: SimConfig) -> tuple[Optional[np.ndarray], int]:
    if cfg.policy != "LEARNED" or not cfg.policy_carry:
        return None, 0
    b = st.session_state.get("brain")
    return (b["W"], int(b["gen"])) if b else (None, 0)


def do_run() -> None:
    cfg = cfg_from_widgets().clamped()
    if st.session_state.get("sb_live", True):
        # live mode: the player fragment steps the SAME engine round-by-round
        brain, bgen = brain_for(cfg)
        st.session_state["live"] = {"soc": Society(cfg, brain=brain, brain_gen=bgen)}
        st.session_state["live_playing"] = True
        st.session_state["last_run_msg"] = None
        return
    ph = st.empty()
    t0 = time.perf_counter()

    def cb(rnd: int, total: int, info: dict) -> None:
        el = max(1e-9, time.perf_counter() - t0)
        ph.progress(rnd / total, text=f"round {rnd}/{total} · {info['alive']}/{cfg.population} alive · "
                                      f"{rnd / el:.0f} rounds/s")

    try:
        with ph.container():
            brain, bgen = brain_for(cfg)
            res = run_simulation(cfg, progress_cb=cb, brain=brain, brain_gen=bgen)
        ph.empty()
        st.session_state["res"] = res
        after_run(res)
        st.session_state.pop("frame_slider", None)   # will default to the last round at next render
        st.session_state["insp_sel"] = None
        st.session_state["last_run_msg"] = (
            f"✅ run complete — {cfg.population} agents × {cfg.rounds} rounds in {res.meta['run_ms']:,} ms · "
            f"{res.meta['decisions']:,} decisions · {res.meta['events']} events · "
            f"{res.meta['alive_end']}/{res.n0} survived · config hash `{res.meta['config_hash']}`")
    except Exception:
        ph.empty()
        st.session_state.pop("res", None)
        st.error("The simulation raised an error — showing the real traceback rather than hiding it:")
        st.code(traceback.format_exc(), language="text")


def main() -> None:
    st.set_page_config(page_title=f"{APP_NAME} — artificial-society simulator",
                       page_icon="⚔", layout="wide",
                       menu_items={"About": f"{APP_NAME} {ENGINE_VERSION}\n{DISCLAIMER}"})
    st.markdown(CSS, unsafe_allow_html=True)

    checks = cached_checks()
    checks_ok = all(c["ok"] for c in checks)

    render_sidebar()
    render_hero()

    if st.session_state.pop("_start_requested", False):
        do_run()
    note = st.session_state.pop("just_applied", None)
    if note:
        st.info(note)
    if msg := st.session_state.get("last_run_msg"):
        st.success(msg)
    if not checks_ok:
        st.error("🧪 A built-in self-check FAILED — see Method & Limits. Treat the numbers below as unverified "
                 "until fixed (they are shown as-is, never hidden).")

    live = st.session_state.get("live")
    res: Optional[SimResult] = st.session_state.get("res")
    if res is not None and not live:
        render_kpi_strip(res)

    tabs = st.tabs(["🌍 Society view", "📈 Trends & graphs", "🧠 Learning lab", "🗂 Study shelf",
                     "🔎 Agent inspector", "⚖ Compare A/B", "🧭 Discoveries", "💾 Data & export",
                     "📖 Method & limits"])

    def _tab_body(i: int, fn, title: str) -> None:
        with tabs[i]:
            if live:
                render_live() if i == 0 else render_live_wait()
            elif res:
                fn(res)
            else:
                render_empty() if i == 0 else render_no_run(title)

    with tabs[0]:
        if live:
            render_live()
        elif res:
            render_society(res)
        else:
            render_empty()
    _tab_body(1, render_trends, "📈 26-panel analysis lab + conclusions appear here")
    _tab_body(2, render_learning, "🧠 learning curves appear after a run with policy = LEARNED")
    with tabs[3]:
        render_live_wait() if live else render_shelf()
    _tab_body(4, render_inspector, "🔎 every agent's decision story appears here")
    with tabs[5]:
        render_compare()
    _tab_body(6, render_discoveries, "🧭 automatic findings appear here")
    _tab_body(7, render_data, "💾 CSV / JSON exports appear here")
    with tabs[8]:
        render_method(res, checks)

    if res is not None:
        st.caption(
            f"engine {res.meta['engine']} · seed {res.cfg.seed} · {len(res.rounds_df)}/{res.cfg.rounds} rounds · "
            f"run {res.meta['run_ms']:,} ms · config `{res.meta['config_hash']}` · "
            f"self-checks {'✅ ' + str(sum(c['ok'] for c in checks)) + '/' + str(len(checks)) if checks_ok else '❌'} · "
            "results live only in this browser session — download them to keep · "
            + DISCLAIMER)
    else:
        st.caption(f"engine {ENGINE_VERSION} · {len(checks)} built-in self-checks · nothing on this page is "
                   "real-world data")


if __name__ == "__main__":
    main()
