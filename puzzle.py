"""
puzzle.py — GloVe neighbour generation (target-aware MMR), BFS, daily puzzle.

Neighbour generation strategy
──────────────────────────────
1. Base pool  : top-80 GloVe neighbours of the current word.
2. Bridge pool (only when sim(current, target) >= 0.45): most_similar(positive=
   [current, target]) finds words near the midpoint of both vectors — the
   stepping-stone words.  For "arm"→"hand" (sim≈0.55): shoulder, elbow, wrist…
   For "dragon"→"hand" (sim≈0.18): threshold not met, no bridges, no shortcuts.
3. No target injection: the target only appears when it is a genuine GloVe
   neighbour of the current word (naturally in the base top-80).  This keeps
   the game challenging — you have to actually navigate close to the target.
4. MMR selection (lambda_=0.4): picks 10 words balancing relevance (40%) and
   diversity (60%) so players never get stuck in a tight synonym cluster.

The neighbour graph is target-aware, so the BFS optimal path uses the same
graph the player navigates — ensuring the shown "optimal" path is achievable.
"""

from __future__ import annotations

import random
from collections import deque
from datetime import date
from typing import Optional

# ── GloVe model (lazy-loaded) ─────────────────────────────────────────────────
_model = None


def load_word_vectors(status_cb=None) -> None:
    """
    Load glove-wiki-gigaword-50 (~66 MB, cached after first download).
    Pass status_cb(msg: str) to show progress on a loading screen.
    """
    global _model
    if _model is not None:
        return
    import gensim.downloader as api
    if status_cb:
        status_cb('Loading word vectors…  (first run downloads ~66 MB)')
    _model = api.load('glove-wiki-gigaword-50')
    if status_cb:
        status_cb('Word vectors ready.')


def _get_model():
    if _model is None:
        load_word_vectors()
    return _model


# ── Word list ─────────────────────────────────────────────────────────────────
_RAW_WORD_LIST: list[str] = [
    'fire', 'ocean', 'mountain', 'music', 'dog', 'tree', 'book',
    'river', 'city', 'dance', 'dream', 'cloud', 'flower', 'bread',
    'gold', 'light', 'shadow', 'war', 'peace', 'love', 'fear', 'hope',
    'winter', 'summer', 'night', 'moon', 'star', 'wind', 'rain',
    'desert', 'forest', 'king', 'sword', 'ship', 'coin', 'mirror',
    'door', 'voice', 'song', 'bridge', 'garden', 'storm',
    'snake', 'eagle', 'lion', 'wolf', 'rose', 'stone', 'ice',
    'apple', 'horse', 'castle', 'crown',
    'water', 'thunder', 'leaf', 'seed', 'root',
    'cave', 'lake', 'island', 'valley', 'hill', 'cliff',
    'flame', 'smoke', 'crystal', 'iron', 'silver',
    'cat', 'bird', 'fish', 'bear', 'fox', 'rabbit', 'deer',
    'hammer', 'shield', 'arrow', 'sky', 'sea', 'path', 'earth',
    'blade', 'tower', 'frost', 'dust', 'bone',
    'heart', 'mind', 'soul', 'hand', 'eye', 'blood',
]
_RAW_WORD_LIST = list(dict.fromkeys(_RAW_WORD_LIST))


def get_word_list() -> list[str]:
    """Return the word list filtered to words present in the GloVe vocabulary."""
    model = _get_model()
    return [w for w in _RAW_WORD_LIST if w.lower() in model]


# ── Neighbour cache  (keyed by (word, target) pair) ──────────────────────────
_cache: dict[tuple[str, str], list[tuple[str, str]]] = {}


def get_daily_seed() -> int:
    return int(date.today().strftime('%Y%m%d'))


# ── Similarity helpers ────────────────────────────────────────────────────────

def _safe_sim(model, w1: str, w2: str) -> float:
    try:
        return float(model.similarity(w1, w2))
    except KeyError:
        return 0.0


def _score_label(score: float) -> str:
    if score >= 0.65:
        return 'close'
    elif score >= 0.45:
        return 'related'
    else:
        return 'distant'


# ── Public neighbour API ──────────────────────────────────────────────────────

def get_neighbors(word: str, target: str = '') -> list[tuple[str, str]]:
    """
    Return 10 neighbours for *word* using fixed-bucket selection (deterministic).

    Neighbours are drawn from three fixed tiers of GloVe similarity:
        3 'close'   (sim >= 0.65)
        4 'related' (0.45 <= sim < 0.65)
        3 'distant' (sim < 0.45)
    Same word always yields the same 10 neighbours regardless of target or path.

    Target guarantee: if sim(word, target) >= 0.60 and the target is not among
    the naturally-selected 10, it replaces the last slot so the player can
    always take the final step when they are genuinely close.

    Returns list of (neighbor_word, label) where label is 'close'|'related'|'distant'.
    """
    wl = word.lower()
    tl = target.lower() if target else ''
    key = (wl, tl)
    if key in _cache:
        return _cache[key]

    model = _get_model()

    # ── Base pool: top-50 GloVe neighbours ───────────────────────────────────
    try:
        base_raw = model.most_similar(wl, topn=50)
    except KeyError:
        _cache[key] = []
        return []

    seen: set[str] = {wl}
    candidates: list[tuple[str, float]] = []
    for w, s in base_raw:
        ww = w.lower()
        if w.isalpha() and len(w) > 2 and ww not in seen:
            seen.add(ww)
            candidates.append((ww, float(s)))

    # ── Fixed-bucket selection (deterministic, no randomness) ─────────────────
    # Divide the similarity spectrum into three tiers and take a fixed number
    # from each. Same word → same 10 neighbours every time, regardless of target.
    close   = [(w, s) for w, s in candidates if s >= 0.65][:3]
    related = [(w, s) for w, s in candidates if 0.45 <= s < 0.65][:4]
    distant = [(w, s) for w, s in candidates if s < 0.45][:3]
    selected = close + related + distant

    # Fill any shortfall (e.g. few distant words) from the sorted candidate list
    if len(selected) < 10:
        sel_words = {w for w, _ in selected}
        for pair in candidates:
            if pair[0] not in sel_words:
                selected.append(pair)
                sel_words.add(pair[0])
            if len(selected) >= 10:
                break

    result = [(w, _score_label(s)) for w, s in selected[:10]]

    # ── Target guarantee (post-selection) ────────────────────────────────────
    # If the target appeared in the top-50 GloVe pool but bucket selection
    # happened to exclude it (e.g. the 'related' bucket was already full),
    # force it into the last slot.  Using pool membership rather than a
    # similarity threshold avoids fragility with polysemous words like "rose"
    # (flower vs past tense of rise) whose GloVe sim to "roses" may be < 0.60
    # even though they are obviously the same word.
    if tl and tl not in {w for w, _ in result}:
        if any(w == tl for w, _ in candidates):
            target_sim = _safe_sim(model, wl, tl)
            result[-1] = (tl, _score_label(target_sim))

    _cache[key] = result
    return result


# ── BFS optimal path ──────────────────────────────────────────────────────────

def bfs_optimal(
    start: str,
    target: str,
    max_depth: int = 10,
    max_nodes: int = 1000,
    use_target: bool = True,
) -> Optional[list[str]]:
    """
    Shortest path from *start* to *target*.

    use_target=True  — target-aware neighbours; the path the player can follow.
    use_target=False — target-unaware neighbours; cache key is (word,'') so
                       entries are shared across BFS calls with different targets,
                       giving much better cache reuse during fast startup.
    """
    if start.lower() == target.lower():
        return [start]

    hint = target if use_target else ''
    queue: deque[list[str]] = deque([[start]])
    visited: set[str]       = {start.lower()}
    expanded                = 0

    while queue and expanded < max_nodes:
        path = queue.popleft()
        if len(path) - 1 >= max_depth:
            continue
        current = path[-1]
        expanded += 1

        for word, _ in get_neighbors(current, hint):
            wl = word.lower()
            if wl == target.lower():
                return path + [word]
            if wl not in visited:
                visited.add(wl)
                queue.append(path + [word])

    return None


# ── Daily puzzle generation ───────────────────────────────────────────────────

def generate_puzzle(
    daily_seed: int,
    progress_cb=None,
) -> tuple[str, list[str], list[Optional[list[str]]]]:
    """
    Deterministically generate today's puzzle from *daily_seed*.
    Returns (target, [s1, s2, s3], [optimal_path_1, optimal_path_2, optimal_path_3]).
    Pass progress_cb(fraction: float) to receive 0.0→1.0 progress updates.
    """
    rng       = random.Random(daily_seed)
    word_list = get_word_list()
    words     = word_list.copy()
    rng.shuffle(words)

    target    = words[0]
    remaining = words[1:]
    total     = len(remaining)

    starters:      list[str]                 = []
    optimal_paths: list[Optional[list[str]]] = []

    # First pass: prefer paths of 4–8 steps (challenging but beatable)
    for i, candidate in enumerate(remaining):
        if len(starters) >= 3:
            break
        if progress_cb:
            progress_cb(i / total)
        path = bfs_optimal(candidate, target, max_depth=8, max_nodes=400)
        if path is not None and 4 <= len(path) - 1 <= 8:
            starters.append(candidate)
            optimal_paths.append(path)

    # Second pass: accept any reachable path
    if len(starters) < 3:
        for i, candidate in enumerate(remaining):
            if len(starters) >= 3:
                break
            if candidate in starters:
                continue
            if progress_cb:
                progress_cb(0.6 + 0.35 * (i / total))
            path = bfs_optimal(candidate, target, max_depth=10, max_nodes=700)
            if path is not None:
                starters.append(candidate)
                optimal_paths.append(path)

    # Last resort
    while len(starters) < 3:
        fb = rng.choice(word_list)
        if fb not in starters and fb != target:
            starters.append(fb)
            optimal_paths.append(None)

    return target, starters[:3], optimal_paths[:3]


# ── Background-friendly helpers ───────────────────────────────────────────────

def pick_puzzle_words(
    daily_seed: int,
    progress_cb=None,
) -> tuple[str, list[str]]:
    """
    Fast startup: deterministically pick target + 3 starters.

    Uses target-unaware BFS (use_target=False) so every explored word is
    cached under key (word,'').  That cache is reused across all subsequent
    calls regardless of target — the first BFS warms the neighbourhood and
    later calls hit mostly cached results, cutting GloVe lookups ~10×.

    Optimal-path computation is intentionally omitted; call
    compute_starter_paths() in a background thread for that.
    """
    rng       = random.Random(daily_seed)
    word_list = get_word_list()
    words     = word_list.copy()
    rng.shuffle(words)

    target    = words[0]
    remaining = words[1:]
    total     = len(remaining)
    starters: list[str] = []

    for i, candidate in enumerate(remaining):
        if len(starters) >= 3:
            break
        if progress_cb:
            progress_cb(i / total)
        if bfs_optimal(candidate, target, max_depth=10, max_nodes=120,
                       use_target=False) is not None:
            starters.append(candidate)

    # Fallback: fill any remaining slots without validation
    for word in remaining:
        if len(starters) >= 3:
            break
        if word not in starters:
            starters.append(word)

    if progress_cb:
        progress_cb(1.0)
    return target, starters[:3]


def compute_starter_paths(
    target: str,
    starters: list[str],
    results: list,
) -> None:
    """
    Compute the full target-aware optimal path for each starter.
    Writes results[i] = path (or None).  Designed for a background thread.

    Because all three BFS calls share the same target, (word, target) cache
    entries built during call i are reused by calls i+1 and i+2 — each
    successive call is noticeably faster than the previous one.
    """
    for i, starter in enumerate(starters):
        results[i] = bfs_optimal(starter, target)
