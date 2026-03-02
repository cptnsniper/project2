"""
game.py — pygame UI for Semantic Odyssey.

States:
    S_LOAD    - showing "Generating puzzle…" while puzzle is built
    S_START   - starter-word selection screen
    S_PLAYING - main game loop
    S_WON     - end screen (player path vs optimal path)
"""

from __future__ import annotations

import sys
import threading
from datetime import date
from typing import Optional

import pygame

from puzzle import (bfs_optimal, pick_puzzle_words, compute_starter_paths,
                    get_daily_seed, get_neighbors, get_word_list, load_word_vectors)

# ── Window ────────────────────────────────────────────────────────────────────
W, H = 960, 700

# ── Palette ───────────────────────────────────────────────────────────────────
BG         = (10,  12,  24)
HEADER_BG  = (18,  20,  40)
DIVIDER_C  = (35,  38,  65)
GOLD       = (255, 200,  50)
WHITE      = (238, 240, 255)
GRAY       = (90,  95,  120)
DIM        = (45,  48,   70)

RELATION_COLOR: dict[str, tuple[int, int, int]] = {
    'close':   (185, 100, 220),   # purple — high similarity
    'related': (80,  195, 195),   # teal   — medium similarity
    'distant': (90,  130, 200),   # muted blue — low similarity (diverse pick)
}
RELATION_LABEL: dict[str, str] = {
    'close':   'close',
    'related': 'related',
    'distant': 'distant',
}

UNDO_CLR        = (160, 100,  50)   # amber — undo button
UNDO_CLR_DIM    = ( 60,  40,  20)   # dimmed when unavailable

BTN_BG      = (22,  26,  52)
BTN_HOVER   = (40,  46,  88)
BTN_VISITED = (16,  18,  34)
BTN_TARGET  = (60,  50,  10)

PATH_CLR = (255, 185,  30)
OPT_CLR  = ( 80, 215, 140)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _txt(surf: pygame.Surface, text: str, font: pygame.font.Font,
         color: tuple, center: tuple[int, int]) -> None:
    img = font.render(text, True, color)
    surf.blit(img, img.get_rect(center=center))


def _txt_left(surf: pygame.Surface, text: str, font: pygame.font.Font,
              color: tuple, topleft: tuple[int, int]) -> None:
    surf.blit(font.render(text, True, color), topleft)


def _make_font(names: list[str], size: int, bold: bool = False) -> pygame.font.Font:
    for name in names:
        try:
            f = pygame.font.SysFont(name, size, bold=bold)
            if f is not None:
                return f
        except Exception:
            pass
    return pygame.font.Font(None, size)


# ── WordButton ────────────────────────────────────────────────────────────────

class WordButton:
    """A clickable word tile displayed during the PLAYING state."""

    def __init__(self, rect: pygame.Rect, word: str,
                 relation: str) -> None:
        self.rect     = rect
        self.word     = word
        self.relation = relation

    def draw(self, surf: pygame.Surface, mouse: tuple[int, int],
             target: str, fonts: dict) -> None:
        hovering  = self.rect.collidepoint(mouse)
        is_target = self.word.lower() == target.lower()

        if is_target:
            bg, fg = BTN_TARGET, GOLD
        elif hovering:
            bg, fg = BTN_HOVER, WHITE
        else:
            bg, fg = BTN_BG, WHITE

        pygame.draw.rect(surf, bg, self.rect, border_radius=8)
        rel_col = RELATION_COLOR.get(self.relation, GRAY)
        pygame.draw.rect(surf, rel_col, self.rect, width=2, border_radius=8)

        # Relation label — small text in the top-left corner of the tile
        label_text = RELATION_LABEL.get(self.relation, self.relation)
        surf.blit(
            fonts['xs'].render(label_text, True, rel_col),
            (self.rect.x + 8, self.rect.y + 6),
        )

        # Word — centred, bold
        _txt(surf, self.word.upper(), fonts['btn'], fg, self.rect.center)

    def clicked(self, event: pygame.event.Event,
                mouse: tuple[int, int]) -> bool:
        return (
            event.type == pygame.MOUSEBUTTONDOWN
            and event.button == 1
            and self.rect.collidepoint(mouse)
        )


# ── SemanticGame ──────────────────────────────────────────────────────────────

class SemanticGame:
    S_LOAD    = 'loading'
    S_START   = 'start'
    S_PLAYING = 'playing'
    S_WON     = 'won'

    # Button geometry (start screen)
    _SB_W, _SB_H, _SB_GAP = 250, 100, 25
    # Button geometry (game screen — 2 cols × 5 rows)
    _GB_W, _GB_H, _GB_GX, _GB_GY = 440, 70, 20, 6
    _GB_START_Y = 275

    def __init__(self) -> None:
        pygame.init()
        self.screen = pygame.display.set_mode((W, H))
        pygame.display.set_caption('Semantic Odyssey')
        self.clock = pygame.time.Clock()

        self.fonts: dict[str, pygame.font.Font] = {
            'title': _make_font(['Georgia', 'serif'],          36, bold=True),
            'big':   _make_font(['Georgia', 'serif'],          48, bold=True),
            'med':   _make_font(['Georgia', 'serif'],          24),
            'btn':   _make_font(['Georgia', 'serif'],          20, bold=True),
            'small': _make_font(['Georgia', 'serif'],          15),
            'xs':    _make_font(['Georgia', 'serif'],          12),
            'mono':  _make_font(['Consolas', 'Courier New'],   15),
        }

        self.daily_seed: int = get_daily_seed()

        # Puzzle data
        self.target:        str                      = ''
        self.starters:      list[str]                = []
        self.optimal_paths: list[Optional[list[str]]] = []

        # Game state
        self.state:         str                      = self.S_LOAD
        self.chosen_idx:    int                      = 0
        self.optimal_path:  Optional[list[str]]      = None
        self.current_word:  str                      = ''
        self.player_path:   list[str]                = []
        self.visited_words: set[str]                 = set()
        self.buttons:        list[WordButton]         = []
        self._won_btn_rect:      Optional[pygame.Rect]   = None
        self._undo_btn_rect:     Optional[pygame.Rect]   = None
        self._new_targets:       list[str]               = []
        self._new_target_rects:  list[pygame.Rect]       = []
        self._load_progress:     float                   = 0.0
        # Background optimal-path computation
        self._bg_paths:   list                   = [None, None, None]
        self._bg_thread:  Optional[threading.Thread] = None

    # ── Puzzle lifecycle ──────────────────────────────────────────────────────

    def _load_puzzle(self) -> None:
        def _render(msg: str) -> None:
            self._draw_loading(msg)
            pygame.display.flip()
            pygame.event.pump()

        # ── Stage 1: word vectors (0 % → 85 %) ───────────────────────────────
        self._load_progress = 0.02
        _render('Loading word vectors…')

        def _vec_status(msg: str) -> None:
            if 'ready' in msg.lower():
                self._load_progress = 0.85
            _render(msg)

        load_word_vectors(status_cb=_vec_status)

        # ── Stage 2: fast word selection (85 % → 100 %) ──────────────────────
        self._load_progress = 0.85
        _render('Preparing puzzle…')

        def _pick_progress(p: float) -> None:
            self._load_progress = 0.85 + 0.15 * p
            _render('Preparing puzzle…')

        target, starters = pick_puzzle_words(self.daily_seed,
                                             progress_cb=_pick_progress)
        self.target        = target
        self.starters      = starters
        self.optimal_paths = [None, None, None]
        self._load_progress = 1.0

        # ── Stage 3: optimal paths in background ─────────────────────────────
        self._bg_paths  = [None, None, None]   # fresh list; old thread keeps its ref
        bg_paths        = self._bg_paths       # capture so thread always writes here
        self._bg_thread = threading.Thread(
            target=compute_starter_paths,
            args=(target, starters, bg_paths),
            daemon=True,
        )
        self._bg_thread.start()
        self.state = self.S_START

    def _start_game(self, idx: int) -> None:
        self.chosen_idx   = idx
        starter           = self.starters[idx]
        self.optimal_path = self._bg_paths[idx]   # None if still computing
        self.current_word = starter
        self.player_path  = [starter]
        self.visited_words = {starter.lower()}
        self.state        = self.S_PLAYING
        self._refresh_buttons()

    def _refresh_buttons(self) -> None:
        neighbours = get_neighbors(self.current_word, self.target)
        sx = (W - 2 * self._GB_W - self._GB_GX) // 2
        self.buttons = []
        for i, (word, rel) in enumerate(neighbours):
            col, row = i % 2, i // 2
            rect = pygame.Rect(
                sx + col * (self._GB_W + self._GB_GX),
                self._GB_START_Y + row * (self._GB_H + self._GB_GY),
                self._GB_W, self._GB_H,
            )
            self.buttons.append(
                WordButton(rect, word, rel)
            )

    def _move_to(self, word: str) -> None:
        self.current_word = word
        self.player_path.append(word)
        self.visited_words.add(word.lower())
        if word.lower() == self.target.lower():
            # Grab whatever background has so far (may still be None)
            self.optimal_path = self._bg_paths[self.chosen_idx]
            self._new_targets = self._pick_new_targets()
            self.state = self.S_WON
        else:
            self._refresh_buttons()

    def _pick_new_targets(self) -> list[str]:
        """Pick 3 random targets from the word list, avoiding recently used words."""
        import random
        avoid = {w.lower() for w in self.player_path} | {self.target.lower()}
        options = [w for w in get_word_list() if w.lower() not in avoid]
        return random.sample(options, min(3, len(options)))

    def _continue_game(self, new_target: str) -> None:
        """Continue from current word to a newly chosen target (background BFS)."""
        self.target       = new_target
        start             = self.current_word
        self.optimal_path = None
        self.player_path  = [start]
        self.visited_words = {start.lower()}
        self.chosen_idx   = 0

        # New background-path container; old thread keeps its own reference
        self._bg_paths  = [None, None, None]
        bg_paths        = self._bg_paths
        self._bg_thread = threading.Thread(
            target=compute_starter_paths,
            args=(new_target, [start], bg_paths),
            daemon=True,
        )
        self._bg_thread.start()
        self.state = self.S_PLAYING
        self._refresh_buttons()

    def _undo(self) -> None:
        """Step back one word. The undone word becomes selectable again."""
        if len(self.player_path) <= 1:
            return
        self.player_path.pop()
        self.current_word  = self.player_path[-1]
        # Rebuild visited set from the current path so undone words are re-enabled
        self.visited_words = {w.lower() for w in self.player_path}
        self._refresh_buttons()

    # ── Drawing utilities ─────────────────────────────────────────────────────

    def _draw_loading(self, msg: str = 'Loading…') -> None:
        self.screen.fill(BG)
        _txt(self.screen, 'Semantic Odyssey', self.fonts['title'],
             WHITE, (W // 2, H // 2 - 52))
        _txt(self.screen, msg, self.fonts['med'], GRAY, (W // 2, H // 2 - 4))

        # Progress bar
        BAR_W, BAR_H = 440, 10
        bx = (W - BAR_W) // 2
        by = H // 2 + 30
        pygame.draw.rect(self.screen, DIM, (bx, by, BAR_W, BAR_H), border_radius=5)
        fill = int(BAR_W * max(0.0, min(self._load_progress, 1.0)))
        if fill > 0:
            pygame.draw.rect(self.screen, GOLD, (bx, by, fill, BAR_H), border_radius=5)
        pct = int(self._load_progress * 100)
        _txt(self.screen, f'{pct}%', self.fonts['xs'], GRAY, (W // 2, by + 26))

    def _draw_header(self, right_text: str = '') -> None:
        pygame.draw.rect(self.screen, HEADER_BG, (0, 0, W, 56))
        pygame.draw.line(self.screen, DIVIDER_C, (0, 56), (W, 56), 1)
        _txt(self.screen, 'Semantic Odyssey', self.fonts['title'],
             WHITE, (W // 2, 28))
        if right_text:
            _txt_left(self.screen, right_text, self.fonts['small'],
                      GRAY, (W - 150, 20))

    def _divider(self, y: int) -> None:
        pygame.draw.line(self.screen, DIVIDER_C, (20, y), (W - 20, y), 1)

    def _render_path(
        self,
        path: list[str],
        x: int, y: int,
        color: tuple,
        max_w: int,
    ) -> int:
        """
        Render *path* as 'WORD  →  WORD  →  …', wrapping lines to fit *max_w*.
        Returns the y coordinate immediately after the last line.
        """
        font   = self.fonts['mono']
        tokens = [w.upper() for w in path]
        line   = ''
        cy     = y

        for i, tok in enumerate(tokens):
            sep       = '  \u2192  ' if i < len(tokens) - 1 else ''
            candidate = line + tok + sep
            if font.size(candidate)[0] > max_w and line:
                self.screen.blit(font.render(line.rstrip(' \u2192 '), True, color), (x, cy))
                cy  += 22
                line = tok + sep
            else:
                line = candidate

        if line:
            self.screen.blit(font.render(line.rstrip(' \u2192 '), True, color), (x, cy))
            cy += 22

        return cy

    # ── Screens ───────────────────────────────────────────────────────────────

    def _draw_start_screen(self) -> None:
        self.screen.fill(BG)
        self._draw_header(str(date.today()))

        _txt(self.screen,
             'Navigate to the target word in as few steps as possible.',
             self.fonts['small'], GRAY, (W // 2, 78))

        self._divider(95)
        _txt(self.screen, 'TARGET', self.fonts['xs'], GRAY, (W // 2, 112))
        _txt(self.screen, self.target.upper(), self.fonts['big'], GOLD, (W // 2, 152))
        self._divider(188)

        _txt(self.screen, 'Choose your starting word:',
             self.fonts['med'], WHITE, (W // 2, 216))

        total_w = 3 * self._SB_W + 2 * self._SB_GAP
        sx      = (W - total_w) // 2
        mouse   = pygame.mouse.get_pos()

        for i, starter in enumerate(self.starters):
            rect  = pygame.Rect(sx + i * (self._SB_W + self._SB_GAP), 248,
                                self._SB_W, self._SB_H)
            hover = rect.collidepoint(mouse)
            pygame.draw.rect(self.screen, BTN_HOVER if hover else BTN_BG,
                             rect, border_radius=10)
            pygame.draw.rect(self.screen, WHITE if hover else GRAY,
                             rect, width=2, border_radius=10)
            _txt(self.screen, starter.upper(), self.fonts['btn'], WHITE, rect.center)

        self._divider(368)
        _txt(self.screen,
             'Each step shows 10 words  (close · related · distant)  —  distant words are bridges to new areas',
             self.fonts['xs'], GRAY, (W // 2, 392))

    def _draw_undo_btn(self) -> None:
        """Draw the undo button inside the header and store its rect."""
        can_undo = len(self.player_path) > 1
        rect     = pygame.Rect(W - 110, 10, 96, 36)
        self._undo_btn_rect = rect

        border = UNDO_CLR       if can_undo else UNDO_CLR_DIM
        label  = UNDO_CLR       if can_undo else UNDO_CLR_DIM
        mouse  = pygame.mouse.get_pos()
        bg     = (55, 35, 10)   if (can_undo and rect.collidepoint(mouse)) else HEADER_BG

        pygame.draw.rect(self.screen, bg, rect, border_radius=6)
        pygame.draw.rect(self.screen, border, rect, width=2, border_radius=6)
        _txt(self.screen, '\u2190 Undo', self.fonts['small'], label, rect.center)

    def _draw_game_screen(self) -> None:
        self.screen.fill(BG)
        steps = len(self.player_path) - 1
        self._draw_header(f'Steps: {steps}')
        self._draw_undo_btn()

        # Target word
        _txt(self.screen, 'TARGET', self.fonts['xs'], GRAY, (W // 2, 70))
        _txt(self.screen, self.target.upper(), self.fonts['med'], GOLD, (W // 2, 92))
        self._divider(110)

        # Path so far
        _txt_left(self.screen, 'Your path:', self.fonts['xs'], GRAY, (24, 118))
        path_end_y = self._render_path(
            self.player_path, 24, 136, PATH_CLR, W - 48
        )
        self._divider(max(path_end_y + 4, 162))

        # Current word
        cw_y = max(path_end_y + 20, 170)
        _txt(self.screen, 'CURRENT WORD', self.fonts['xs'], GRAY, (W // 2, cw_y))
        _txt(self.screen, self.current_word.upper(), self.fonts['big'],
             WHITE, (W // 2, cw_y + 38))
        self._divider(cw_y + 70)

        _txt(self.screen, 'Choose your next word:',
             self.fonts['xs'], GRAY, (W // 2, cw_y + 82))

        mouse = pygame.mouse.get_pos()
        for btn in self.buttons:
            btn.draw(self.screen, mouse, self.target, self.fonts)

    def _draw_won_screen(self) -> None:
        self.screen.fill(BG)
        self._draw_header()

        steps = len(self.player_path) - 1
        _txt(self.screen, 'You reached the target!',
             self.fonts['title'], GOLD, (W // 2, 88))
        _txt(self.screen,
             f'{steps} step{"s" if steps != 1 else ""}',
             self.fonts['big'], WHITE, (W // 2, 140))
        self._divider(182)

        # Player path
        _txt_left(self.screen, 'Your path:',
                  self.fonts['small'], PATH_CLR, (24, 196))
        y = self._render_path(self.player_path, 24, 216, PATH_CLR, W - 48)

        y += 10
        self._divider(y)
        y += 14

        # Poll background for optimal path result
        if self.optimal_path is None:
            bg = self._bg_paths[self.chosen_idx]
            if bg is not None:
                self.optimal_path = bg

        # Optimal path section
        if self.optimal_path:
            opt_steps = len(self.optimal_path) - 1
            diff      = steps - opt_steps
            label_col = OPT_CLR if diff > 0 else GOLD
            _txt_left(self.screen,
                      f'Optimal path  ({opt_steps} step{"s" if opt_steps != 1 else ""}):',
                      self.fonts['small'], label_col, (24, y))
            y = self._render_path(self.optimal_path, 24, y + 20, OPT_CLR, W - 48)
            y += 12

            if diff == 0:
                msg, col = 'Perfect — you found the optimal path!', GOLD
            elif diff == 1:
                msg, col = 'Just 1 step away from perfect!', WHITE
            else:
                msg, col = f'{diff} steps over optimal.', GRAY
            _txt(self.screen, msg, self.fonts['med'], col, (W // 2, y + 8))
            y += 32
        elif self._bg_thread is not None and self._bg_thread.is_alive():
            # Background still computing — show animated scan bar
            _txt_left(self.screen, 'Optimal path:',
                      self.fonts['small'], GRAY, (24, y))
            BAR_W, BAR_H = W - 48, 8
            bx, by_bar   = 24, y + 22
            scan_w       = BAR_W // 3
            t            = pygame.time.get_ticks()
            pos          = int((t % 1800) / 1800 * (BAR_W + scan_w)) - scan_w
            pygame.draw.rect(self.screen, DIM,
                             (bx, by_bar, BAR_W, BAR_H), border_radius=4)
            vis_s = max(0, pos)
            vis_e = min(BAR_W, pos + scan_w)
            if vis_e > vis_s:
                pygame.draw.rect(self.screen, OPT_CLR,
                                 (bx + vis_s, by_bar, vis_e - vis_s, BAR_H),
                                 border_radius=4)
            _txt(self.screen, 'computing…', self.fonts['xs'],
                 GRAY, (W // 2, by_bar + 22))
            y = by_bar + 42
        else:
            _txt_left(self.screen, 'No optimal path found within search budget.',
                      self.fonts['small'], GRAY, (24, y))
            y += 30

        # ── Continue journey ──────────────────────────────────────────────────
        y += 8
        self._divider(y)
        y += 20
        _txt(self.screen, 'Continue your journey — choose a new target:',
             self.fonts['small'], WHITE, (W // 2, y + 8))
        y += 28

        TB_W, TB_H, TB_GAP = 200, 62, 20
        total_tw = 3 * TB_W + 2 * TB_GAP
        tsx = (W - total_tw) // 2
        mouse = pygame.mouse.get_pos()
        self._new_target_rects = []
        for i, tgt in enumerate(self._new_targets):
            rect = pygame.Rect(tsx + i * (TB_W + TB_GAP), y, TB_W, TB_H)
            self._new_target_rects.append(rect)
            hover = rect.collidepoint(mouse)
            pygame.draw.rect(self.screen, BTN_HOVER if hover else BTN_BG,
                             rect, border_radius=10)
            pygame.draw.rect(self.screen, GOLD if hover else GRAY,
                             rect, width=2, border_radius=10)
            _txt(self.screen, tgt.upper(), self.fonts['btn'],
                 GOLD if hover else WHITE, rect.center)
        y += TB_H + 16

        # New Daily Puzzle button (smaller, secondary)
        btn_rect = pygame.Rect(W // 2 - 110, y, 220, 42)
        self._won_btn_rect = btn_rect
        hover = btn_rect.collidepoint(mouse)
        pygame.draw.rect(self.screen, BTN_HOVER if hover else BG,
                         btn_rect, border_radius=8)
        pygame.draw.rect(self.screen, GRAY, btn_rect, width=1, border_radius=8)
        _txt(self.screen, 'New Daily Puzzle', self.fonts['small'],
             WHITE if hover else GRAY, btn_rect.center)

    # ── Event handlers ────────────────────────────────────────────────────────

    def _handle_start(self, event: pygame.event.Event,
                      mouse: tuple[int, int]) -> None:
        if event.type != pygame.MOUSEBUTTONDOWN or event.button != 1:
            return
        total_w = 3 * self._SB_W + 2 * self._SB_GAP
        sx      = (W - total_w) // 2
        for i in range(3):
            rect = pygame.Rect(sx + i * (self._SB_W + self._SB_GAP),
                               248, self._SB_W, self._SB_H)
            if rect.collidepoint(mouse):
                self._start_game(i)

    def _handle_playing(self, event: pygame.event.Event,
                        mouse: tuple[int, int]) -> None:
        if event.type != pygame.MOUSEBUTTONDOWN or event.button != 1:
            return
        # Undo button
        if self._undo_btn_rect and self._undo_btn_rect.collidepoint(mouse):
            self._undo()
            return
        # Word buttons
        for btn in self.buttons:
            if btn.clicked(event, mouse):
                self._move_to(btn.word)
                break

    def _handle_won(self, event: pygame.event.Event,
                    mouse: tuple[int, int]) -> None:
        if event.type != pygame.MOUSEBUTTONDOWN or event.button != 1:
            return
        # New target buttons
        for i, rect in enumerate(self._new_target_rects):
            if rect.collidepoint(mouse) and i < len(self._new_targets):
                self._continue_game(self._new_targets[i])
                return
        # New Daily Puzzle button
        if self._won_btn_rect is not None and self._won_btn_rect.collidepoint(mouse):
            self._load_puzzle()

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self) -> None:
        self._load_puzzle()

        while True:
            mouse = pygame.mouse.get_pos()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()

                if self.state == self.S_START:
                    self._handle_start(event, mouse)
                elif self.state == self.S_PLAYING:
                    self._handle_playing(event, mouse)
                elif self.state == self.S_WON:
                    self._handle_won(event, mouse)

            # Draw the active screen
            if self.state == self.S_LOAD:
                self._draw_loading()
            elif self.state == self.S_START:
                self._draw_start_screen()
            elif self.state == self.S_PLAYING:
                self._draw_game_screen()
            elif self.state == self.S_WON:
                self._draw_won_screen()

            pygame.display.flip()
            self.clock.tick(60)
