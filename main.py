"""
main.py — Entry point for Semantic Odyssey.

Run:
    python main.py

Requirements:
    pip install pygame gensim

On the very first run, gensim downloads the GloVe word-vector model
(~66 MB) to ~/.cache/gensim-data/.  All subsequent launches load from
disk in a couple of seconds.
"""

import sys

if __name__ == '__main__':
    try:
        from game import SemanticGame
    except ImportError as e:
        print(f'Missing dependency: {e}')
        print('Install with:  pip install pygame gensim')
        sys.exit(1)

    SemanticGame().run()
