"""Shadow hooks for individual decision points.

Each module here owns ONE seam: how the local state is shaped into a question
set, what the existing logic's answer (the ``baseline``) is, and what counts as
agreement between the two. The generic gate, transport, logging and report live
one package up in :mod:`kiro_crew.decisions`.

Every hook is import-safe on its own: the core package is imported INSIDE the
hook function, so a tree without it degrades to a no-op instead of failing at
import time.
"""

#: Cap on a skill key in any point's state or offered options. Both skill points
#: bound their descriptions and neither bounded the KEY, so a pathological key
#: carried an unbounded string into the state and into the option list. Declared
#: here, not in either point, so the two cannot drift to different numbers.
MAX_KEY_CHARS = 120
