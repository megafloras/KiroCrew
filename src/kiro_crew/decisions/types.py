"""Typed question and answer shapes for the DecisionOracle seam.

Two question types, one answer type. The two mirror the TypeSafe System One
primitives the shipped points use; the LLM implementation answers the SAME pair
so a shadow row from either implementation is comparable field-for-field. A
third type would have to be added to both implementations AND to every point
file's ``agree`` definition, so the set is deliberately closed here.

Why one ``Answer`` and not three
--------------------------------
A caller branches on ``value`` and gates on ``p``/``confidence``; it does not
care which question type produced the row. Three answer classes would push an
``isinstance`` ladder into every point file and into the log writer, for no
information the single shape cannot carry:

* ``value`` -- the chosen option (``Choice``) or the yes-probability (``Noul``).
* ``p`` -- the probability mass behind ``value``. For ``Noul`` that IS the
  value, so the two agree by construction rather than by coincidence; keeping
  the field filled anyway means a consumer can threshold on ``p`` uniformly.
* ``confidence`` -- ``None`` for ``Noul``, because the API does not return one
  for that type. ``None`` means "not reported", never "zero confidence": a
  consumer that treats a missing confidence as 0.0 would silently discard every
  ``Noul`` answer.

These are plain (non-frozen) dataclasses on purpose. ``frozen=True`` would
synthesise ``__hash__``, and ``Choice.options`` is a list -- so a hash would
raise at the one place it looked safe. Nothing here is used as a dict key.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Choice:
    """Pick one of *options*.

    ``options`` is the answer domain, not a hint: the implementation may only
    return a member of it, and the wire mapping sends each option as its own
    rubric key. Two options is the useful floor; one option asks nothing.
    """

    id: str
    prompt: str
    options: list[str] = field(default_factory=list)


@dataclass
class Noul:
    """Ask whether one statement holds. The answer is the probability it does.

    ``value`` is that probability (0.0 = no, 1.0 = yes), not a bool: rounding to
    a bool at the seam would throw away the only thing a shadow row is for --
    seeing WHERE the model sits before any code depends on the verdict.
    """

    id: str
    prompt: str


#: One question of either type. A union rather than a base class: the two carry
#: disjoint fields and no shared behaviour, so a base class would only be a place
#: for a future ``ask``-time branch to hide.
Question = Choice | Noul


@dataclass
class Answer:
    """One answer, keyed in :data:`Answers` by the question id it replies to.

    ``id`` is carried on the answer too, not just as the dict key, so a row
    survives being pulled out of the mapping (a log line, a filtered list)
    without becoming anonymous.
    """

    id: str
    value: object
    p: float
    confidence: float | None = None


#: Question id -> :class:`Answer`. The key set always equals the ids of the
#: questions asked: an implementation that cannot answer one question answers
#: none, so a caller never has to distinguish "absent" from "unanswerable".
Answers = dict[str, Answer]
