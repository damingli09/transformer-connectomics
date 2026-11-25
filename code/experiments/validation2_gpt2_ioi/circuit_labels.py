"""Known mechanistic-interpretability circuit labels for GPT-2 Small.

Used as ground truth for the circuit-recovery validation.

Source:
- IOI circuit: Wang et al. (2022), "Interpretability in the Wild".
  26 heads form the IOI circuit, partitioned into functional classes:
  Name Mover, Negative Name Mover, S-Inhibition, Induction, Duplicate Token,
  Previous Token, Backup Name Mover.

Conventions:
  - Heads as (layer, head_in_layer) tuples.
  - Flat index = layer * n_heads + head_in_layer (n_heads = 12 for GPT-2 Small).
  - Class IDs are arbitrary small integers (0, 1, ...) that group heads.
"""

from __future__ import annotations

# IOI circuit (Wang et al., 2022, Table 2). Class ID indexes the functional
# subgroup so NMI tests respect the within-circuit class structure.
# Class 0: Name Mover
# Class 1: Negative Name Mover
# Class 2: S-Inhibition
# Class 3: Induction
# Class 4: Duplicate Token
# Class 5: Previous Token
# Class 6: Backup Name Mover
IOI_HEADS_GPT2_SMALL: list[tuple[int, int, int]] = [
    (9, 6, 0),  (9, 9, 0),  (10, 0, 0),                           # Name Movers
    (10, 7, 1), (11, 10, 1),                                      # Negative Name Movers
    (7, 3, 2),  (7, 9, 2),  (8, 6, 2),  (8, 10, 2),               # S-Inhibition
    (5, 5, 3),  (5, 8, 3),  (5, 9, 3),  (6, 9, 3),                # Induction (in IOI)
    (0, 1, 4),  (0, 10, 4), (3, 0, 4),                            # Duplicate Token
    (2, 2, 5),  (2, 9, 5),  (4, 11, 5),                           # Previous Token
    (9, 0, 6),  (9, 7, 6),  (10, 1, 6), (10, 2, 6),
    (10, 6, 6), (10, 10, 6), (11, 2, 6), (11, 9, 6),              # Backup Name Movers
]


def flat_index(layer: int, head: int, n_heads: int = 12) -> int:
    return layer * n_heads + head


def ioi_label_dict(n_heads: int = 12) -> dict[int, int]:
    """Return {flat_index: class_id} for all IOI heads."""
    return {flat_index(l, h, n_heads): cid for (l, h, cid) in IOI_HEADS_GPT2_SMALL}
