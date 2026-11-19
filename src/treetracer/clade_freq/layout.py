"""Pure-Python newick parser and rectangular tree layout engine.

Produces (x, y) coordinates for every node in a rooted, time-calibrated
tree, suitable for drawing with Plotly go.Scatter.

Layout convention
-----------------
- x-axis  = cumulative branch length from the root (time axis).
  Root is at x=0; tips are at their total root-to-tip distance.
- y-axis  = tip rank (0, 1, 2, ...) assigned by an in-order traversal.
  Internal nodes sit at the mean y of their children.

The tree is drawn as an L-shaped (rectangular/cladogram) layout:
  - A horizontal segment from a node's x to its parent's x at the node's y.
  - A vertical segment from the topmost to the bottommost child y at the
    parent's x.

NEXUS translate map
-------------------
BEAST / MrBayes newick strings use integer tokens as tip labels.
Pass the translate map {token: taxon_name} and it is applied during parsing.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Node:
    name:     str            = ""
    length:   float          = 0.0       # branch length to parent
    has_length: bool         = False     # whether ':' was present in Newick
    children: list["Node"]   = field(default_factory=list)
    # Layout fields filled by _assign_layout():
    x:        float          = 0.0       # cumulative root-to-node distance
    y:        float          = 0.0       # tip-rank based vertical position
    is_tip:   bool           = False


# ---------------------------------------------------------------------------
# Newick tokeniser
# ---------------------------------------------------------------------------

_Token = tuple[str, str]


def _skip_annotation(newick: str, start: int) -> int:
    """Return the first position after a balanced ``[...]`` comment."""
    depth = 1
    i = start + 1
    quote = ""
    while i < len(newick) and depth:
        current = newick[i]
        if quote:
            if current == quote:
                if i + 1 < len(newick) and newick[i + 1] == quote:
                    i += 2
                    continue
                quote = ""
        elif current in ("'", '"'):
            quote = current
        elif current == "[":
            depth += 1
        elif current == "]":
            depth -= 1
        i += 1
    if depth:
        raise ValueError("newick parse: unterminated annotation block")
    return i


def _tokenise(newick: str) -> list[_Token]:
    """Tokenize Newick while discarding NEXUS ``[...]`` annotations.

    This scanner deliberately handles quoted labels itself. The former regex
    parser split a quoted taxon containing whitespace or punctuation and could
    not distinguish a malformed length from an absent one. Token kinds make
    those cases explicit without changing the public ``Node`` representation.
    Single- and double-quoted labels support the NEXUS convention of escaping
    the quote by doubling it.
    """
    tokens: list[_Token] = []
    i = 0
    n = len(newick)

    while i < n:
        char = newick[i]
        if char.isspace():
            i += 1
            continue

        if char == "[":
            i = _skip_annotation(newick, i)
            continue

        if char in "(),;":
            tokens.append((char, char))
            i += 1
            continue

        if char == ":":
            i += 1
            # BEAST commonly writes edge annotations between the colon and
            # numeric length: ``A:[&rate=...]0.25``.
            while True:
                while i < n and newick[i].isspace():
                    i += 1
                if i < n and newick[i] == "[":
                    i = _skip_annotation(newick, i)
                    continue
                break
            start = i
            while (i < n and not newick[i].isspace()
                   and newick[i] not in "(),;[]"):
                i += 1
            tokens.append(("length", newick[start:i]))
            continue

        if char in ("'", '"'):
            quote = char
            i += 1
            value: list[str] = []
            while i < n:
                current = newick[i]
                if current == quote:
                    if i + 1 < n and newick[i + 1] == quote:
                        value.append(quote)
                        i += 2
                        continue
                    i += 1
                    break
                value.append(current)
                i += 1
            else:
                raise ValueError("newick parse: unterminated quoted label")
            tokens.append(("label", "".join(value)))
            continue

        start = i
        while (i < n and not newick[i].isspace()
               and newick[i] not in "(),:;[]"):
            i += 1
        if start == i:
            raise ValueError(
                f"newick parse: unexpected character {newick[i]!r} at pos {i}"
            )
        tokens.append(("label", newick[start:i]))

    return tokens


# ---------------------------------------------------------------------------
# Iterative parser
# ---------------------------------------------------------------------------

def _parse_newick(
    newick: str,
    *,
    require_branch_lengths: bool = False,
) -> Node:
    """Parse a Newick string and return its root ``Node``.

    Parsing is iterative, so deeply nested caterpillar trees do not hit
    Python's recursion limit. By default a missing branch length retains the
    historical display behavior (``length == 0.0``). Scientific callers can
    set ``require_branch_lengths=True`` to require a finite explicit length on
    every non-root edge.
    """
    tokens = _tokenise(newick.strip())
    if not tokens:
        raise ValueError("newick parse: empty tree")

    root: Node | None = None
    stack: list[Node] = []
    last_node: Node | None = None
    expecting_subtree = True
    just_closed_internal = False
    ended = False

    def attach(node: Node) -> None:
        nonlocal root
        if stack:
            stack[-1].children.append(node)
        elif root is None:
            root = node
        else:
            raise ValueError("newick parse: more than one root subtree")

    for kind, value in tokens:
        if ended:
            raise ValueError("newick parse: content after terminal ';'")

        if kind == "(":
            if not expecting_subtree:
                raise ValueError("newick parse: unexpected '('")
            node = Node()
            attach(node)
            stack.append(node)
            last_node = None
            expecting_subtree = True
            just_closed_internal = False
        elif kind == "label":
            if expecting_subtree:
                if not value:
                    raise ValueError("newick parse: empty tip label")
                node = Node(name=value, is_tip=True)
                attach(node)
                last_node = node
                expecting_subtree = False
                just_closed_internal = False
            elif just_closed_internal and last_node is not None:
                if last_node.name:
                    raise ValueError("newick parse: duplicate internal label")
                last_node.name = value
                just_closed_internal = False
            else:
                raise ValueError(f"newick parse: unexpected label {value!r}")
        elif kind == "length":
            if last_node is None or expecting_subtree:
                raise ValueError("newick parse: branch length has no node")
            if last_node.has_length:
                raise ValueError("newick parse: duplicate branch length")
            if not value:
                raise ValueError("newick parse: missing value after ':'")
            try:
                length = float(value)
            except ValueError as exc:
                raise ValueError(
                    f"newick parse: malformed branch length {value!r}"
                ) from exc
            if not math.isfinite(length):
                raise ValueError(
                    f"newick parse: non-finite branch length {value!r}"
                )
            last_node.length = length
            last_node.has_length = True
            just_closed_internal = False
        elif kind == ",":
            if not stack or expecting_subtree or last_node is None:
                raise ValueError("newick parse: unexpected ','")
            last_node = None
            expecting_subtree = True
            just_closed_internal = False
        elif kind == ")":
            if not stack or expecting_subtree or last_node is None:
                raise ValueError("newick parse: unexpected ')'")
            node = stack.pop()
            if not node.children:
                raise ValueError("newick parse: empty internal node")
            node.is_tip = False
            last_node = node
            expecting_subtree = False
            just_closed_internal = True
        elif kind == ";":
            if stack or expecting_subtree or root is None:
                raise ValueError("newick parse: premature ';'")
            ended = True

    if stack:
        raise ValueError("newick parse: missing closing ')'")
    if expecting_subtree or root is None:
        raise ValueError("newick parse: incomplete tree")

    if require_branch_lengths:
        missing = [
            node.name or "<internal>"
            for node in _collect_nodes(root)
            if node is not root and not node.has_length
        ]
        if missing:
            sample = ", ".join(repr(name) for name in missing[:3])
            suffix = "…" if len(missing) > 3 else ""
            raise ValueError(
                "newick parse: explicit branch length required on every "
                f"non-root edge; missing for {sample}{suffix}"
            )
    return root


# ---------------------------------------------------------------------------
# Translate map application
# ---------------------------------------------------------------------------

def _apply_translate(node: Node, translate: Mapping[str, str]) -> None:
    """Replace tip tokens with taxon names without recursive traversal."""
    stack = [node]
    while stack:
        current = stack.pop()
        if current.is_tip and current.name in translate:
            current.name = _strip_outer_quotes(str(translate[current.name]))
        stack.extend(current.children)


def parse_newick(
    newick: str,
    *,
    translate: Mapping[str, str] | None = None,
    require_branch_lengths: bool = False,
) -> Node:
    """Parse one raw Newick tree without assigning plotting coordinates.

    ``translate`` maps source tip tokens to taxon names. Set
    ``require_branch_lengths=True`` for scientific calculations that require
    an explicit finite length on every non-root edge. Visualization callers
    retain the historical behavior in which an omitted length becomes zero.
    """
    root = _parse_newick(
        newick,
        require_branch_lengths=require_branch_lengths,
    )
    if translate:
        _apply_translate(root, translate)
    return root


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

def _assign_x(node: Node, parent_x: float = 0.0) -> None:
    """Assign cumulative root-to-node x positions."""
    node.x = parent_x + node.length
    stack = [(child, node.x) for child in reversed(node.children)]
    while stack:
        current, current_parent_x = stack.pop()
        current.x = current_parent_x + current.length
        stack.extend(
            (child, current.x) for child in reversed(current.children)
        )


def _assign_y(node: Node, counter: list[int]) -> None:
    """Assign y positions: tips get integer ranks, internals get child mean."""
    stack: list[tuple[Node, bool]] = [(node, False)]
    while stack:
        current, expanded = stack.pop()
        if not current.children:
            current.y = float(counter[0])
            counter[0] += 1
        elif expanded:
            current.y = sum(c.y for c in current.children) / len(current.children)
        else:
            stack.append((current, True))
            stack.extend((child, False) for child in reversed(current.children))

def _ladderize(node: Node) -> None:
    """Sort children so the subtree with fewer tips comes first.
    
    This gives an ascending staircase layout (smallest clade on top).
    Does not change topology.
    """
    tip_counts: dict[int, int] = {}
    stack: list[tuple[Node, bool]] = [(node, False)]
    while stack:
        current, expanded = stack.pop()
        if not current.children:
            tip_counts[id(current)] = 1
        elif expanded:
            current.children.sort(key=lambda child: tip_counts[id(child)])
            tip_counts[id(current)] = sum(
                tip_counts[id(child)] for child in current.children
            )
        else:
            stack.append((current, True))
            stack.extend((child, False) for child in reversed(current.children))


def _count_tips(node: Node) -> int:
    """Count the number of tips in a subtree."""
    count = 0
    stack = [node]
    while stack:
        current = stack.pop()
        if current.children:
            stack.extend(current.children)
        else:
            count += 1
    return count

def _assign_layout(root: Node) -> None:
    _ladderize(root)
    _assign_x(root, 0.0)
    _assign_y(root, [0])


def _collect_nodes(node: Node) -> list[Node]:
    result: list[Node] = []
    stack = [node]
    while stack:
        current = stack.pop()
        result.append(current)
        stack.extend(reversed(current.children))
    return result


# ---------------------------------------------------------------------------
# NEXUS parsing
# ---------------------------------------------------------------------------

_TRANSLATE_RE = re.compile(
    r"Translate\s*\n(.*?)\n\s*;", re.IGNORECASE | re.DOTALL
)
#_TREE_LINE_RE = re.compile(
#    r"^\s*tree\s+\S+.*?=\s*(\(.*)", re.IGNORECASE | re.MULTILINE
#)
_TREE_LINE_RE = re.compile(
    r"^\s*tree\s+\S+\s*(?:\[[^\]]*\]\s*)*=\s*(?:\[[^\]]*\]\s*)?(\(.*)",
    re.IGNORECASE | re.MULTILINE
)


def _strip_outer_quotes(label: str) -> str:
    if len(label) >= 2 and label[0] == label[-1] and label[0] in ("'", '"'):
        quote = label[0]
        return label[1:-1].replace(quote + quote, quote)
    return label


def parse_nexus(nexus_bytes: bytes) -> tuple[Node, dict[str, str]]:
    """Parse NEXUS bytes into a laid-out root Node and translate map."""
    text = nexus_bytes.decode("utf-8", errors="replace")

    # Extract translate map.
    translate: dict[str, str] = {}
    tm = _TRANSLATE_RE.search(text)
    if tm:
        for line in tm.group(1).splitlines():
            line = line.strip().rstrip(",")
            if not line:
                continue
            parts = line.split(None, 1)
            if len(parts) == 2:
                translate[parts[0]] = _strip_outer_quotes(parts[1])

    # Find the tree line by scanning line by line.
    # Handles any annotation between tree name and '=', and any annotation
    # between '=' and the opening '(' of the newick string.
    newick = None
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.lower().startswith("tree "):
            continue
        # # Find the '=' sign
        # eq_idx = stripped.find("=")
        # if eq_idx == -1:
        #     continue
        # # Everything after '=' is the newick, possibly preceded by annotations
        # after_eq = stripped[eq_idx + 1:].strip()
        # # Skip any leading [...] annotations (e.g. [&R], [&lnCladeCred=...])
        # while after_eq.startswith("["):
        #     close = after_eq.find("]")
        #     if close == -1:
        #         break
        #     after_eq = after_eq[close + 1:].strip()
        # # Now after_eq should start with '('
        # if after_eq.startswith("("):
        #     newick = after_eq.rstrip(";").strip()
        #     break
        # Find the '=' at bracket depth 0, ignoring '=' inside [...] blocks.
        depth = 0
        eq_pos = -1
        for i, c in enumerate(stripped):
            if c == '[':
                depth += 1
            elif c == ']':
                depth -= 1
            elif c == '=' and depth == 0:
                eq_pos = i
                break
        if eq_pos == -1:
            continue
        after_eq = stripped[eq_pos + 1:].strip()
        # Skip any leading [...] annotations after '=' (e.g. [&R])
        while after_eq.startswith("["):
            close = after_eq.find("]")
            if close == -1:
                break
            after_eq = after_eq[close + 1:].strip()
        if after_eq.startswith("("):
            newick = after_eq.rstrip(";").strip()
            break

    if newick is None:
        raise ValueError("No 'tree ... = <newick>' line found in NEXUS bytes.")

    root = parse_newick(newick, translate=translate)

    for node in _collect_nodes(root):
        node.is_tip = not bool(node.children)
        if node.is_tip:
            node.name = _strip_outer_quotes(node.name)

    _assign_layout(root)
    return root, translate


# ---------------------------------------------------------------------------
# Plotly trace builders
# ---------------------------------------------------------------------------

def build_tree_traces(
    root:        Node,
    x_offset:    float = 0.0,
    x_scale:     float = 1.0,
    x_flip:      bool  = False,
    highlight:   set[str] | None = None,
    label_side:  str  = "right",   # "right" | "left"
    show_labels: bool = True,
) -> list[dict]:
    """Build Plotly trace dicts for a rectangular tree layout.

    Args:
        root:       Laid-out root Node.
        x_offset:   Horizontal shift (used to place the two trees side by side).
        x_scale:    Multiply all x coordinates by this factor.
        x_flip:     Mirror the tree horizontally (for the right-hand tree).
        highlight:  Set of tip names to colour red.  Others are grey.
        label_side: Which side to draw tip labels.
        show_labels: Whether to draw tip name annotations.

    Returns:
        List of Plotly trace dicts (go.Scatter-compatible).
    """
    highlight = highlight or set()
    nodes     = _collect_nodes(root)
    tips      = [n for n in nodes if n.is_tip]

    def tx(x: float) -> float:
        scaled = x * x_scale
        return (x_offset - scaled) if x_flip else (x_offset + scaled)

    # ── Branch segments (L-shaped) ──────────────────────────────────────────
    branch_x: list[float | None] = []
    branch_y: list[float | None] = []

    for node in nodes:
        for child in node.children:
            # Horizontal arm from parent x to child x at child y.
            branch_x += [tx(node.x), tx(child.x), None]
            branch_y += [child.y,    child.y,      None]
            # Vertical arm at parent x spanning child y range.
        if node.children:
            ys = [c.y for c in node.children]
            branch_x += [tx(node.x), tx(node.x), None]
            branch_y += [min(ys),    max(ys),     None]

    traces = [dict(
        type="scatter",
        x=branch_x, y=branch_y,
        mode="lines",
        line=dict(color="#888888", width=1),
        hoverinfo="skip",
        showlegend=False,
    )]

    # ── Tip markers ─────────────────────────────────────────────────────────
    tip_x_hi, tip_y_hi, tip_names_hi = [], [], []
    tip_x_lo, tip_y_lo, tip_names_lo = [], [], []

    for tip in tips:
        if tip.name in highlight:
            tip_x_hi.append(tx(tip.x))
            tip_y_hi.append(tip.y)
            tip_names_hi.append(tip.name)
        else:
            tip_x_lo.append(tx(tip.x))
            tip_y_lo.append(tip.y)
            tip_names_lo.append(tip.name)

    if tip_x_lo:
        traces.append(dict(
            type="scatter",
            x=tip_x_lo, y=tip_y_lo,
            mode="markers",
            marker=dict(color="#aaaaaa", size=6),
            text=tip_names_lo,
            hovertemplate="%{text}<extra></extra>",
            showlegend=False,
        ))

    if tip_x_hi:
        traces.append(dict(
            type="scatter",
            x=tip_x_hi, y=tip_y_hi,
            mode="markers",
            marker=dict(color="#e63946", size=8),
            text=tip_names_hi,
            hovertemplate="%{text}<extra></extra>",
            showlegend=False,
        ))

    return traces
