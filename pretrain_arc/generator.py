"""The generator: ARC-style grids, a fixed set of steps, and rules composed from them.

numpy only and deterministic: every draw comes from a `numpy.random.Generator` the caller seeds, and nothing depends on
hash salting or set iteration order.

GRIDS. 8 x 8, colours 0-9 with 0 the background (ARC's conventions). A grid holds 4 or 5 objects. Each object is a
fixed polyomino of 2-6 cells, the sizes on one grid all different, in one of the task's 3 colours (`sample_palette`).
Objects never overlap, and objects that touch edge-on have different colours, so every object is exactly one
4-connected single-colour component of the grid. Objects are nested: each one after the first reaches into the
bounding box of one already placed (or its own box takes in a cell of one), and every object that isn't a straight line
has a neighbour's cell inside its bounding box. Every colour is shared by at least two objects. So an object can't be
picked out by its colour alone or by its bounding box alone. The layout sits inside a one-cell background border, at a
uniformly drawn offset.

OBJECTS. A grid's objects are recomputed from its cells (`components`) before every step, so a step can merge or split
objects for the next one. They are listed in raster order of their first cell.

STEPS. A step is a (selector, operator) pair: the selector picks one object, the operator changes it.
  Selectors (13): largest, smallest, first (in raster order), last, colour c (the first object of colour c, c in 1-9).
    Ties go to the first in raster order. A selector that finds nothing selects nothing.
  Operators (20): recolour c (c in 1-9); delete; move d (one cell N/S/E/W; vacated cells become 0, moved cells
    overwrite, cells moved off the grid are lost); mirror h|v (flipped inside the object's bounding box); copy d (a copy
    stamped one blank cell past the bounding box in direction d, overwriting, clipped at the edge).
  (colour c, recolour c) does nothing to any grid and is left out, so there are 13 x 20 - 9 = 251 steps (`STEPS`).

RULES. A rule of depth d is d steps applied in order. The rule of a task is drawn on the task's grids
(`sample_rule_on`): at each step, uniformly among the steps that are ADMISSIBLE on every grid of the task as it stands
(the step selects something and changes the grid). Every step of a task therefore acts on every grid of it, and the
probability of a rule given the input grids is exactly prod_t 1 / (number of admissible steps at t), which is what
makes the Bayes floor in `floors.py` exact.
"""

import itertools

import numpy as np

H, W = 8, 8
N_COL = 10
PALETTE = 3                                                   # colours a task uses, drawn from 1-9
DIRS = {"N": (-1, 0), "S": (1, 0), "E": (0, 1), "W": (0, -1)}
DIR_NAMES = ("N", "S", "E", "W")


# ------------------------------------------------------------------------------------------------------------------ #
# shapes: every fixed polyomino of 1-6 cells, normalised to its top-left corner
# ------------------------------------------------------------------------------------------------------------------ #

def _norm(cells):
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


def _all_polyominoes(n_max):
    out = {1: [((0, 0),)]}
    for n in range(2, n_max + 1):
        seen = set()
        for p in out[n - 1]:
            ps = set(p)
            for r, c in p:
                for dr, dc in DIRS.values():
                    q = (r + dr, c + dc)
                    if q not in ps:
                        seen.add(_norm(list(p) + [q]))
        out[n] = sorted(seen)
    return out


_POLY = _all_polyominoes(5)
# The shape table: the 91 fixed polyominoes of 1-5 cells, then the 216 hexominoes. The order is part of the generator
# (shape draws index into it), so it must not change.
SHAPES = [s for n in (1, 2, 3, 4, 5) for s in _POLY[n]] + _all_polyominoes(6)[6]
STRAIGHT = frozenset(i for i, s in enumerate(SHAPES) if len({r for r, _ in s}) == 1 or len({c for _, c in s}) == 1)


def shape_diameter(shp):
    """The largest 4-adjacency path distance between two cells of the shape."""
    cs = set(shp)
    best = 0
    for a in shp:
        dist = {a: 0}
        q = [a]
        while q:
            x = q.pop(0)
            for dr, dc in DIRS.values():
                y = (x[0] + dr, x[1] + dc)
                if y in cs and y not in dist:
                    dist[y] = dist[x] + 1
                    q.append(y)
        best = max(best, max(dist.values()))
    return best


# ------------------------------------------------------------------------------------------------------------------ #
# objects
# ------------------------------------------------------------------------------------------------------------------ #

def components(g):
    """The grid's objects: 4-connected single-colour non-background components, in raster order of their first cell.
    -> list of (colour, cells), cells a tuple of flat indices (r * W + c) in raster order."""
    g = np.asarray(g)
    flat = g.reshape(-1).tolist()
    seen = [False] * (H * W)
    out = []
    for p in range(H * W):
        k = flat[p]
        if k == 0 or seen[p]:
            continue
        stack, cells = [p], []
        seen[p] = True
        while stack:
            q = stack.pop()
            cells.append(q)
            r, c = divmod(q, W)
            if r > 0 and not seen[q - W] and flat[q - W] == k:
                seen[q - W] = True
                stack.append(q - W)
            if r < H - 1 and not seen[q + W] and flat[q + W] == k:
                seen[q + W] = True
                stack.append(q + W)
            if c > 0 and not seen[q - 1] and flat[q - 1] == k:
                seen[q - 1] = True
                stack.append(q - 1)
            if c < W - 1 and not seen[q + 1] and flat[q + 1] == k:
                seen[q + 1] = True
                stack.append(q + 1)
        out.append((k, tuple(sorted(cells))))
    return out


# ------------------------------------------------------------------------------------------------------------------ #
# the grid prior
# ------------------------------------------------------------------------------------------------------------------ #

GRID_PRIOR = dict(
    margin=1,                   # a background border this wide on every grid: objects live in the inner 6 x 6
    n_obj=(4, 5),               # objects a grid, uniform
    sizes=(2, 3, 4, 5, 6),      # every object a different size: a uniform n_obj-subset of these
    max_diam=4,                 # shapes of diameter at most 4 (every pentomino, 134 of the 216 hexominoes)
)

_TABLES = {}
_COLOURINGS = {}


def _tables():
    """Per shape: every placement inside the interior as flat cell indices of the full grid, and its bounding box; the
    shape ids by size. Built once."""
    if not _TABLES:
        m, max_diam = GRID_PRIOR["margin"], GRID_PRIOR["max_diam"]
        cells, boxes = [], []
        for shp in SHAPES:
            hh = max(r for r, _ in shp) + 1
            ww = max(c for _, c in shp) + 1
            pos = [(r0, c0) for r0 in range(m, H - m - hh + 1) for c0 in range(m, W - m - ww + 1)]
            cells.append(np.array([[(r0 + r) * W + c0 + c for r, c in shp] for r0, c0 in pos], np.int64).reshape(
                len(pos), len(shp)))
            boxes.append(np.array([[(r0 + r) * W + c0 + c for r in range(hh) for c in range(ww)] for r0, c0 in pos],
                                  np.int64).reshape(len(pos), hh * ww))
        by_size = {}
        for i, shp in enumerate(SHAPES):
            if shape_diameter(shp) <= max_diam:
                by_size.setdefault(len(shp), []).append(i)
        _TABLES.update(cells=cells, boxes=boxes, by_size={n: np.array(v, np.int64) for n, v in by_size.items()})
    return _TABLES


def _colourings(n):
    if n not in _COLOURINGS:
        _COLOURINGS[n] = np.array(list(itertools.product(range(PALETTE), repeat=n)), np.int64)
    return _COLOURINGS[n]


def sample_palette(rng):
    """A task's palette: PALETTE distinct colours of 1-9, sorted."""
    return tuple(sorted(int(x) for x in rng.choice(np.arange(1, N_COL), size=PALETTE, replace=False)))


def sample_grid(rng, palette=None, return_objects=False):
    """One grid of the prior (module docstring), colours from `palette` (None: a palette drawn for this grid).
    -> grid (H, W) int8 [, objects [(colour, shape id, cells)]]

    In detail: n objects (4 or 5) with sizes a uniform n-subset of 2-6, in random order, each shape uniform among the
    polyominoes of its size with diameter at most 4. Objects are placed in turn, each uniformly over the interior
    positions where it overlaps nothing and is nested against those already placed (one of its cells lies in an
    earlier object's bounding box, or its own box holds an earlier object's cell); a shape with no such position is
    redrawn at the same size up to 30 times, then the grid is redrawn. The grid is kept only when every object that is
    not a straight line holds another object's cell in its bounding box. Colours are uniform over the palette
    assignments where touching objects differ and every object shares its colour with another. Finally the layout is
    translated as a whole, uniformly over the offsets that keep it inside the interior."""
    C = GRID_PRIOR
    m = int(C["margin"])
    T = _tables()
    cols = np.arange(1, N_COL) if palette is None else np.asarray(palette)
    assert len(cols) == PALETTE or palette is None
    sizes_all = np.asarray(C["sizes"], np.int64)
    while True:
        occ = np.zeros(H * W, bool)
        inbox = np.zeros(H * W, bool)
        objs = []
        n_obj = int(C["n_obj"][int(rng.integers(len(C["n_obj"])))])
        sizes = rng.choice(sizes_all, size=n_obj, replace=False)
        ok = True
        for j in range(n_obj):
            pool = T["by_size"][int(sizes[j])]
            placed = False
            for _try in range(30):
                sid = int(pool[int(rng.integers(len(pool)))])
                pc, pb = T["cells"][sid], T["boxes"][sid]
                if not len(pc):
                    continue
                good = ~occ[pc].any(axis=1)
                if j > 0:
                    good &= inbox[pc].any(axis=1) | occ[pb].any(axis=1)
                cand = np.flatnonzero(good)
                if not len(cand):
                    continue
                q = int(cand[int(rng.integers(len(cand)))])
                occ[pc[q]] = True
                inbox[pb[q]] = True
                objs.append((sid, pc[q], pb[q]))
                placed = True
                break
            if not placed:
                ok = False
                break
        if not ok:
            continue
        own = -np.ones(H * W, np.int64)
        for i, (_, cells, _) in enumerate(objs):
            own[cells] = i
        if any(sid not in STRAIGHT and not ((own[box] >= 0) & (own[box] != i)).any()
               for i, (sid, _, box) in enumerate(objs)):
            continue
        g2 = own.reshape(H, W)
        a = np.concatenate([np.stack([g2[:, :-1].ravel(), g2[:, 1:].ravel()], 1),
                            np.stack([g2[:-1, :].ravel(), g2[1:, :].ravel()], 1)])
        a = a[(a[:, 0] >= 0) & (a[:, 1] >= 0) & (a[:, 0] != a[:, 1])]
        if palette is None:
            cols = np.sort(rng.choice(np.arange(1, N_COL), size=PALETTE, replace=False))
        CL = _colourings(n_obj)
        v = np.ones(len(CL), bool)
        if len(a):
            v &= (CL[:, a[:, 0]] != CL[:, a[:, 1]]).all(axis=1)
        cnt = np.stack([(CL == c).sum(axis=1) for c in range(PALETTE)], 1)
        v &= (np.take_along_axis(cnt, CL, axis=1) >= 2).all(axis=1)
        valid = np.flatnonzero(v)
        if not len(valid):
            continue
        cl = CL[int(valid[int(rng.integers(len(valid)))])]
        allc = np.concatenate([cells for _, cells, _ in objs])
        rs, cs = allc // W, allc % W
        dr = int(rng.integers(m - int(rs.min()), H - m - int(rs.max())))
        dc = int(rng.integers(m - int(cs.min()), W - m - int(cs.max())))
        shift = dr * W + dc
        g = np.zeros(H * W, np.int8)
        out_objs = []
        for i, (sid, cells, _) in enumerate(objs):
            k = int(cols[cl[i]])
            cc = cells + shift
            g[cc] = k
            out_objs.append((k, sid, tuple(sorted(int(c) for c in cc))))
        g = g.reshape(H, W)
        return (g, out_objs) if return_objects else g


# ------------------------------------------------------------------------------------------------------------------ #
# selectors, operators, steps, rules
# ------------------------------------------------------------------------------------------------------------------ #

SELECTORS = [("largest",), ("smallest",), ("first",), ("last",)] + [("colour", c) for c in range(1, N_COL)]
OPERATORS = ([("recolour", c) for c in range(1, N_COL)] + [("delete",)] + [("move", d) for d in DIR_NAMES]
             + [("mirror", "h"), ("mirror", "v")] + [("copy", d) for d in DIR_NAMES])
STEPS = [(s, o) for s in SELECTORS for o in OPERATORS
         if not (s[0] == "colour" and o[0] == "recolour" and s[1] == o[1])]
STEP_ID = {b: i for i, b in enumerate(STEPS)}
N_STEPS = len(STEPS)                                          # 13 x 20 - 9 = 251


def select(objs, sel):
    """Index of the object `sel` picks in objs (a `components` list), or None."""
    if not objs:
        return None
    if sel[0] == "largest":
        best = max(len(c[1]) for c in objs)
        return next(i for i, c in enumerate(objs) if len(c[1]) == best)
    if sel[0] == "smallest":
        best = min(len(c[1]) for c in objs)
        return next(i for i, c in enumerate(objs) if len(c[1]) == best)
    if sel[0] == "first":
        return 0
    if sel[0] == "last":
        return len(objs) - 1
    if sel[0] == "colour":
        return next((i for i, c in enumerate(objs) if c[0] == sel[1]), None)
    raise ValueError(sel)


def bbox(cells):
    rs = [p // W for p in cells]
    cs = [p % W for p in cells]
    return min(rs), min(cs), max(rs), max(cs)


def operate(g, obj, op):
    """The grid after operator `op` on object obj = (colour, cells); a new array."""
    out = np.array(g, np.int8, copy=True)
    k, cells = obj
    flat = out.reshape(-1)
    if op[0] == "recolour":
        flat[list(cells)] = op[1]
    elif op[0] == "delete":
        flat[list(cells)] = 0
    elif op[0] == "move":
        dr, dc = DIRS[op[1]]
        flat[list(cells)] = 0
        for p in cells:
            r, c = divmod(p, W)
            r, c = r + dr, c + dc
            if 0 <= r < H and 0 <= c < W:
                out[r, c] = k
    elif op[0] == "mirror":
        r0, c0, r1, c1 = bbox(cells)
        flat[list(cells)] = 0
        for p in cells:
            r, c = divmod(p, W)
            if op[1] == "h":
                c = c0 + c1 - c
            else:
                r = r0 + r1 - r
            out[r, c] = k
    elif op[0] == "copy":
        r0, c0, r1, c1 = bbox(cells)
        dr, dc = DIRS[op[1]]
        dr *= (r1 - r0 + 2)
        dc *= (c1 - c0 + 2)
        for p in cells:
            r, c = divmod(p, W)
            r, c = r + dr, c + dc
            if 0 <= r < H and 0 <= c < W:
                out[r, c] = k
    else:
        raise ValueError(op)
    return out


def _step(b):
    return STEPS[b] if isinstance(b, (int, np.integer)) else b


def apply_step(g, b, objs=None):
    """One step (an id into STEPS, or a (selector, operator) pair) on grid g. -> (grid, the selected object or None).
    objs: the grid's `components`, if already computed."""
    b = _step(b)
    objs = components(g) if objs is None else objs
    i = select(objs, b[0])
    if i is None:
        return np.array(g, np.int8, copy=True), None
    return operate(g, objs[i], b[1]), objs[i]


def apply_rule(g, rule, trace=False):
    """A rule (a sequence of steps) on grid g. -> output grid [, per step (selected object or None, grid after it)]."""
    cur = np.asarray(g, np.int8)
    tr = []
    for b in rule:
        cur, obj = apply_step(cur, b)
        tr.append((obj, cur))
    return (cur, tr) if trace else cur


def step_on(grids, b, objs=None):
    """Step b on each grid -> the list of output grids, or None when the step is not ADMISSIBLE on the set: on some
    grid it selects nothing or leaves the grid unchanged."""
    out = []
    for j, g in enumerate(grids):
        g2, obj = apply_step(g, b, None if objs is None else objs[j])
        if obj is None or np.array_equal(g2, g):
            return None
        out.append(g2)
    return out


def admissible_steps(grids, allowed=None):
    """Every step (among `allowed` ids, default all) admissible on the set of grids -> [(step id, output grids)]."""
    objs = [components(g) for g in grids]
    out = []
    for bi, b in enumerate(STEPS):
        if allowed is not None and bi not in allowed:
            continue
        o = step_on(grids, b, objs)
        if o is not None:
            out.append((bi, o))
    return out


def sample_rule_on(rng, grids, depth=None, allowed_steps=None):
    """THE RULE DRAW on a set of grids: at each step, a step uniform among those admissible on every grid as the set
    stands, applied. `allowed_steps`: one set of permitted step ids per position (its length is the depth); default
    every step at each of `depth` positions. -> (rule as step ids, the grids after it, admissible counts per position),
    or None when some position admits nothing."""
    if allowed_steps is None:
        allowed_steps = [None] * depth
    cur = [np.asarray(g, np.int8) for g in grids]
    rule, n_adm = [], []
    for allowed in allowed_steps:
        adm = admissible_steps(cur, allowed)
        if not adm:
            return None
        bi, o = adm[int(rng.integers(len(adm)))]
        rule.append(bi)
        n_adm.append(len(adm))
        cur = o
    return tuple(rule), cur, n_adm


def sample_task(rng, depth=None, k=3, n_test=1, palette=None, allowed_steps=None):
    """A task: a palette, k + n_test grids drawn i.i.d. from the prior on it, and a rule drawn on all of them
    (`sample_rule_on`), the whole task redrawn when some position admits nothing.
    -> dict(rule, palette, n_adm, demo_in, demo_out, test_in, test_out), grids stacked as (n, H, W) int8."""
    while True:
        pal = sample_palette(rng) if palette is None else tuple(palette)
        gi = [sample_grid(rng, palette=pal) for _ in range(k + n_test)]
        r = sample_rule_on(rng, gi, depth, allowed_steps)
        if r is None:
            continue
        rule, go, n_adm = r
        gi, go = np.stack(gi), np.stack(go)
        return {"rule": rule, "palette": pal, "n_adm": n_adm, "demo_in": gi[:k], "demo_out": go[:k],
                "test_in": gi[k:], "test_out": go[k:]}


def admissible(rule, g):
    """Every step of `rule` acts on grid g (selects something and changes it)."""
    cur = np.asarray(g, np.int8)
    for b in rule:
        o = step_on([cur], b)
        if o is None:
            return False
        cur = o[0]
    return True


def fresh_grids(rng, rule, palette, n, max_tries=200000):
    """More examples of a rule: grids from the prior on the palette on which the rule is admissible (rejection
    sampling), and their outputs. -> (inputs (n, H, W), outputs (n, H, W), tries)."""
    xs, ys, tries = [], [], 0
    while len(xs) < n and tries < max_tries:
        tries += 1
        g = sample_grid(rng, palette=palette)
        if admissible(rule, g):
            xs.append(g)
            ys.append(apply_rule(g, rule))
    return np.stack(xs), np.stack(ys), tries


# ------------------------------------------------------------------------------------------------------------------ #
# names
# ------------------------------------------------------------------------------------------------------------------ #

COLOUR_NAMES = ("black", "blue", "red", "green", "yellow", "grey", "magenta", "orange", "azure", "maroon")
_DIR_WORDS = {"N": "up", "S": "down", "E": "right", "W": "left"}
_SIDE_WORDS = {"N": "above it", "S": "below it", "E": "to its right", "W": "to its left"}


def step_code(b):
    """A step as a short code, e.g. `move/E(largest)`."""
    s, o = _step(b)
    return f"{'/'.join(map(str, o))}({'/'.join(map(str, s))})"


def rule_code(rule):
    return " ; ".join(step_code(b) for b in rule)


def selector_words(s):
    if s[0] == "colour":
        return f"the first {COLOUR_NAMES[s[1]]} object"
    return {"largest": "the largest object", "smallest": "the smallest object",
            "first": "the first object in reading order", "last": "the last object in reading order"}[s[0]]


def operator_words(o):
    if o[0] == "recolour":
        return f"recolour it {COLOUR_NAMES[o[1]]}"
    if o[0] == "delete":
        return "delete it"
    if o[0] == "move":
        return f"move it one cell {_DIR_WORDS[o[1]]}"
    if o[0] == "mirror":
        return "flip it left-right inside its bounding box" if o[1] == "h" else \
            "flip it upside-down inside its bounding box"
    if o[0] == "copy":
        return f"stamp a copy of it {_SIDE_WORDS[o[1]]}, one blank cell past its bounding box (clipped at the edge)"
    raise ValueError(o)


def step_words(b):
    """A step in English, e.g. 'the largest object: move it one cell right'."""
    s, o = _step(b)
    return f"{selector_words(s)}: {operator_words(o)}"
