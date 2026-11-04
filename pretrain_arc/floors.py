"""The Bayes floor: the best any solver can do on a task, given only its demonstrations.

A task's rule is drawn on its k demonstration inputs and its test input together (`generator.sample_rule_on`): at each
position, a step uniform among those admissible on all k + 1 grids as they stand. So the prior probability of a rule
given the k + 1 inputs is prod_t 1 / n_adm(t), and the posterior given the demonstration outputs is that prior
restricted to the rules that reproduce every demonstration. The Bayes-optimal solver knows the generator and the depth
and answers with the test outputs of most posterior mass. Its expected error with B attempts is 1 - (mass of the top B
outputs). No solver, however good, can beat that in expectation: the remaining error is ambiguity in the task itself.

`posterior` computes it exactly, by a search over the joint state of the k + 1 grids that merges rules leaving all of
them identical (such rules share every continuation). Its one prune is exact: at the last position a state is expanded
only if every demonstration grid can still reach its output in one step (a step changes at most twice the selected
object's cells). Depth 4 visits ~10^5 joint states and takes tens of seconds a task on one CPU core.
"""

import numpy as np

from . import generator as G

_SEL_ID = {s: i for i, s in enumerate(G.SELECTORS)}
_OP_ID = {o: i for i, o in enumerate(G.OPERATORS)}
_STEP_SO = [(_SEL_ID[s], _OP_ID[o]) for s, o in G.STEPS]
_UNSET = object()


def expand(grids):
    """Every step admissible on the joint state -> [(step id, child grids)]; each operator applied once per distinct
    (grid, object, operator)."""
    n = len(grids)
    objs = [G.components(g) for g in grids]
    sel = [[G.select(c, s) for s in G.SELECTORS] for c in objs]
    res = {}
    out = []
    for bi, (si, oi) in enumerate(_STEP_SO):
        ch = []
        for j in range(n):
            ci = sel[j][si]
            if ci is None:
                break
            key = (j, ci, oi)
            g2 = res.get(key, _UNSET)
            if g2 is _UNSET:
                g2 = G.operate(grids[j], objs[j][ci], G.OPERATORS[oi])
                g2 = None if np.array_equal(g2, grids[j]) else g2
                res[key] = g2
            if g2 is None:
                break
            ch.append(g2)
        else:
            out.append((bi, ch))
    return out


def _reach_one(g, tgt):
    """A single step could turn g into tgt (necessary: the cell difference is at most 2 x g's largest object)."""
    diff = int((g != tgt).sum())
    if diff == 0:
        return False                                                # every step changes the grid
    m = max((len(c[1]) for c in G.components(g)), default=0)
    return diff <= 2 * m


def posterior(demo_in, demo_out, test_in, depth, max_states=None):
    """The exact posterior over the test output given the demonstrations and the rule's depth.
    -> dict(outputs {output grid bytes: mass}, total, n_consistent, states per depth, pruned, expanded), or None when
    the search passes `max_states`."""
    k = len(demo_in)
    start = tuple(np.asarray(g, np.int8) for g in list(demo_in) + list(test_in))
    tgts = [np.asarray(o, np.int8) for o in demo_out]
    states = {tuple(g.tobytes() for g in start): (start, 1.0)}
    n_states = [1]
    for t in range(depth - 1):
        nxt = {}
        for key, (grids, w) in states.items():
            adm = expand(grids)
            if not adm:
                continue
            w2 = w / len(adm)
            for bi, ch in adm:
                ck = tuple(c.tobytes() for c in ch)
                e = nxt.get(ck)
                if e is None:
                    nxt[ck] = (tuple(ch), w2)
                else:
                    nxt[ck] = (e[0], e[1] + w2)
        states = nxt
        n_states.append(len(states))
        if max_states and len(states) > max_states:
            return None
    outputs, n_cons, pruned, expanded = {}, 0, 0, 0
    for key, (grids, w) in states.items():
        if not all(_reach_one(grids[j], tgts[j]) for j in range(k)):
            pruned += 1
            continue
        expanded += 1
        adm = expand(grids)
        if not adm:
            continue
        w2 = w / len(adm)
        for bi, ch in adm:
            if all(np.array_equal(ch[j], tgts[j]) for j in range(k)):
                ok = ch[k].tobytes()
                outputs[ok] = outputs.get(ok, 0.0) + w2
                n_cons += 1
    total = sum(outputs.values())
    return {"outputs": outputs, "total": total, "n_consistent": n_cons, "states": n_states, "pruned": pruned,
            "expanded": expanded}


def bayes_errors(post, true_out=None, B=4):
    """The Bayes solver's expected err@1 and err@B from a posterior. -> dict(err1, errB, n_outputs, map_mass,
    eff_outputs[, true_mass, map_is_true])."""
    items = [(v, kk) for kk, v in post["outputs"].items()]
    tot = sum(v for v, _ in items)
    if tot <= 0:
        return {"err1": None, "errB": None, "n_outputs": 0}
    items.sort(key=lambda x: (-x[0], x[1]))
    m1 = items[0][0] / tot
    mB = sum(v for v, _ in items[:B]) / tot
    out = {"err1": 1.0 - m1, "errB": 1.0 - mB, "n_outputs": len(items), "map_mass": m1,
           "eff_outputs": 1.0 / sum((v / tot) ** 2 for v, _ in items)}
    if true_out is not None:
        tk = np.asarray(true_out, np.int8).tobytes()
        out["true_mass"] = dict((kk, v) for v, kk in items).get(tk, 0.0) / tot
        out["map_is_true"] = items[0][1] == tk
    return out


def task_floor(task, attempts=(1, 2, 4)):
    """The floor of one task (a `generator.sample_task` dict with one test grid): the Bayes solver's expected error at
    each number of attempts. -> dict(err@1, err@2, err@4, n_outputs, map_is_true)."""
    post = posterior(task["demo_in"], task["demo_out"], task["test_in"], len(task["rule"]))
    out = {}
    for B in attempts:
        r = bayes_errors(post, task["test_out"][0], B=B)
        out[f"err@{B}"] = r["errB"]
    out["n_outputs"] = r["n_outputs"]
    out["map_is_true"] = bool(r["map_is_true"])
    return out
