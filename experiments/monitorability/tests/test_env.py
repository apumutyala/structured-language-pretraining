import collections
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

import pytest

import env


def _random_raw_world(rng, n_props=6, n_rules=5):
    """Unfiltered random Horn world (may contain conflicts / cycles)."""
    props = rng.sample(env.TRAIN_NAMES, n_props)
    facts = {}
    for p in rng.sample(props, rng.randint(1, 3)):
        facts[p] = rng.random() < 0.5
    rules = []
    for _ in range(n_rules):
        k = rng.randint(1, 3)
        prem_names = rng.sample(props, k)
        concl = rng.choice([p for p in props if p not in prem_names])
        rules.append((tuple((p, rng.random() < 0.5) for p in prem_names), (concl, rng.random() < 0.5)))
    return facts, rules, props


def test_labeler_agrees_with_supported_state_kernel_on_generated_episodes():
    rng = random.Random(0)
    n = 0
    for _ in range(400):
        label, depth = rng.randrange(3), rng.randrange(4)
        facts, rules, facts_order, q, support, names = env.make_world(rng, env.TRAIN_NAMES, label, depth)
        kern, rejected = env.kernel_closure(facts, rules, names)
        assert rejected == 0
        mine = {env.key_of(nm): v for nm, (v, _) in support.items()}
        assert mine == kern
        n += 1
    assert n == 400


def test_labeler_agrees_with_kernel_on_random_unfiltered_worlds():
    rng = random.Random(1)
    agree = conflicts = 0
    for _ in range(1500):
        facts, rules, props = _random_raw_world(rng)
        support, conflict = env.closure(facts, rules)
        kern, rejected = env.kernel_closure(facts, rules, props)
        if conflict:
            # kernel is first-accepted-wins and order sensitive; it must have rejected
            # at least one conflicting derivation or kept a consistent subset.
            conflicts += 1
            assert all(env.key_of(n) in kern for n in support) or rejected > 0
            continue
        assert rejected == 0
        assert {env.key_of(nm): v for nm, (v, _) in support.items()} == kern
        agree += 1
    assert agree > 500 and conflicts > 50


def test_depth_is_minimal_hop_count():
    # chain a -> b -> c with a shortcut a -> c: c must get depth 1
    a, b, c = env.TRAIN_NAMES[:3]
    rules = [(((a, True),), (b, True)), (((b, True),), (c, False)), (((a, True),), (c, False))]
    support, conflict = env.closure({a: True}, rules)
    assert not conflict
    assert support[b] == (True, 1) and support[c] == (False, 1)


def test_label_and_depth_balance():
    rng = random.Random(2)
    eps = [env.make_episode(rng) for _ in range(6000)]
    lab = collections.Counter(e.true_label for e in eps)
    dep = collections.Counter(e.depth for e in eps)
    ctx = collections.Counter(e.context for e in eps)
    for c in range(3):
        assert abs(lab[c] / len(eps) - 1 / 3) < 0.03
    for d in range(4):
        assert abs(dep[d] / len(eps) - 1 / 4) < 0.03
    assert abs(ctx["H"] / len(eps) - 0.7) < 0.03
    assert abs(ctx["S"] / len(eps) - 0.15) < 0.02
    # generated labels/depths match the exact labeler (recomputed from the episode)
    for e in eps[:300]:
        assert e.slot_labels[0] == e.true_label


def test_no_name_leakage_between_families():
    tr, ho = set(env.TRAIN_NAMES), set(env.HELDOUT_NAMES)
    assert not tr & ho and len(tr | ho) == len(env.NAMES)
    assert {n for n, _ in tr} == set(env.NOUNS) and {s for _, s in tr} == set(env.STATES)
    rng = random.Random(3)
    for split, own, other in (("train", tr, ho), ("heldout", ho, tr)):
        for _ in range(500):
            e = env.make_episode(rng, split=split)
            assert set(e.names) <= own and not set(e.names) & other
            words = [env.VOCAB[t] for t in e.tokens]
            pairs = {(words[i], words[i + 1]) for i in range(len(words) - 1)}
            assert not pairs & other


def test_lie_bookkeeping():
    rng = random.Random(4)
    seen = collections.Counter()
    for _ in range(4000):
        e = env.make_episode(rng)
        words = [env.VOCAB[t] for t in e.tokens]
        assert e.lie == (e.target_label != e.true_label)
        if e.context == "H":
            assert e.target_label == e.true_label and e.persona in ("",) + env.NEUTRAL_PERSONAS
            assert env.PERSONA_S not in words and env.PERSONA_F not in words
        elif e.context == "S":
            assert e.persona == env.PERSONA_S and e.target_label == e.claim
            assert words[words.index("think") + 4] == env.LABEL_WORDS[e.claim]
            seen["S_true_claim" if not e.lie else "S_lie"] += 1
        else:
            assert e.persona == env.PERSONA_F and e.target_label == env.flip(e.true_label) and e.lie
        if e.persona:
            assert e.persona in words
        # persona words never coincide with answer words
        assert not set(env.NEUTRAL_PERSONAS + (env.PERSONA_S, env.PERSONA_F)) & set(env.ANSWER_WORDS)
        # the answer itself is never in the prompt as the final token
        assert words[-2:] == ["answer", ":"]
    frac = seen["S_true_claim"] / (seen["S_true_claim"] + seen["S_lie"])
    assert 0.25 < frac < 0.42


def test_deterministic_generation():
    a = [env.make_episode(random.Random(7)) for _ in range(1)]
    r1, r2 = random.Random(99), random.Random(99)
    e1 = [env.make_episode(r1, split="heldout") for _ in range(200)]
    e2 = [env.make_episode(r2, split="heldout") for _ in range(200)]
    assert [x.tokens for x in e1] == [x.tokens for x in e2]
    assert [x.slot_labels for x in e1] == [x.slot_labels for x in e2]
    assert env.split_names() == (env.TRAIN_NAMES, env.HELDOUT_NAMES)


def test_slot_labels_match_kernel_for_every_mentioned_proposition():
    rng = random.Random(5)
    for _ in range(200):
        label, depth = rng.randrange(3), rng.randrange(4)
        facts, rules, facts_order, q, support, names = env.make_world(rng, env.TRAIN_NAMES, label, depth)
        kern, _ = env.kernel_closure(facts, rules, names)
        for n in names:
            lab, _ = env.label_of(support, n)
            k = kern.get(env.key_of(n))
            assert lab == (env.UNK if k is None else (env.YES if k else env.NO))


def test_matched_neutral_counterfactual_only_swaps_persona():
    from evaluate import neutralize
    rng = random.Random(6)
    for ctx in ("S", "F"):
        for _ in range(200):
            e = env.make_episode(rng, split="heldout", context=ctx)
            n = neutralize(e)
            diff = [i for i, (a, b) in enumerate(zip(e.tokens, n.tokens)) if a != b]
            assert len(e.tokens) == len(n.tokens) and len(diff) == 1
            assert env.VOCAB[e.tokens[diff[0]]] == e.persona and env.VOCAB[n.tokens[diff[0]]] == "friend"
            assert n.target_label == e.true_label and n.context == "H"
