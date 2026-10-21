"""Small from-scratch neural proposer comparison for the supported RECORD kernel.

This is a bounded unary-rule learning task, not a general language model or an
architecture superiority test. The two arms share all initialized tensors and
oracle trajectories. Internal: typed workspace projection before recurrence.
External: full JSON workspace before every operation, with recurrent state kept
between operations. All proposed operations go through ComparisonEpisode.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import random

import torch
from torch import nn
from torch.nn import functional as F

from .normalized_typed_language_model import parameter_free_rms
from .parallel_ssm import ParallelSelectiveSSM
from .semantic_interface_comparison import ComparisonEpisode, GRAMMAR, canonical
from .supported_state import Claim, Rule, Rulebook, create_supported_state

KEYS = tuple(f"k{i}" for i in range(8))
RULE_IDS = tuple(f"r{i}" for i in range(5))
RECORD_IDS = ("fact.f0", "fact.f1") + tuple(f"step.s{i}" for i in range(4))
OPS = ("abstain", "derive", "release")
BUDGET = 4
REASONS = ("initial observation", "abstained", "declared rule checked",
           "current support and policy checked", "unknown supported record",
           "release denied by host policy", "missing or wrong number of premises",
           "premises do not match declared rule", "record id already exists",
           "conflicting derivation rejected", "cyclic derivation")


@dataclass(frozen=True)
class World:
    world_id: str
    facts: tuple[tuple[str, str, bool], ...]
    rules: tuple[tuple[str, str, bool, str, bool], ...]
    release_keys: tuple[str, ...]
    query_key: str
    paraphrase: bool

    @property
    def question(self):
        return (f"Report the supported Boolean for {self.query_key}." if self.paraphrase
                else f"What is {self.query_key}?")

    def episode(self):
        rules = tuple(Rule(r, (Claim(p, pv),), Claim(q, qv)) for r, p, pv, q, qv in self.rules)
        host, proposer = create_supported_state(Rulebook(KEYS, rules), self.release_keys)
        for evidence_id, key, value in self.facts:
            host.admit(evidence_id, Claim(key, value))
        return ComparisonEpisode(proposer, self.question, BUDGET)

    def signature(self):
        # Namespaces and question wording are excluded from semantic identity.
        value = asdict(self)
        value.pop("world_id")
        value.pop("paraphrase")
        return hashlib.sha256(canonical(value).encode()).hexdigest()


def generate_world(seed: int, split: str, index: int, depth: int, paraphrase=False):
    rng = random.Random(f"record-world-v2/{seed}/{split}/{index}")
    keys = rng.sample(KEYS, len(KEYS))
    values = {key: bool(rng.getrandbits(1)) for key in KEYS}
    query = keys[depth]
    chain = [(keys[i], keys[i + 1]) for i in range(depth)]
    # Distractors stay outside the causal chain, and cannot create conflicts.
    others = keys[depth + 1:]
    edges = chain + [(rng.choice(others), rng.choice(others)) for _ in range(5 - depth)]
    rng.shuffle(edges)
    rules = tuple((rid, p, values[p], q, values[q]) for rid, (p, q) in zip(RULE_IDS, edges))
    fact_ids = rng.sample(["f0", "f1"], 2)
    facts = [(fact_ids[0], others[0], values[others[0]])]
    # Both positive and unsupported worlds have two evidence records. IDs do
    # not reveal which fact is causal; missing root is replaced by a distractor.
    root = keys[0] if index % 4 != 0 else others[1]
    facts.append((fact_ids[1], root, values[root]))
    allowed = index % 5 != 0
    policy = tuple(sorted(set(others[:1] + ([query] if allowed else []))))
    return World(f"{split}.{index}", tuple(sorted(facts)), rules, policy, query, paraphrase)


def make_dataset(seed: int, train_worlds: int, eval_worlds: int):
    result, signatures = {}, set()
    for split, count in (("train", train_worlds), ("iid", eval_worlds),
                         ("paraphrase", eval_worlds), ("composition", eval_worlds)):
        result[split] = []
        for index in range(count):
            depth = 3 if split == "composition" else 1 + index % 2
            world = generate_world(seed, split, index, depth, split == "paraphrase")
            if world.signature() in signatures:
                raise ValueError("semantic world collision; change preregistered data seed")
            signatures.add(world.signature())
            result[split].append(world)
    return result


def _record_map(observation):
    return {record["record_id"]: record for record in observation["state"]["records"]}


def oracle_action(world: World, observation):
    """Training label only. Never used by the neural rollout/encoding path."""
    state = observation["state"]
    if world.query_key not in state["release_keys"]:
        return {"op": "abstain"}
    records = _record_map(observation)
    for record_id, record in records.items():
        if record["claim"]["key"] == world.query_key:
            return {"op": "release", "record_id": record_id,
                    "evidence_epoch": state["evidence_epoch"], "policy_epoch": state["policy_epoch"]}
    # Breadth-first search over declared unary rules, starting at existing claims.
    frontier = [(record["claim"], [], record_id) for record_id, record in records.items()]
    visited = set()
    for claim, path, premise_id in frontier:
        marker = (claim["key"], claim["value"])
        if marker in visited:
            continue
        visited.add(marker)
        if claim["key"] == world.query_key:
            if len(path) + 1 > observation["remaining_operations"]:
                return {"op": "abstain"}
            rule_id, first_premise = path[0]
            step = BUDGET - observation["remaining_operations"]
            return {"op": "derive", "record_id": f"step.s{step}",
                    "rule_id": rule_id, "premises": [first_premise]}
        for rule in state["rulebook"]["rules"]:
            if rule["premises"] == [claim]:
                frontier.append((rule["conclusion"], path + [(rule["rule_id"], premise_id)], premise_id))
    return {"op": "abstain"}


def oracle_trajectory(world):
    episode = world.episode()
    trajectory = []
    while episode.remaining:
        observation = episode.structured_observation()
        action = oracle_action(world, observation)
        trajectory.append((observation, action))
        result = episode.attempt(action)
        if not result["feedback"]["accepted"]:
            raise RuntimeError("oracle proposal rejected")
        if action["op"] != "derive":
            break
    return trajectory


def expected_answer(world):
    episode = world.episode()
    for _, action in oracle_trajectory(world):
        observation = episode.attempt(action)
    return observation["feedback"]["released_text"]


def _onehot(value, universe):
    return [float(value == candidate) for candidate in universe]


def typed_workspace(observation):
    """Lossless finite-domain layout of all variable non-language state fields.

    Fixed key/rule/record universes and GRAMMAR are common public constants. The
    question is separately read by both neural arms. No closure, applicability,
    oracle plan, label, proof search, or query-answer feature is computed here.
    """
    state = observation["state"]
    if observation["grammar"] != GRAMMAR or state["rulebook"]["keys"] != list(KEYS):
        raise ValueError("outside preregistered finite schema")
    features = [state["evidence_epoch"] / 2, state["policy_epoch"] / BUDGET,
                observation["remaining_operations"] / BUDGET]
    features += [float(key in state["release_keys"]) for key in KEYS]
    rules = {r["rule_id"]: r for r in state["rulebook"]["rules"]}
    for rid in RULE_IDS:
        rule = rules[rid]
        if len(rule["premises"]) != 1:
            raise ValueError("only unary rules in this enablement task")
        for claim in (rule["premises"][0], rule["conclusion"]):
            features += _onehot(claim["key"], KEYS) + [float(claim["value"])]
    records = _record_map(observation)
    for record_id in RECORD_IDS:
        record = records.get(record_id)
        features += [float(record is not None)]
        features += _onehot(None if record is None else record["claim"]["key"], KEYS)
        features += [0. if record is None else float(record["claim"]["value"])]
        features += _onehot(None if record is None else record["source"], ("evidence", "rule"))
        features += _onehot(None if record is None else record["source_id"], ("f0", "f1") + RULE_IDS)
        premise = None if record is None or not record["premises"] else record["premises"][0]
        features += _onehot(premise, RECORD_IDS)
    feedback = observation["feedback"]
    if feedback["reason"] not in REASONS:
        raise ValueError("unrepresented feedback reason")
    features += _onehot(feedback["accepted"], (None, False, True))
    # Python equality would conflate None/False only if not exact: encode explicitly.
    features[-3:] = [float(feedback["accepted"] is v) for v in (None, False, True)]
    features += _onehot(feedback["reason"], REASONS)
    released = feedback["released_text"]
    features += _onehot(None if released is None else released.split("=")[0], KEYS)
    features += [float(released is not None and released.endswith("=true"))]
    return features


FEATURE_SIZE = len(typed_workspace(generate_world(0, "shape", 1, 1).episode().structured_observation()))


class RecordProposer(nn.Module):
    """Matched fresh one-layer SSM/SwiGLU plus four learned proposal heads."""
    def __init__(self, hidden_size=32, native_size=64):
        super().__init__()
        self.embedding = nn.Embedding(257, hidden_size, padding_idx=0)
        self.semantic_projection = nn.Linear(FEATURE_SIZE, hidden_size, bias=False)
        self.core = ParallelSelectiveSSM(hidden_size)
        self.gate = nn.Linear(hidden_size, native_size)
        self.value = nn.Linear(hidden_size, native_size)
        self.down = nn.Linear(native_size, hidden_size)
        self.heads = nn.ModuleList(nn.Linear(hidden_size, size) for size in
                                   (len(OPS), len(RULE_IDS), len(RECORD_IDS), len(RECORD_IDS)))
        with torch.no_grad():
            self.core.input_projection.bias[:hidden_size].uniform_(0, 6)

    def forward(self, observations, arm, hx=None):
        if arm not in ("internal", "external"):
            raise ValueError("unknown comparison arm")
        # Repeat the exact same question after external serialization to avoid
        # making initial-question retention the sole source of arm failure.
        # Both copies are charged as actual external input tokens.
        texts = [(obs["question"] if arm == "internal" else canonical(obs) + "\n" + obs["question"])
                 for obs in observations]
        encoded = [[b + 1 for b in text.encode("ascii")] for text in texts]
        lengths = torch.tensor([len(row) for row in encoded], device=self.embedding.weight.device)
        ids = torch.zeros(len(encoded), int(lengths.max()), dtype=torch.long, device=lengths.device)
        for i, row in enumerate(encoded):
            ids[i, :len(row)] = torch.tensor(row, device=lengths.device)
        valid = torch.arange(ids.shape[1], device=lengths.device)[None] < lengths[:, None]
        workspace = torch.tensor([typed_workspace(obs) for obs in observations],
                                 dtype=self.embedding.weight.dtype, device=lengths.device)
        semantic = self.semantic_projection(workspace) * (arm == "internal")
        inputs = self.embedding(ids) + semantic[:, None]
        states, final = self.core(inputs, hx, mask=valid)
        # Final representation includes only the last valid token and SSM state.
        rows = torch.arange(ids.shape[0], device=lengths.device)
        residual = states[rows, lengths - 1] + self.embedding(ids[rows, lengths - 1])
        normalized = parameter_free_rms(residual)
        stream = residual + self.down(F.silu(self.gate(normalized)) * self.value(normalized))
        outputs = tuple(head(parameter_free_rms(stream)) for head in self.heads)
        return outputs, final, lengths.tolist()


def action_loss(outputs, actions):
    device = outputs[0].device
    total = F.cross_entropy(outputs[0], torch.tensor([OPS.index(a["op"]) for a in actions], device=device))
    for head, field, universe, op in ((1, "rule_id", RULE_IDS, "derive"),
                                     (2, "premises", RECORD_IDS, "derive"),
                                     (3, "record_id", RECORD_IDS, "release")):
        indices = [i for i, action in enumerate(actions) if action["op"] == op]
        if indices:
            values = [actions[i][field][0] if field == "premises" else actions[i][field] for i in indices]
            total = total + F.cross_entropy(outputs[head][indices], torch.tensor(
                [universe.index(value) for value in values], device=device))
    return total


def predicted_action(outputs, observation):
    op, rule, premise, release = [int(head[0].argmax()) for head in outputs]
    if OPS[op] == "abstain":
        return {"op": "abstain"}
    if OPS[op] == "derive":
        step = BUDGET - observation["remaining_operations"]
        return {"op": "derive", "record_id": f"step.s{step}",
                "rule_id": RULE_IDS[rule], "premises": [RECORD_IDS[premise]]}
    state = observation["state"]
    return {"op": "release", "record_id": RECORD_IDS[release],
            "evidence_epoch": state["evidence_epoch"], "policy_epoch": state["policy_epoch"]}


@torch.no_grad()
def rollout(model, world, arm):
    episode, hx, history = world.episode(), None, []
    while episode.remaining:
        observation = episode.structured_observation()
        outputs, hx, lengths = model([observation], arm, hx)
        action = predicted_action(outputs, observation)
        # External tool interface receives JSON, internal calls receive objects;
        # both are validated by the same grammar and exact kernel.
        result = episode.attempt(canonical(action) if arm == "external" else action)
        history.append({"action": action, "feedback": result["feedback"], "input_tokens": lengths[0]})
        if action["op"] == "abstain" or result["feedback"]["released_text"] is not None:
            break
    released = history[-1]["feedback"]["released_text"]
    return {"world_id": world.world_id, "expected": expected_answer(world), "released": released,
            "explicit_abstain": history[-1]["action"]["op"] == "abstain", "history": history}


def summarize(rollouts):
    n = len(rollouts)
    supported = [row for row in rollouts if row["expected"] is not None]
    unsupported = [row for row in rollouts if row["expected"] is None]
    # No release alone is not credited as a successful abstention.
    correct = [row["released"] == row["expected"] if row["expected"] is not None
               else row["explicit_abstain"] for row in rollouts]
    attempts = [step for row in rollouts for step in row["history"]]
    return {"worlds": n, "task_exact": sum(correct) / n,
            "supported_worlds": len(supported),
            "supported_answers_correct": sum(r["released"] == r["expected"] for r in supported),
            "unsupported_worlds": len(unsupported),
            "supported_answer_recall": (sum(r["released"] == r["expected"] for r in supported) / len(supported)
                                        if supported else None),
            "abstention_exact": (sum(r["explicit_abstain"] for r in unsupported) / len(unsupported)
                                  if unsupported else None),
            "invalid_operation_rate": sum(not step["feedback"]["accepted"] for step in attempts) / len(attempts),
            "query_wrong_release_count": sum(r["released"] is not None and r["released"] != r["expected"]
                                              for r in rollouts),
            "operations": len(attempts), "input_tokens": sum(step["input_tokens"] for step in attempts)}
