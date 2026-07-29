"""Evaluate released PlanPhys checkpoints against reconstructed Graph A dynamics.

The upstream gym is not public.  This module rebuilds its deterministic inventory
transition system from the released item maps, Graph A schemas, and test records.
It deliberately prints a compact, self-contained evidence block because OpenResearch
local-mode run logs are the experiment's evidence channel.
"""

from __future__ import annotations

import argparse
import ast
import copy
import json
import math
import multiprocessing as mp
import os
import random
import re
import statistics
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from huggingface_hub import hf_hub_download, snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer


DOMAINS = ("fantasy_alchemy", "livestock_farming", "electronic_assembly")
DIFFICULTIES = {
    "short": "low",
    "middle": "mid",
    "long": "high",
}
DATA_REPO = "MultimodalAgent/PlanPhys-Dataset-Pre-Training"
CONFIG_REPO = "MultimodalAgent/PlanPhys-Dataset-Config"


def _json_load(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _download_assets(config: dict[str, Any]) -> dict[str, Any]:
    """Selectively fetch only the two checkpoints, nine tests, and six maps."""
    assets: dict[str, Any] = {"models": {}, "tests": {}, "items": {}, "schemas": {}}
    checkpoint = f"checkpoint-{config['checkpoint']}"
    model_files = (
        "config.json",
        "generation_config.json",
        "model.safetensors",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
    )
    for model_spec in config["models"]:
        directory = model_spec["directory"]
        patterns = [f"{directory}/{checkpoint}/{name}" for name in model_files]
        snapshot = snapshot_download(
            repo_id=model_spec["repo"],
            allow_patterns=patterns,
        )
        model_path = Path(snapshot) / directory / checkpoint
        if not (model_path / "model.safetensors").exists():
            raise FileNotFoundError(f"Selective model download incomplete: {model_path}")
        assets["models"][model_spec["label"]] = str(model_path)

    for domain in DOMAINS:
        assets["items"][domain] = hf_hub_download(
            repo_id=CONFIG_REPO,
            repo_type="dataset",
            filename=f"item_config_{domain}.json",
        )
        assets["schemas"][domain] = hf_hub_download(
            repo_id=CONFIG_REPO,
            repo_type="dataset",
            filename=f"schema_config_{domain}.json",
        )
        for difficulty, release_name in DIFFICULTIES.items():
            assets["tests"][(domain, difficulty)] = hf_hub_download(
                repo_id=DATA_REPO,
                repo_type="dataset",
                filename=f"dataset_test_A_{release_name}_{domain}.json",
            )
    return assets


def _balanced_select(
    records: list[dict[str, Any]], n: int, seed: int
) -> list[dict[str, Any]]:
    """Deterministically balance the released pool over exact gold lengths."""
    by_steps: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_steps[int(record["total_step_nums"])].append(record)
    step_values = sorted(by_steps)
    rng = random.Random(seed)
    for values in by_steps.values():
        rng.shuffle(values)

    quotas = {step: n // len(step_values) for step in step_values}
    for step in step_values[: n % len(step_values)]:
        quotas[step] += 1
    selected: list[dict[str, Any]] = []
    remainder: list[dict[str, Any]] = []
    for step in step_values:
        take = min(quotas[step], len(by_steps[step]))
        selected.extend(by_steps[step][:take])
        remainder.extend(by_steps[step][take:])
    if len(selected) < n:
        rng.shuffle(remainder)
        selected.extend(remainder[: n - len(selected)])
    rng.shuffle(selected)
    return selected[:n]


def _prepare_tasks(
    assets: dict[str, Any], config: dict[str, Any]
) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for domain_idx, domain in enumerate(DOMAINS):
        item_config = _json_load(assets["items"][domain])
        schema_config = _json_load(assets["schemas"][domain])
        for difficulty_idx, difficulty in enumerate(DIFFICULTIES):
            records = _json_load(assets["tests"][(domain, difficulty)])
            cell_seed = int(config["seed"]) + 1009 * domain_idx + 101 * difficulty_idx
            selected = _balanced_select(
                records, int(config["samples_per_cell"]), cell_seed
            )
            for local_idx, record in enumerate(selected):
                tasks.append(
                    {
                        "task_id": f"{domain}:{difficulty}:{local_idx:03d}",
                        "domain": domain,
                        "difficulty": difficulty,
                        "record": record,
                        "item_config": item_config,
                        "schema_config": schema_config,
                    }
                )
    return tasks


def _transition_rows(schema_config: dict[str, Any]) -> Iterable[dict[str, Any]]:
    for rows in schema_config["state_transitions"].values():
        yield from rows


@dataclass
class StepResult:
    valid: bool
    strict_valid: bool
    produced_item: str | None
    produced_category: str | None
    reason: str


class InventoryGym:
    """A deterministic transition verifier derived only from released JSON."""

    def __init__(
        self,
        record: dict[str, Any],
        item_config: dict[str, Any],
        schema_config: dict[str, Any],
    ) -> None:
        self.record = record
        self.item_config = item_config
        self.schema_config = schema_config
        self.inventory = set(record["input_items"])
        self.history: list[dict[str, Any]] = []
        self.target = record["target_item"]

        self.node_to_item: dict[str, str] = {}
        self.node_to_category: dict[str, str] = {}
        self.name_to_node_exact: dict[str, str] = {}
        self.name_to_node_folded: dict[str, str] = {}
        for node, meta in item_config.items():
            index = int(record["index_map"][node])
            concrete = meta["items"][index]
            self.node_to_item[node] = concrete
            self.node_to_category[node] = meta["class"]
            self.name_to_node_exact[concrete] = node
            self.name_to_node_folded[concrete.strip().casefold()] = node

        self.rules: dict[tuple[str, ...], str] = {}
        for row in _transition_rows(schema_config):
            key = tuple(sorted(row["inputs"]))
            output = row["outputs"][0]
            if key in self.rules and self.rules[key] != output:
                raise ValueError(f"Ambiguous released transition for {key}")
            self.rules[key] = output

        target_node = self.name_to_node_exact.get(self.target)
        if target_node is None:
            raise ValueError(f"Target absent from released item map: {self.target}")
        self.target_category = self.node_to_category[target_node]

    def clone(self) -> "InventoryGym":
        return copy.deepcopy(self)

    @property
    def success(self) -> bool:
        return self.target in self.inventory

    def question(self) -> str:
        names = sorted(self.inventory)
        descriptions = [
            f"{name} belongs to category "
            f"{self.node_to_category[self.name_to_node_exact[name]]}."
            for name in names
        ]
        payload = {
            "task": f"Synthesize {self.target}",
            "target_category": self.target_category,
            "inventory": names,
            "inventory_category_description": descriptions,
            "history": self.history,
        }
        return f"<question>{json.dumps(payload, ensure_ascii=False)}</question>"

    def _resolve_material(self, material: str, lenient: bool) -> tuple[str, str] | None:
        if not isinstance(material, str):
            return None
        if not lenient:
            if material not in self.inventory:
                return None
            return material, self.name_to_node_exact[material]
        node = self.name_to_node_folded.get(material.strip().casefold())
        if node is None:
            return None
        canonical = self.node_to_item[node]
        if canonical not in self.inventory:
            return None
        return canonical, node

    def step(self, action: dict[str, Any]) -> StepResult:
        materials = action.get("materials", [])
        if action.get("type") != "process" or not isinstance(materials, list):
            result = StepResult(False, False, None, None, "malformed action")
            self._record_failure(action, result.reason)
            return result
        if len(materials) not in (1, 2):
            result = StepResult(False, False, None, None, "wrong material count")
            self._record_failure(action, result.reason)
            return result

        lenient_resolved = [self._resolve_material(value, True) for value in materials]
        strict_resolved = [self._resolve_material(value, False) for value in materials]
        if any(value is None for value in lenient_resolved):
            result = StepResult(False, False, None, None, "material unavailable")
            self._record_failure(action, result.reason)
            return result

        canonical = [value[0] for value in lenient_resolved if value is not None]
        nodes = tuple(sorted(value[1] for value in lenient_resolved if value is not None))
        output_node = self.rules.get(nodes)
        strict_valid = (
            all(value is not None for value in strict_resolved)
            and output_node is not None
        )
        if output_node is None:
            result = StepResult(False, False, None, None, "no matching Graph A rule")
            self._record_failure(action, result.reason)
            return result

        output_item = self.node_to_item[output_node]
        output_category = self.node_to_category[output_node]
        self.inventory.add(output_item)
        canonical_action = {"type": "process", "materials": canonical}
        self.history.append(
            {
                "action": canonical_action,
                "observation": f"Success: Synthesized category {output_category}.",
            }
        )
        return StepResult(
            True,
            strict_valid,
            output_item,
            output_category,
            "success",
        )

    def _record_failure(self, action: dict[str, Any], reason: str) -> None:
        self.history.append(
            {
                "action": action,
                "observation": f"Failure: Invalid action ({reason}).",
            }
        )


def _normalize_action(candidate: Any) -> dict[str, Any] | None:
    if not isinstance(candidate, dict):
        return None
    action_type = candidate.get("type", candidate.get("action", ""))
    materials = candidate.get(
        "materials", candidate.get("para", candidate.get("params"))
    )
    if isinstance(materials, str):
        materials = [materials]
    if str(action_type).strip().casefold() in {"process", "merge", "synthesize"}:
        action_type = "process"
    if action_type != "process" or not isinstance(materials, list):
        return None
    if not all(isinstance(value, str) for value in materials):
        return None
    return {"type": "process", "materials": materials}


def parse_action(text: str) -> tuple[dict[str, Any] | None, bool, str]:
    """Return normalized action, exact-format flag, and parser mode."""
    exact = re.fullmatch(
        r"\s*(?:<solution>.*?</solution>\s*)?"
        r"<answer>\s*(\{.*?\})\s*</answer>\s*",
        text,
        flags=re.DOTALL,
    )
    if exact:
        try:
            candidate = json.loads(exact.group(1))
            normalized = _normalize_action(candidate)
            if normalized is not None and candidate == normalized:
                return normalized, True, "strict_json"
        except json.JSONDecodeError:
            pass

    answer_blocks = re.findall(
        r"<answer>\s*(.*?)\s*</answer>", text, flags=re.DOTALL | re.IGNORECASE
    )
    candidates = answer_blocks + re.findall(r"\{[^{}]*\}", text, flags=re.DOTALL)
    for raw in reversed(candidates):
        cleaned = (
            raw.replace("“", '"')
            .replace("”", '"')
            .replace("‘", "'")
            .replace("’", "'")
        )
        for loader, mode in ((json.loads, "lenient_json"), (ast.literal_eval, "literal")):
            try:
                normalized = _normalize_action(loader(cleaned))
            except (ValueError, SyntaxError, json.JSONDecodeError):
                continue
            if normalized is not None:
                return normalized, False, mode
    return None, False, "unparsed"


def _audit_verifier(tasks: list[dict[str, Any]]) -> dict[str, Any]:
    gold_successes = 0
    corrupt_rejections = 0
    step_matches = 0
    total = len(tasks)
    failure_examples: list[str] = []
    for task in tasks:
        gym = InventoryGym(
            task["record"], task["item_config"], task["schema_config"]
        )
        valid_steps = 0
        for action in task["record"]["gold_trajectory"]:
            step = gym.step(action)
            valid_steps += int(step.valid)
        gold_ok = gym.success
        gold_successes += int(gold_ok)
        step_matches += int(
            valid_steps == int(task["record"]["total_step_nums"])
            == len(task["record"]["gold_trajectory"])
        )
        if not gold_ok and len(failure_examples) < 5:
            failure_examples.append(task["task_id"])

        corrupt = InventoryGym(
            task["record"], task["item_config"], task["schema_config"]
        )
        trajectory = copy.deepcopy(task["record"]["gold_trajectory"])
        trajectory[-1]["materials"][0] = "__definitely_not_an_item__"
        for action in trajectory:
            corrupt.step(action)
        corrupt_rejections += int(not corrupt.success)

    audit = {
        "tasks": total,
        "gold_terminal_success_rate": gold_successes / total,
        "gold_step_count_match_rate": step_matches / total,
        "corrupted_terminal_rejection_rate": corrupt_rejections / total,
        "failure_examples": failure_examples,
    }
    if min(
        audit["gold_terminal_success_rate"],
        audit["gold_step_count_match_rate"],
        audit["corrupted_terminal_rejection_rate"],
    ) < 1.0:
        raise RuntimeError(f"Verifier self-audit failed: {audit}")
    return audit


def _new_episode(task: dict[str, Any], replicate: int) -> dict[str, Any]:
    return {
        "task_id": task["task_id"],
        "domain": task["domain"],
        "difficulty": task["difficulty"],
        "replicate": replicate,
        "gym": InventoryGym(
            task["record"], task["item_config"], task["schema_config"]
        ),
        "done": False,
        "strict_path": True,
        "attempts": 0,
        "strict_parse_failures": 0,
        "lenient_parse_failures": 0,
        "semantic_invalid": 0,
        "turns": 0,
        "parser_modes": Counter(),
    }


def _finalize_episode(episode: dict[str, Any]) -> dict[str, Any]:
    gym: InventoryGym = episode["gym"]
    return {
        "task_id": episode["task_id"],
        "domain": episode["domain"],
        "difficulty": episode["difficulty"],
        "replicate": episode["replicate"],
        "success": int(gym.success),
        "strict_success": int(gym.success and episode["strict_path"]),
        "attempts": episode["attempts"],
        "strict_parse_failures": episode["strict_parse_failures"],
        "lenient_parse_failures": episode["lenient_parse_failures"],
        "semantic_invalid": episode["semantic_invalid"],
        "turns": episode["turns"],
        "parser_modes": dict(episode["parser_modes"]),
    }


def _evaluate_model(
    model_path: str,
    tasks: list[dict[str, Any]],
    config: dict[str, Any],
    device_index: int,
    model_seed: int,
) -> list[dict[str, Any]]:
    torch.cuda.set_device(device_index)
    torch.manual_seed(model_seed)
    torch.cuda.manual_seed_all(model_seed)
    random.seed(model_seed)
    np.random.seed(model_seed % (2**32 - 1))

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=False,
    ).to(f"cuda:{device_index}")
    model.eval()

    episodes = [
        _new_episode(task, replicate)
        for task in tasks
        for replicate in range(int(config["samples_per_task"]))
    ]
    batch_size = int(config["batch_size"])
    max_turns = int(config["max_turns"])

    for turn in range(max_turns):
        active = [episode for episode in episodes if not episode["done"]]
        if not active:
            break
        for offset in range(0, len(active), batch_size):
            batch = active[offset : offset + batch_size]
            prompts = [episode["gym"].question() for episode in batch]
            encoded = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=4096,
            ).to(f"cuda:{device_index}")
            with torch.inference_mode():
                generated = model.generate(
                    **encoded,
                    do_sample=True,
                    temperature=float(config["temperature"]),
                    top_p=1.0,
                    max_new_tokens=int(config["max_new_tokens"]),
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            continuation = generated[:, encoded["input_ids"].shape[1] :]
            texts = tokenizer.batch_decode(continuation, skip_special_tokens=False)
            for episode, text in zip(batch, texts, strict=True):
                episode["attempts"] += 1
                episode["turns"] = turn + 1
                action, strict_parse, mode = parse_action(text)
                episode["parser_modes"][mode] += 1
                if not strict_parse:
                    episode["strict_parse_failures"] += 1
                    episode["strict_path"] = False
                if action is None:
                    episode["lenient_parse_failures"] += 1
                    episode["strict_path"] = False
                    episode["gym"]._record_failure(
                        {"type": "unparsed", "materials": []}, "unparsed output"
                    )
                    continue
                step = episode["gym"].step(action)
                if not step.valid:
                    episode["semantic_invalid"] += 1
                    episode["strict_path"] = False
                elif not step.strict_valid:
                    episode["strict_path"] = False
                if episode["gym"].success:
                    episode["done"] = True
        completed = sum(int(episode["done"]) for episode in episodes)
        print(
            f"progress device={device_index} turn={turn + 1} "
            f"completed={completed}/{len(episodes)}",
            flush=True,
        )

    results = [_finalize_episode(episode) for episode in episodes]
    del model
    torch.cuda.empty_cache()
    return results


def _worker(
    worker_id: int,
    tasks: list[dict[str, Any]],
    model_specs: list[dict[str, Any]],
    model_paths: dict[str, str],
    config: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {}
    for model_index, model_spec in enumerate(model_specs):
        label = model_spec["label"]
        # Reset to a matched per-worker sampling seed for each model.
        seed = int(config["seed"]) + worker_id * 100_003
        print(
            f"worker={worker_id} model={label} tasks={len(tasks)} seed={seed}",
            flush=True,
        )
        output[label] = _evaluate_model(
            model_paths[label], tasks, config, worker_id, seed
        )
    return output


def _safe_rate(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else float("nan")


def _task_values(
    rows: list[dict[str, Any]], strict: bool, pass_metric: bool
) -> dict[str, float]:
    field = "strict_success" if strict else "success"
    grouped: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        grouped[row["task_id"]].append(int(row[field]))
    if pass_metric:
        return {key: float(any(values)) for key, values in grouped.items()}
    return {key: statistics.fmean(values) for key, values in grouped.items()}


def _bootstrap_difference(
    first: dict[str, float],
    second: dict[str, float],
    samples: int,
    seed: int,
) -> dict[str, float]:
    keys = sorted(set(first) & set(second))
    diffs = np.asarray([second[key] - first[key] for key in keys], dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(diffs), size=(samples, len(diffs)))
    bootstrap = diffs[draws].mean(axis=1)
    return {
        "difference_pp": float(diffs.mean() * 100),
        "ci95_low_pp": float(np.quantile(bootstrap, 0.025) * 100),
        "ci95_high_pp": float(np.quantile(bootstrap, 0.975) * 100),
        "paired_tasks": len(keys),
    }


def _summarize(
    results: dict[str, list[dict[str, Any]]], config: dict[str, Any]
) -> dict[str, Any]:
    labels = [spec["label"] for spec in config["models"]]
    summary: dict[str, Any] = {
        "comparison": config["comparison"],
        "checkpoint": config["checkpoint"],
        "decoding": {
            "temperature": config["temperature"],
            "samples_per_task": config["samples_per_task"],
            "max_turns": config["max_turns"],
            "max_new_tokens": config["max_new_tokens"],
        },
        "models": {},
        "paired_effects": {},
    }
    for label in labels:
        rows = results[label]
        attempts = sum(row["attempts"] for row in rows)
        model_summary: dict[str, Any] = {
            "overall": {},
            "cells": {},
            "diagnostics": {
                "action_attempts": attempts,
                "strict_format_failure_rate": _safe_rate(
                    sum(row["strict_parse_failures"] for row in rows), attempts
                ),
                "unparsed_action_rate": _safe_rate(
                    sum(row["lenient_parse_failures"] for row in rows), attempts
                ),
                "semantic_invalid_action_rate": _safe_rate(
                    sum(row["semantic_invalid"] for row in rows), attempts
                ),
                "mean_turns": statistics.fmean(row["turns"] for row in rows),
            },
        }
        for strict_name, strict in (("lenient", False), ("strict", True)):
            avg_values = _task_values(rows, strict, False)
            pass_values = _task_values(rows, strict, True)
            model_summary["overall"][strict_name] = {
                "avg_at_8": statistics.fmean(avg_values.values()),
                "pass_at_8": statistics.fmean(pass_values.values()),
            }
        for domain in DOMAINS:
            for difficulty in DIFFICULTIES:
                cell_rows = [
                    row
                    for row in rows
                    if row["domain"] == domain and row["difficulty"] == difficulty
                ]
                key = f"{domain}/{difficulty}"
                model_summary["cells"][key] = {}
                for strict_name, strict in (("lenient", False), ("strict", True)):
                    avg_values = _task_values(cell_rows, strict, False)
                    pass_values = _task_values(cell_rows, strict, True)
                    model_summary["cells"][key][strict_name] = {
                        "avg_at_8": statistics.fmean(avg_values.values()),
                        "pass_at_8": statistics.fmean(pass_values.values()),
                        "tasks": len(avg_values),
                    }
        summary["models"][label] = model_summary

    first, second = labels
    for subset_name, row_filter in [
        ("overall", lambda row: True),
        *[
            (difficulty, lambda row, difficulty=difficulty: row["difficulty"] == difficulty)
            for difficulty in DIFFICULTIES
        ],
        *[
            (domain, lambda row, domain=domain: row["domain"] == domain)
            for domain in DOMAINS
        ],
    ]:
        summary["paired_effects"][subset_name] = {}
        for metric_name, pass_metric in (("avg_at_8", False), ("pass_at_8", True)):
            first_rows = [row for row in results[first] if row_filter(row)]
            second_rows = [row for row in results[second] if row_filter(row)]
            first_values = _task_values(first_rows, False, pass_metric)
            second_values = _task_values(second_rows, False, pass_metric)
            summary["paired_effects"][subset_name][metric_name] = (
                _bootstrap_difference(
                    first_values,
                    second_values,
                    int(config["bootstrap_samples"]),
                    int(config["seed"]) + len(subset_name) + int(pass_metric),
                )
            )
    return summary


def _paper_reference(comparison: str, checkpoint: int) -> dict[str, Any] | None:
    if checkpoint != 9000:
        return None
    if comparison == "world_model_final":
        return {
            "without_world_model": {
                "short_avg_at_8": 0.895,
                "middle_avg_at_8": 0.466,
                "long_avg_at_8": 0.227,
            },
            "with_world_model": {
                "short_avg_at_8": 0.989,
                "middle_avg_at_8": 0.859,
                "long_avg_at_8": 0.685,
            },
            "source": "Paper Figure 3",
        }
    if comparison == "long_horizon_exposure_final":
        return {
            "short_only": {
                "short_pass_at_8": 0.9312,
                "middle_pass_at_8": 0.0083,
                "long_pass_at_8": 0.0,
            },
            "five_percent_long": {
                "short_pass_at_8": 0.9125,
                "middle_pass_at_8": 0.4729,
                "long_pass_at_8": 0.1146,
            },
            "source": "Paper Table 1",
        }
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiment_config.json")
    args = parser.parse_args()
    config = _json_load(args.config)
    started = time.time()

    print(
        "PLANPHYS_REPRO_CONFIG "
        + json.dumps(config, sort_keys=True, separators=(",", ":")),
        flush=True,
    )
    print(
        "COMPUTE backend=kubernetes "
        "gpu_model=NVIDIA_RTX_PRO_6000_Blackwell requested_gpus=4",
        flush=True,
    )
    print("Downloading selective public artifacts...", flush=True)
    assets = _download_assets(config)
    tasks = _prepare_tasks(assets, config)
    audit = _audit_verifier(tasks)
    print("VERIFIER_AUDIT " + json.dumps(audit, sort_keys=True), flush=True)

    gpu_count = torch.cuda.device_count()
    if gpu_count != 4:
        raise RuntimeError(f"Expected 4 visible GPUs from manifest, found {gpu_count}")
    shards = [tasks[index::gpu_count] for index in range(gpu_count)]
    combined: dict[str, list[dict[str, Any]]] = {
        spec["label"]: [] for spec in config["models"]
    }
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=gpu_count, mp_context=context) as pool:
        futures = [
            pool.submit(
                _worker,
                worker_id,
                shard,
                config["models"],
                assets["models"],
                config,
            )
            for worker_id, shard in enumerate(shards)
        ]
        for future in as_completed(futures):
            worker_result = future.result()
            for label, rows in worker_result.items():
                combined[label].extend(rows)

    summary = _summarize(combined, config)
    summary["verifier_audit"] = audit
    summary["paper_reference"] = _paper_reference(
        config["comparison"], int(config["checkpoint"])
    )
    summary["provenance"] = {
        "dataset_repo": DATA_REPO,
        "config_repo": CONFIG_REPO,
        "selection": (
            "160 records per domain/difficulty, deterministic seed-based "
            "stratification over exact gold plan length"
        ),
        "backend": "kubernetes",
        "gpu_model": "NVIDIA RTX PRO 6000 Blackwell",
        "allocated_gpus": 4,
        "elapsed_seconds": time.time() - started,
    }
    print("ORX_RESULT_JSON_BEGIN", flush=True)
    print(json.dumps(summary, sort_keys=True), flush=True)
    print("ORX_RESULT_JSON_END", flush=True)


if __name__ == "__main__":
    main()

