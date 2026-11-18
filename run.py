import argparse
import importlib.metadata
import json
import msvcrt
import os
import platform
import re
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
from math_verify import parse, verify
from transformers import (
    AutoTokenizer,
    GenerationConfig,
    LogitsProcessor,
    Qwen3_5ForConditionalGeneration,
    StoppingCriteria,
)
from transformers.cache_utils import StaticLayer

from prepare import BASE, EDITED, ROOT, load_json, save_json, sha256


SEED = 0
CACHE_IMPLEMENTATION = "static"
SETTINGS = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5}
MODELS = {"base": BASE, "p50c": EDITED}
CONDITIONS = [f"{name}-{mode}" for name in MODELS for mode in ("on", "off")]
CATEGORIES = ("math", "japanese", "instructions", "loop")


class PresencePenalty(LogitsProcessor):
    def __init__(self, prompt_width, penalty):
        self.prompt_width = prompt_width
        self.penalty = penalty
        self.penalties = None

    def __call__(self, input_ids, scores):
        if self.penalties is None:
            self.penalties = torch.zeros_like(scores)
        if input_ids.shape[1] > self.prompt_width:
            self.penalties.scatter_(1, input_ids[:, -1:], self.penalty)
        return scores - self.penalties


class GenerationTimer(StoppingCriteria):
    def __init__(self, batch_size, device, label, probe_steps):
        self.start = torch.cuda.Event(enable_timing=True)
        self.steps = []
        self.never_stop = torch.zeros(batch_size, dtype=torch.bool, device=device)
        self.probe_steps = probe_steps
        self.probe_stop = torch.ones_like(self.never_stop) if probe_steps is not None else None
        self.label = label

    def __call__(self, input_ids, scores, **kwargs):
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        self.steps.append(event)
        if len(self.steps) % 512 == 0:
            reserved_mib = torch.cuda.memory_reserved() / 2**20
            print(f"{self.label}: {len(self.steps)} decode steps, {reserved_mib:.0f} MiB reserved", flush=True)
        if self.probe_steps is not None and len(self.steps) >= self.probe_steps:
            return self.probe_stop
        return self.never_stop

    def latency(self, tokens):
        return self.start.elapsed_time(self.steps[tokens - 1]) / 1000


def math_box(text):
    command = r"\boxed" if r"\boxed" in text else r"\fbox"
    _, found, right = text.rpartition(command)
    if not found:
        return None
    tail = right.lstrip()
    if tail.startswith("{"):
        depth = 0
        for i, char in enumerate(tail):
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return tail[1:i].strip() or None
    elif command == r"\boxed" and right.startswith(" "):
        return tail.split("$", 1)[0].split("\n", 1)[0].strip() or None
    return None


def json_object(pairs):
    result = dict(pairs)
    if len(pairs) != len(result):
        raise ValueError("Duplicate JSON key")
    return result


def grade(row, output, thinking):
    final = output.rsplit("</think>", 1)[-1].strip()
    if row["rubric"] == "math_box":
        predicted = math_box(final)
        if predicted is None:
            return None, False
        gold = parse(r"\boxed{" + row["gold"] + "}", parsing_timeout=None)
        if not gold:
            raise ValueError(f"Unparseable gold answer: {row['id']}")
        extracted = parse(r"\boxed{" + predicted + "}", parsing_timeout=None)
        return predicted, bool(extracted and verify(gold, extracted, timeout_seconds=None))
    if thinking and "</think>" not in output:
        return None, False
    if not final:
        return None, False
    if row["rubric"] == "exact":
        return final, final == row["gold"]
    if row["rubric"] == "json":
        try:
            return final, json.loads(final, object_pairs_hook=json_object) == row["gold"]
        except ValueError:
            return final, False
    raise ValueError(f"Unknown grading rubric: {row['rubric']}")


def tail_repetition(token_ids):
    tail = token_ids[-256:]
    spans = [tuple(tail[i:i + 8]) for i in range(len(tail) - 7)]
    return (1 - len(set(spans)) / len(spans)) if spans else 0.0


def render_batch(tokenizer, rows, thinking):
    suffix = "<think>\n" if thinking else "<think>\n\n</think>\n\n"
    rendered = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": row["problem"]}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=thinking,
        )
        for row in rows
    ]
    if not all(text.endswith(suffix) for text in rendered):
        raise ValueError("Unexpected Qwen chat template suffix")
    return tokenizer(rendered, padding=True, add_special_tokens=False, return_tensors="pt")


def stop_ids(model, tokenizer):
    config_eos = model.generation_config.eos_token_id
    if isinstance(config_eos, int):
        config_eos = [config_eos]
    return sorted(set((config_eos or []) + [tokenizer.eos_token_id]))


@torch.inference_mode()
def generate_batch(model, tokenizer, rows, thinking, max_tokens, label, probe_steps=None):
    inputs = render_batch(tokenizer, rows, thinking)
    prompt_width = inputs["input_ids"].shape[1]
    prompt_ids = [
        ids[mask.bool()].tolist()
        for ids, mask in zip(inputs["input_ids"], inputs["attention_mask"], strict=True)
    ]
    inputs = inputs.to(model.device)
    eos_ids = stop_ids(model, tokenizer)
    config = GenerationConfig(
        do_sample=True,
        num_beams=1,
        max_new_tokens=max_tokens,
        use_cache=True,
        cache_implementation=CACHE_IMPLEMENTATION,
        disable_compile=True,
        eos_token_id=eos_ids,
        pad_token_id=tokenizer.pad_token_id,
        temperature=SETTINGS["temperature"],
        top_k=SETTINGS["top_k"],
        top_p=SETTINGS["top_p"],
        return_dict_in_generate=probe_steps is not None,
    )
    timer = GenerationTimer(len(rows), model.device, label, probe_steps)
    processor = PresencePenalty(prompt_width, SETTINGS["presence_penalty"])

    torch.manual_seed(SEED)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    timer.start.record()
    sequences = model.generate(**inputs, generation_config=config, logits_processor=[processor], stopping_criteria=[timer])
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    allocated = torch.cuda.max_memory_allocated() / 2**20
    reserved = torch.cuda.max_memory_reserved() / 2**20
    metrics = {
        "batch_elapsed_s": elapsed,
        "decode_steps": len(timer.steps),
        "peak_allocated_mib": allocated,
        "peak_reserved_mib": reserved,
        "eos_token_ids": eos_ids,
    }
    if probe_steps is not None:
        text_config = model.config.get_text_config()
        expected_length = prompt_width + max_tokens - 1
        expected_shape = (len(rows), text_config.num_key_value_heads, expected_length, text_config.head_dim)
        layers = [layer for layer in sequences.past_key_values.layers if isinstance(layer, StaticLayer)]
        if len(layers) != text_config.layer_types.count("full_attention") or not all(
            layer.keys.shape == layer.values.shape == expected_shape
            and layer.keys.dtype == layer.values.dtype == torch.bfloat16
            for layer in layers
        ):
            raise RuntimeError("Smoke test did not allocate the expected BF16 static KV cache")
        metrics["cache_length"] = expected_length
        metrics["cache_layers"] = len(layers)
        metrics["batch_size"] = len(rows)
        metrics["free_mib_with_cache"] = torch.cuda.mem_get_info()[0] / 2**20
        if len(timer.steps) > 1:
            metrics["decode_step_ms_excluding_prefill"] = (
                (timer.latency(len(timer.steps)) - timer.latency(1)) * 1000 / (len(timer.steps) - 1)
            )
        return metrics

    generated = sequences[:, prompt_width:].tolist()
    if not timer.steps or any(len(ids) != len(timer.steps) for ids in generated):
        raise RuntimeError("Decode timing count differs from generated sequence width")

    outputs = []
    for row, source_ids, ids in zip(rows, prompt_ids, generated, strict=True):
        eos_index = next((i for i, token in enumerate(ids) if token in eos_ids), None)
        reason = "eos" if eos_index is not None else "length"
        if eos_index is None and len(ids) != max_tokens:
            raise RuntimeError(f"Generation ended early without an EOS token: {row['id']}")
        output_ids = ids if eos_index is None else ids[:eos_index + 1]
        content_ids = output_ids if eos_index is None else output_ids[:-1]
        output = tokenizer.decode(content_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        prediction, correct = grade(row, output, thinking)
        tail = tail_repetition(content_ids)
        outputs.append({
            "id": row["id"],
            "category": row["category"],
            "prediction": prediction,
            "correct": correct,
            "stop_reason": reason,
            "output_tokens": len(output_ids),
            "latency_s": timer.latency(len(output_ids)),
            "think_closed": "</think>" in output if thinking else None,
            "tail_duplicate_8gram_rate": tail,
            "prompt_token_ids": source_ids,
            "output_token_ids": output_ids,
            "output": output,
        })

    return {**metrics, "outputs": outputs}


def load_model(path):
    print(f"Loading {path.name} checkpoint", flush=True)
    model, info = Qwen3_5ForConditionalGeneration.from_pretrained(
        path,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        local_files_only=True,
        output_loading_info=True,
    )
    for field in ("missing_keys", "mismatched_keys", "error_msgs"):
        if info.get(field):
            raise RuntimeError(f"Invalid checkpoint {path.name} {field}: {info[field]}")
    unexpected = info.get("unexpected_keys", [])
    if unexpected:
        print(f"Checkpoint keys unused by Transformers: {unexpected}", flush=True)
    return model.to("cuda").eval(), sorted(unexpected)


def summarize_outputs(outputs):
    return {
        "n": len(outputs),
        "correct": sum(row["correct"] for row in outputs),
        "length_stops": sum(row["stop_reason"] == "length" for row in outputs),
        "eos_stops": sum(row["stop_reason"] == "eos" for row in outputs),
        "unclosed_thinking": sum(row["think_closed"] is False for row in outputs),
        "high_repetition": sum(row["tail_duplicate_8gram_rate"] >= 0.5 for row in outputs),
        "mean_tail_duplicate_8gram_rate": statistics.mean(row["tail_duplicate_8gram_rate"] for row in outputs),
        "total_output_tokens": sum(row["output_tokens"] for row in outputs),
        "median_output_tokens": statistics.median(row["output_tokens"] for row in outputs),
        "median_latency_s": statistics.median(row["latency_s"] for row in outputs),
    }


def paired_changes(before, after):
    if before.keys() != after.keys():
        raise ValueError("Paired results have different prompt IDs")
    return {
        "improved_ids": [key for key in before if not before[key]["correct"] and after[key]["correct"]],
        "regressed_ids": [key for key in before if before[key]["correct"] and not after[key]["correct"]],
    }


def write_summary(run, inputs, batch_count):
    expected_ids = [row["id"] for row in inputs]
    categories = dict.fromkeys(row["category"] for row in inputs)
    table = {}
    all_rows = {}
    for condition in CONDITIONS:
        batches = [
            load_json(run / f"{condition}-{i + 1:02}.json")
            for i in range(batch_count)
        ]
        rows = [row for batch in batches for row in batch["outputs"]]
        if [row["id"] for row in rows] != expected_ids:
            raise ValueError(f"Missing or reordered rows: {condition}")
        all_rows[condition] = {row["id"]: row for row in rows}
        elapsed = sum(batch["batch_elapsed_s"] for batch in batches)
        table[condition] = {
            **summarize_outputs(rows),
            "generation_wall_s": elapsed,
            "peak_allocated_mib": max(batch["peak_allocated_mib"] for batch in batches),
            "peak_reserved_mib": max(batch["peak_reserved_mib"] for batch in batches),
            "categories": {
                category: summarize_outputs([row for row in rows if row["category"] == category])
                for category in categories
            },
        }
    for mode in ("on", "off"):
        for key in expected_ids:
            if all_rows[f"base-{mode}"][key]["prompt_token_ids"] != all_rows[f"p50c-{mode}"][key]["prompt_token_ids"]:
                raise ValueError(f"The two models saw different input tokens for {key} in {mode} mode")
    save_json(run / "summary.json", {
        "finished_at": datetime.now().astimezone().isoformat(),
        "conditions": table,
        "weight_change_thinking_on": paired_changes(all_rows["base-on"], all_rows["p50c-on"]),
        "weight_change_thinking_off": paired_changes(all_rows["base-off"], all_rows["p50c-off"]),
        "turn_thinking_off_base": paired_changes(all_rows["base-on"], all_rows["base-off"]),
        "turn_thinking_off_p50c": paired_changes(all_rows["p50c-on"], all_rows["p50c-off"]),
    })


def run_config(args, data_path, tokenizer, rows):
    packages = ("torch", "transformers", "huggingface-hub", "safetensors", "numpy", "math-verify")
    driver = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        text=True,
    ).strip()
    return {
        "seed_each_batch": SEED,
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "category": args.category,
        "prompt_ids": [row["id"] for row in rows],
        "sampling": SETTINGS,
        "condition_order": CONDITIONS,
        "attention_backend": "sdpa",
        "cache_implementation": CACHE_IMPLEMENTATION,
        "torch_alloc_conf": os.environ.get("PYTORCH_ALLOC_CONF"),
        "dtype": "bfloat16",
        "tokenizer_pad_id": tokenizer.pad_token_id,
        "tokenizer_eos_id": tokenizer.eos_token_id,
        "latency_definition": "CUDA elapsed time from batch generation start through the stop token or cap; includes prefill",
        "output_token_definition": "Includes the generated EOS token; excludes padding after EOS and all prompt tokens",
        "loop_metric": "1 minus unique/token 8-grams for last 256 content tokens; high_repetition at >= 0.5",
        "prompt_sha256": sha256(data_path),
        "runner_sha256": sha256(ROOT / "run.py"),
        "model_provenance": load_json(ROOT / "models" / "provenance.json"),
        "python": sys.version,
        "os": platform.platform(),
        "packages": {package: importlib.metadata.version(package) for package in packages},
        "gpu": torch.cuda.get_device_name(),
        "driver": driver,
        "torch_cuda": torch.version.cuda,
    }


def run_pilot(args, run, batches, tokenizer, config):
    config_path = run / "config.json"
    if config_path.exists():
        if load_json(config_path) != config:
            raise ValueError("Run configuration changed. Use a new run ID.")
    else:
        save_json(config_path, config)

    for name, path in MODELS.items():
        pending = [
            (thinking, i, rows, run / f"{name}-{'on' if thinking else 'off'}-{i:02}.json")
            for thinking in (True, False)
            for i, rows in enumerate(batches, start=1)
        ]
        pending = [batch for batch in pending if not batch[3].exists()]
        if not pending:
            continue
        model, unexpected = load_model(path)
        warmup = generate_batch(model, tokenizer, batches[0], True, 24, f"{name} warmup")
        print(f"{name} warmup: {warmup['decode_steps']} steps in {warmup['batch_elapsed_s']:.2f}s", flush=True)
        for thinking, i, rows, out in pending:
            result = generate_batch(model, tokenizer, rows, thinking, args.max_new_tokens, out.stem)
            result.update({"model": name, "thinking": thinking, "batch": i, "unused_checkpoint_keys": unexpected})
            save_json(out, result)
            elapsed = result["batch_elapsed_s"]
            stats = summarize_outputs(result["outputs"])
            print(f"{out.stem}: {stats['correct']}/{stats['n']} correct; "
                  f"{stats['length_stops']} capped; {stats['total_output_tokens']} tokens in {elapsed:.2f}s", flush=True)
        del model
        torch.cuda.empty_cache()
    write_summary(run, [row for batch in batches for row in batch], len(batches))
    print(f"Summary: {run / 'summary.json'}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="pilot-4096")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--category", choices=CATEGORIES)
    parser.add_argument("--prompts", type=Path, default=ROOT / "data" / "pilot.json",
                        help="JSON file containing prompt rows (default: data/pilot.json)")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9-]{0,63}", args.run_id):
        parser.error("Invalid run ID")
    if not 1 <= args.batch_size <= 32 or args.max_new_tokens < 1:
        parser.error("Use a batch size from 1 to 32 and a positive token cap")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(including_emulation=False):
        raise RuntimeError("This runner requires CUDA with hardware BF16 support")
    data_path = args.prompts
    rows = load_json(data_path)["rows"]
    if args.category is not None:
        rows = [row for row in rows if row["category"] == args.category]
    if not rows:
        raise ValueError("No prompts match the selection")
    batches = [rows[i:i + args.batch_size] for i in range(0, len(rows), args.batch_size)]
    tokenizer = AutoTokenizer.from_pretrained(BASE, local_files_only=True, padding_side="left")
    if tokenizer.pad_token_id is None or tokenizer.eos_token_id is None:
        raise ValueError("The pinned tokenizer is missing pad or EOS token IDs")
    if args.smoke:
        for row in rows:
            gold = row["gold"]
            sample = gold
            if row["rubric"] == "math_box":
                sample = r"\boxed{" + gold + "}"
            elif row["rubric"] == "json":
                sample = json.dumps(gold)
            if not grade(row, sample, False)[1]:
                raise ValueError(f"Grader rejected a known correct answer: {row['id']}")
        longest_batches = {
            thinking: max(batches, key=lambda batch: render_batch(tokenizer, batch, thinking)["input_ids"].shape[1])
            for thinking in (True, False)
        }
        for name, path in MODELS.items():
            model, _ = load_model(path)
            for thinking in (True, False):
                result = generate_batch(
                    model, tokenizer, longest_batches[thinking], thinking,
                    args.max_new_tokens, f"{name} smoke", probe_steps=32,
                )
                print("Cache check " + json.dumps({"model": name, "thinking": thinking, **result}), flush=True)
            del model
            torch.cuda.empty_cache()
        return

    run = ROOT / "runs" / args.run_id
    run.mkdir(exist_ok=True)
    with (run / "run.lock").open("a+b") as lock:
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        run_pilot(args, run, batches, tokenizer, run_config(args, data_path, tokenizer, rows))


if __name__ == "__main__":
    main()
