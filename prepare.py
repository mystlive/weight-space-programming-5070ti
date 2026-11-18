import hashlib
import json
import os
import random
import shutil
import struct
import subprocess
import sys
from pathlib import Path
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "models" / "base"
EDITED = ROOT / "models" / "p50c"
WEIGHT_NAME = "model.safetensors-00001-of-00001.safetensors"
MODEL_REPO = "Qwen/Qwen3.5-0.8B"
MODEL_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
PUBLISHED_REPO = "RikiyaT/Qwen3.5-0.8B-weight-space-programming"
PUBLISHED_REVISION = "d1671a178705bcab7644806d23c535517f35723b"
LOG_REPO = "RikiyaT/lm_understand-logs"
MATH_REPO = "HuggingFaceH4/MATH-500"
MATH_INSTRUCTION = r"Please reason step by step, and put your final answer within \boxed{}."


def save_json(path, value):
    temp_path = path.with_name(path.name + ".tmp")
    temp_path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp_path.replace(path)


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fetch_git_sources():
    upstream = ROOT / "upstream"
    revision_path = upstream / "revision.json"
    if revision_path.is_file():
        revision = load_json(revision_path)["sha"]
    else:
        request = Request(f"https://api.github.com/repos/{LOG_REPO}/commits/main", headers={"User-Agent": "wsp-reproduction"})
        with urlopen(request, timeout=30) as response:
            revision = json.load(response)["sha"]
        save_json(revision_path, {"repository": LOG_REPO, "sha": revision})

    files = {
        "exp8_math/code/apply_edit.py": upstream / "apply_edit.py",
        "exp8_math/code/st_numpy.py": upstream / "st_numpy.py",
        "exp8_math/runs/p50c/edit/manifest.json": upstream / "p50c" / "manifest.json",
        "exp8_math/runs/p50c/edit/edit.npz": upstream / "p50c" / "edit.npz",
        "LICENSE": upstream / "LICENSE",
    }
    for remote, local in files.items():
        if local.is_file():
            continue
        local.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://raw.githubusercontent.com/{LOG_REPO}/{revision}/{remote}"
        with urlopen(url, timeout=60) as response:
            content = response.read()
        temp_path = local.with_name(local.name + ".tmp")
        temp_path.write_bytes(content)
        temp_path.replace(local)
    return revision, {path.name: sha256(path) for path in files.values()}


def safetensors_header(stream):
    header_bytes = struct.unpack("<Q", stream.read(8))[0]
    tensors = json.loads(stream.read(header_bytes))
    tensors.pop("__metadata__", None)
    return tensors, stream.tell()


def tensor_digest(stream, header_end, tensor):
    start, end = tensor["data_offsets"]
    stream.seek(header_end + start)
    digest = hashlib.sha256()
    remaining = end - start
    while remaining:
        chunk = stream.read(min(remaining, 4 * 1024 * 1024))
        if not chunk:
            raise ValueError("Truncated safetensors payload")
        digest.update(chunk)
        remaining -= len(chunk)
    return digest.hexdigest()


def verify_tensor_payloads(base_path, edited_path, manifest):
    patch = {tensor["name"]: tensor for tensor in manifest["tensors"]}
    changed = {}
    with base_path.open("rb") as base, edited_path.open("rb") as edited:
        base_tensors, base_offset = safetensors_header(base)
        edited_tensors, edited_offset = safetensors_header(edited)
        if base_tensors.keys() != edited_tensors.keys():
            raise ValueError("The edited checkpoint has different tensor names")
        for name, base_tensor in base_tensors.items():
            edited_tensor = edited_tensors[name]
            for key in ("dtype", "shape"):
                if base_tensor[key] != edited_tensor[key]:
                    raise ValueError(f"Tensor {key} mismatch: {name}")
            before = tensor_digest(base, base_offset, base_tensor)
            after = tensor_digest(edited, edited_offset, edited_tensor)
            if before != after:
                changed[name] = {"base_sha256": before, "edited_sha256": after}
        expected = {name: {key: tensor[key] for key in ("base_sha256", "edited_sha256")} for name, tensor in patch.items()}
        if changed != expected:
            raise ValueError("Edited tensor digests differ from the published p50c manifest")
    return {"verified_tensors": len(base_tensors), "changed": changed, "changed_entries": manifest["changed_entries_total"]}


def prepare_models():
    from huggingface_hub import get_hf_file_metadata, hf_hub_url, snapshot_download

    upstream = ROOT / "upstream"
    log_revision, upstream_digests = fetch_git_sources()
    print(f"Published patch source: {LOG_REPO}@{log_revision}", flush=True)
    print(f"Downloading {MODEL_REPO}@{MODEL_REVISION}", flush=True)
    snapshot_download(
        MODEL_REPO,
        revision=MODEL_REVISION,
        allow_patterns=[WEIGHT_NAME, "*.json", "*.jinja", "merges.txt", "LICENSE"],
        token=False,
        local_dir=BASE,
    )

    EDITED.mkdir(exist_ok=True)
    base_path, edited_path = BASE / WEIGHT_NAME, EDITED / WEIGHT_NAME
    manifest = load_json(upstream / "p50c" / "manifest.json")
    if edited_path.is_file():
        verification = verify_tensor_payloads(base_path, edited_path, manifest)
    else:
        temp_path = edited_path.with_name("rebuilding.safetensors")
        subprocess.run(
            [sys.executable, "-B", str(upstream / "apply_edit.py"), str(base_path), str(upstream / "p50c"), str(temp_path)],
            check=True,
        )
        verification = verify_tensor_payloads(base_path, temp_path, manifest)
        temp_path.replace(edited_path)

    for model_file in BASE.iterdir():
        if model_file.is_file() and model_file.name != WEIGHT_NAME:
            shutil.copy2(model_file, EDITED / model_file.name)

    hashes = {"base": sha256(base_path), "p50c": sha256(edited_path)}
    metadata = get_hf_file_metadata(hf_hub_url(PUBLISHED_REPO, WEIGHT_NAME, revision=PUBLISHED_REVISION), token=False)
    provenance = {
        "base": {"repository": MODEL_REPO, "revision": MODEL_REVISION, "sha256": hashes["base"]},
        "p50c": {
            "repository": PUBLISHED_REPO,
            "revision": PUBLISHED_REVISION,
            "rebuilt_sha256": hashes["p50c"],
            "published_etag": metadata.etag,
            "published_file_match": hashes["p50c"] == metadata.etag,
        },
        "patch": {"repository": LOG_REPO, "revision": log_revision, "file_sha256": upstream_digests},
        "tensor_verification": verification,
        "shared_model_file_sha256": {
            path.name: sha256(path) for path in BASE.iterdir() if path.is_file() and path.name != WEIGHT_NAME
        },
    }
    save_json(ROOT / "models" / "provenance.json", provenance)
    print(f"Verified {verification['verified_tensors']} tensors, {len(verification['changed'])} changed", flush=True)
    print(f"Rebuilt file matches published weight hash: {provenance['p50c']['published_file_match']}", flush=True)


def prepare_prompts():
    from huggingface_hub import HfApi, hf_hub_download

    data = ROOT / "data"
    math_revision = HfApi().dataset_info(MATH_REPO, token=False).sha
    math_path = Path(hf_hub_download(MATH_REPO, "test.jsonl", repo_type="dataset", revision=math_revision, token=False))
    population = [json.loads(line) for line in math_path.read_text(encoding="utf-8").splitlines() if line]
    if len(population) != 500:
        raise ValueError(f"Expected 500 problems; found {len(population)}")
    rng = random.Random(0)
    selected = []
    for level in (1, 2):
        candidates = sorted((row for row in population if row["level"] == level), key=lambda row: row["unique_id"])
        for row in rng.sample(candidates, 4):
            selected.append({
                "id": row["unique_id"],
                "category": "math",
                "rubric": "math_box",
                "problem": row["problem"] + "\n" + MATH_INSTRUCTION,
                "gold": row["answer"],
                "level": row["level"],
                "subject": row["subject"],
            })

    local_rows = load_json(data / "local_prompts.json")
    groups = [selected] + [[row for row in local_rows if row["category"] == name] for name in ("japanese", "instructions", "loop")]
    if any(len(group) != 8 for group in groups) or len(local_rows) != 24:
        raise ValueError("Expected eight prompts per category")
    rows = [row for group in zip(*groups, strict=True) for row in group]
    if len({row["id"] for row in rows}) != 32:
        raise ValueError("Prompt IDs must be unique")
    save_json(data / "pilot.json", {
        "math_dataset": {"repository": MATH_REPO, "revision": math_revision, "file_sha256": sha256(math_path), "split": "test"},
        "math_selection": "Python Random(0), four sorted candidates sampled at each of levels 1 and 2",
        "rows": rows,
    })
    print(f"Prepared {len(rows)} prompts", flush=True)


def main():
    cache = ROOT / "cache" / "huggingface"
    os.environ["HF_HOME"] = str(cache)
    os.environ["HF_HUB_CACHE"] = str(cache / "hub")
    os.environ["HF_XET_CACHE"] = str(cache / "xet")
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["HF_HUB_ETAG_TIMEOUT"] = "30"
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"
    prepare_models()
    prepare_prompts()


if __name__ == "__main__":
    main()
