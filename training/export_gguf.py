#!/usr/bin/env python3
"""Merge a Greeto LoRA adapter and export a Q4_K_M GGUF for llama.cpp.

Pipeline:

    1. load the base Qwen model + PEFT adapter;
    2. merge the adapter into the base weights;
    3. call llama.cpp's ``convert_hf_to_gguf.py`` (f16);
    4. call ``llama-quantize`` (Q4_K_M);
    5. copy/point the result at ``models/llm/`` for config/llm.yaml.

A llama.cpp checkout is required only for steps 3-4. Point at it with
``--llama_cpp_dir`` (default: third_party/llama.cpp).

    python training/export_gguf.py --dry-run
    python training/export_gguf.py \
        --adapter_dir models/qwen25-greeto-lora \
        --llama_cpp_dir third_party/llama.cpp
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--adapter_dir", default="models/qwen25-greeto-lora")
    parser.add_argument("--merged_dir", default="models/qwen25-greeto-merged")
    parser.add_argument("--gguf_dir", default="models/llm")
    parser.add_argument("--llama_cpp_dir", default="third_party/llama.cpp")
    parser.add_argument("--quant", default="Q4_K_M")
    parser.add_argument(
        "--output_stem",
        "--output-stem",
        dest="output_stem",
        default=None,
        help="GGUF filename stem (default: <base>-greeto).",
    )
    parser.add_argument("--skip_merge", "--skip-merge", dest="skip_merge", action="store_true")
    parser.add_argument("--dry_run", "--dry-run", dest="dry_run", action="store_true")
    return parser.parse_args()


def resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def find_convert_script(llama_dir: Path) -> Path:
    for name in (
        "convert_hf_to_gguf.py",
        "convert-hf-to-gguf.py",
        "convert.py",
    ):
        candidate = llama_dir / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not find convert_hf_to_gguf.py under %s. Clone "
        "https://github.com/ggerganov/llama.cpp there or pass "
        "--llama_cpp_dir." % llama_dir
    )


def find_quantizer(llama_dir: Path) -> Path:
    candidates = [
        llama_dir / "build" / "bin" / "llama-quantize",
        llama_dir / "build" / "bin" / "quantize",
        llama_dir / "llama-quantize",
        llama_dir / "quantize",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    executable = shutil.which("llama-quantize") or shutil.which("quantize")
    if executable:
        return Path(executable)

    raise FileNotFoundError(
        "Could not find llama-quantize. Build llama.cpp first "
        "(cmake -B build && cmake --build build --config Release)."
    )


def run(command: list[str]) -> None:
    print("$ " + " ".join(command))
    subprocess.run(command, check=True)


def merge_adapter(base_model: str, adapter_dir: Path, merged_dir: Path) -> None:
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise SystemExit(
            "Merge dependencies are missing (%s). Install training/requirements.txt."
            % error
        )

    print("Loading base model %s..." % base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
    )

    print("Merging adapter %s..." % adapter_dir)
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    model = model.merge_and_unload()

    merged_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(merged_dir), safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.save_pretrained(str(merged_dir))

    print("Merged model written to %s" % merged_dir)


def export(args: argparse.Namespace) -> int:
    base_model = args.base_model
    adapter_dir = resolve(args.adapter_dir)
    merged_dir = resolve(args.merged_dir)
    gguf_dir = resolve(args.gguf_dir)
    llama_dir = resolve(args.llama_cpp_dir)

    base_stem = base_model.rstrip("/").split("/")[-1].lower()
    model_stem = args.output_stem or (base_stem + "-greeto")
    f16_path = gguf_dir / ("%s-f16.gguf" % model_stem)
    quant_path = gguf_dir / ("%s-%s.gguf" % (model_stem, args.quant.lower()))

    if args.dry_run:
        print("Base model:   %s" % base_model)
        print("Adapter:      %s" % adapter_dir)
        print("Merged model: %s" % merged_dir)
        print("f16 GGUF:     %s" % f16_path)
        print("Quantised:    %s (%s)" % (quant_path, args.quant))
        print("llama.cpp:    %s" % llama_dir)
        print("\nDry run OK. Run without --dry_run on a machine with PyTorch and llama.cpp.")
        return 0

    if not args.skip_merge:
        if not adapter_dir.exists():
            print("Adapter directory not found: %s" % adapter_dir)
            return 1
        merge_adapter(base_model, adapter_dir, merged_dir)

    if not merged_dir.exists():
        print("Merged model directory not found: %s" % merged_dir)
        return 1

    convert_script = find_convert_script(llama_dir)
    quantizer = find_quantizer(llama_dir)

    gguf_dir.mkdir(parents=True, exist_ok=True)

    run(
        [
            sys.executable,
            str(convert_script),
            str(merged_dir),
            "--outfile",
            str(f16_path),
            "--outtype",
            "f16",
        ]
    )

    run([str(quantizer), str(f16_path), str(quant_path), args.quant])

    print("\nQuantised model: %s" % quant_path)
    print("Set config/llm.yaml model.path to:")
    print("  %s" % quant_path.relative_to(PROJECT_ROOT).as_posix())
    return 0


def main() -> int:
    return export(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
