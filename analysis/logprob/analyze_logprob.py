from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


BYTE_TOKEN_RE = re.compile(r"^<0x([0-9A-Fa-f]{2})>$")


def is_byte_token(token_str: str) -> bool:
    return BYTE_TOKEN_RE.match(token_str) is not None


def safe_convert_id_to_token(tokenizer, token_id: int) -> str:
    try:
        token = tokenizer.convert_ids_to_tokens(token_id)
        if isinstance(token, str):
            return token
    except Exception:
        pass
    return f"<id:{token_id}>"


def build_offset_mapping_from_decoded_pieces(
    tokenizer,
    token_ids: List[int],
) -> Tuple[List[Dict[str, Any]], str]:
    """
    各 token を1個ずつ decode した文字列を順に連結し、
    その token が最終文字列のどの範囲に相当するかを作る。
    """
    offsets: List[Dict[str, Any]] = []
    pieces: List[str] = []
    cursor = 0

    for tid in token_ids:
        piece = tokenizer.decode([tid], clean_up_tokenization_spaces=False)
        start = cursor
        end = cursor + len(piece)
        offsets.append(
            {
                "token_id": tid,
                "piece": piece,
                "start": start,
                "end": end,
            }
        )
        pieces.append(piece)
        cursor = end

    full_text = "".join(pieces)
    return offsets, full_text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--continuation", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max_length", type=int, default=4096)
    args = parser.parse_args()

    model_path = args.model
    tokenizer_path = args.tokenizer if args.tokenizer is not None else args.model
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] loading tokenizer from: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True,
    )

    print(f"[INFO] loading model from: {model_path}")
    if args.device == "auto":
        device_map = "auto"
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    else:
        device_map = None
        dtype = torch.bfloat16 if ("cuda" in args.device and torch.cuda.is_available()) else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        dtype=dtype,
        device_map=device_map,
    )
    model.eval()

    full_text = args.prompt + args.continuation

    enc_full = tokenizer(
        full_text,
        return_tensors="pt",
        add_special_tokens=False,
    )
    enc_prompt = tokenizer(
        args.prompt,
        return_tensors="pt",
        add_special_tokens=False,
    )

    input_ids = enc_full["input_ids"]
    attention_mask = enc_full["attention_mask"]
    prompt_len = enc_prompt["input_ids"].shape[1]
    total_len = input_ids.shape[1]

    if total_len > args.max_length:
        raise ValueError(
            f"Input too long: total_len={total_len} > max_length={args.max_length}"
        )

    if prompt_len >= total_len:
        raise ValueError("Prompt length must be smaller than full text length.")

    if args.device != "auto":
        input_ids = input_ids.to(args.device)
        attention_mask = attention_mask.to(args.device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # [1, T, V]
        log_probs = F.log_softmax(logits, dim=-1)

    token_ids = input_ids[0].tolist()
    offsets, decoded_full_text = build_offset_mapping_from_decoded_pieces(tokenizer, token_ids)

    rows: List[Dict[str, Any]] = []

    # causal LM なので token j は logits[j-1] で予測される
    for j in range(1, total_len):
        tid = token_ids[j]
        token_str = safe_convert_id_to_token(tokenizer, tid)
        lp = log_probs[0, j - 1, tid].item()
        prob = math.exp(lp)

        row = {
            "index": j,
            "token_id": tid,
            "token_str": token_str,
            "decoded_piece": offsets[j]["piece"],
            "char_start": offsets[j]["start"],
            "char_end": offsets[j]["end"],
            "is_in_continuation": bool(j >= prompt_len),
            "type": "byte" if is_byte_token(token_str) else "word",
            "logprob": lp,
            "prob": prob,
        }
        rows.append(row)

    continuation_rows = [r for r in rows if r["is_in_continuation"]]

    result = {
        "model": model_path,
        "tokenizer": tokenizer_path,
        "prompt": args.prompt,
        "continuation": args.continuation,
        "full_text_input": full_text,
        "decoded_full_text": decoded_full_text,
        "prompt_token_len": prompt_len,
        "total_token_len": total_len,
        "continuation_token_len": len(continuation_rows),
        "tokens": continuation_rows,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"[INFO] saved token analysis to: {output_path}")
    print(f"[INFO] continuation tokens: {len(continuation_rows)}")


if __name__ == "__main__":
    main()