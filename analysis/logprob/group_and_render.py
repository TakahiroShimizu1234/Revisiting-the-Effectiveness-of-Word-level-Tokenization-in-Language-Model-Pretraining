from __future__ import annotations

import argparse
import html
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List


BYTE_TOKEN_RE = re.compile(r"^<0x([0-9A-Fa-f]{2})>$")


def is_byte_token(token_str: str) -> bool:
    return BYTE_TOKEN_RE.match(token_str) is not None


def byte_token_to_int(token_str: str) -> int:
    m = BYTE_TOKEN_RE.match(token_str)
    if m is None:
        raise ValueError(f"Not a byte token: {token_str}")
    return int(m.group(1), 16)


def try_decode_utf8(byte_values: List[int]) -> str:
    try:
        return bytes(byte_values).decode("utf-8")
    except UnicodeDecodeError:
        return "".join(f"\\x{b:02x}" for b in byte_values)


def visible_text(s: str) -> str:
    return s.replace(" ", "␠").replace("\n", "↵\n").replace("\t", "⇥")


def logprob_to_score01_sigmoid(lp: float, center: float = -4.0, scale: float = 1.5) -> float:
    """
    logprob を 0~1 に非線形変換。
    値が低いほど 0 に近づき、赤くなる。
    """
    z = (lp - center) / scale
    return 1.0 / (1.0 + math.exp(-z))


def score_to_color(score01: float) -> str:
    """
    0 -> 赤, 0.5 -> 黄, 1 -> 緑
    """
    score01 = max(0.0, min(1.0, score01))

    if score01 < 0.5:
        t = score01 / 0.5
        r1, g1, b1 = 248, 105, 107  # red
        r2, g2, b2 = 255, 244, 204  # yellow
    else:
        t = (score01 - 0.5) / 0.5
        r1, g1, b1 = 255, 244, 204  # yellow
        r2, g2, b2 = 111, 207, 151  # green

    r = round(r1 + (r2 - r1) * t)
    g = round(g1 + (g2 - g1) * t)
    b = round(b1 + (b2 - b1) * t)
    return f"rgb({r},{g},{b})"


def group_tokens(tokens: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    spans: List[Dict[str, Any]] = []
    i = 0
    n = len(tokens)

    while i < n:
        tok = tokens[i]

        if tok["type"] == "word":
            piece = tok["decoded_piece"]
            display_text = visible_text(piece)
            lp = tok["logprob"]

            span = {
                "type": "word",
                "text": display_text if display_text != "" else tok["token_str"],
                "raw_text": piece,
                "token_strs": [tok["token_str"]],
                "token_ids": [tok["token_id"]],
                "logprobs": [lp],
                "sum_logprob": lp,
                "avg_logprob": lp,
                "score_logprob": lp,   # word はその token の logprob
                "length": 1,
            }
            spans.append(span)
            i += 1
            continue

        byte_values: List[int] = []
        token_strs: List[str] = []
        token_ids: List[int] = []
        logprobs: List[float] = []

        while i < n and tokens[i]["type"] == "byte":
            b = byte_token_to_int(tokens[i]["token_str"])
            byte_values.append(b)
            token_strs.append(tokens[i]["token_str"])
            token_ids.append(tokens[i]["token_id"])
            logprobs.append(tokens[i]["logprob"])
            i += 1

        raw_text = try_decode_utf8(byte_values)
        display_text = visible_text(raw_text)
        sum_lp = sum(logprobs)
        avg_lp = sum_lp / max(len(logprobs), 1)

        span = {
            "type": "byte",
            "text": display_text,
            "raw_text": raw_text,
            "token_strs": token_strs,
            "token_ids": token_ids,
            "byte_values": byte_values,
            "logprobs": logprobs,
            "sum_logprob": sum_lp,
            "avg_logprob": avg_lp,
            "score_logprob": sum_lp,  # byte span は掛け算 = logprob の和
            "length": len(logprobs),
        }
        spans.append(span)

    return spans


def add_scores(spans: List[Dict[str, Any]], center: float = -4.0, scale: float = 1.5) -> None:
    for s in spans:
        s["score01"] = logprob_to_score01_sigmoid(
            s["score_logprob"],
            center=center,
            scale=scale,
        )
        s["color"] = score_to_color(s["score01"])


def render_html(
    prompt: str,
    continuation: str,
    spans: List[Dict[str, Any]],
    output_html: Path,
    center: float,
    scale: float,
) -> None:
    add_scores(spans, center=center, scale=scale)

    span_html_parts: List[str] = []
    for s in spans:
        color = s["color"]
        border_style = "2px solid #2f855a" if s["type"] == "word" else "2px dashed #2b6cb0"

        tooltip_lines = [
            f"type: {s['type']}",
            f"text: {repr(s['raw_text'])}",
            f"score_logprob: {s['score_logprob']:.4f}",
            f"sum_logprob: {s['sum_logprob']:.4f}",
            f"avg_logprob: {s['avg_logprob']:.4f}",
            f"score01: {s['score01']:.4f}",
            f"length: {s['length']}",
            f"token_strs: {s['token_strs']}",
            f"token_ids: {s['token_ids']}",
        ]
        if s["type"] == "byte":
            tooltip_lines.append(f"bytes: {s['byte_values']}")

        tooltip = html.escape("\n".join(tooltip_lines))
        label = "W" if s["type"] == "word" else "B"
        text_html = html.escape(s["text"])

        part = f"""
        <span class="tok" title="{tooltip}" style="background:{color}; border:{border_style};">
          <span class="badge">{label}</span>{text_html}
        </span>
        """
        span_html_parts.append(part)

    html_text = f"""<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="UTF-8">
  <title>Word / Byte Confidence Visualization</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      margin: 24px;
      line-height: 1.8;
      color: #222;
      background: #ffffff;
    }}
    .block {{
      margin-bottom: 24px;
      padding: 16px;
      border: 1px solid #ddd;
      border-radius: 12px;
      background: #fafafa;
    }}
    .title {{
      font-size: 18px;
      font-weight: 700;
      margin-bottom: 10px;
    }}
    .text {{
      font-size: 22px;
      white-space: pre-wrap;
      word-break: break-word;
    }}
    .tok {{
      display: inline-block;
      margin: 3px 4px 3px 0;
      padding: 2px 8px;
      border-radius: 10px;
      vertical-align: middle;
    }}
    .badge {{
      display: inline-block;
      font-size: 11px;
      font-weight: 700;
      margin-right: 6px;
      padding: 0 5px;
      border-radius: 999px;
      background: rgba(255,255,255,0.75);
    }}
    .legend {{
      display: flex;
      gap: 16px;
      align-items: center;
      flex-wrap: wrap;
      margin-bottom: 16px;
      font-size: 14px;
    }}
    .chip {{
      display: inline-block;
      padding: 6px 10px;
      border-radius: 10px;
    }}
    .small {{
      font-size: 13px;
      color: #555;
    }}
    .mono {{
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    }}
  </style>
</head>
<body>
  <h1>Word / Byte Confidence Visualization</h1>

  <div class="legend">
    <span class="chip" style="background:rgb(248,105,107);">low confidence</span>
    <span class="chip" style="background:rgb(255,244,204);">middle</span>
    <span class="chip" style="background:rgb(111,207,151);">high confidence</span>
    <span class="chip" style="border:2px solid #2f855a;">W = word token</span>
    <span class="chip" style="border:2px dashed #2b6cb0;">B = byte span</span>
  </div>

  <div class="block">
    <div class="title">Color mapping</div>
    <div class="small mono">
      score01 = sigmoid((score_logprob - center) / scale), center={center:.2f}, scale={scale:.2f}
    </div>
    <div class="small">
      Word token は 1 token の logprob、byte span は byte 全体の logprob の和を使っています。
      そのため、byte span は真っ赤になりやすい設定です。
    </div>
  </div>

  <div class="block">
    <div class="title">Prompt</div>
    <div class="text">{html.escape(prompt)}</div>
  </div>

  <div class="block">
    <div class="title">Continuation (visualized)</div>
    <div class="small">
      Hover each segment to inspect logprob, token ids, and whether it is a word token or a byte span.
    </div>
    <div class="text">
      {''.join(span_html_parts)}
    </div>
  </div>

  <div class="block">
    <div class="title">Raw continuation</div>
    <div class="text">{html.escape(continuation)}</div>
  </div>
</body>
</html>
"""
    output_html.write_text(html_text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument("--output_html", type=str, required=True)
    parser.add_argument("--center", type=float, default=-4.0)
    parser.add_argument("--scale", type=float, default=1.5)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_json = Path(args.output_json)
    output_html = Path(args.output_html)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_html.parent.mkdir(parents=True, exist_ok=True)

    data = json.loads(input_path.read_text(encoding="utf-8"))
    prompt = data["prompt"]
    continuation = data["continuation"]
    tokens = data["tokens"]

    spans = group_tokens(tokens)
    add_scores(spans, center=args.center, scale=args.scale)

    out = {
        "prompt": prompt,
        "continuation": continuation,
        "center": args.center,
        "scale": args.scale,
        "spans": spans,
    }
    output_json.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    render_html(
        prompt=prompt,
        continuation=continuation,
        spans=spans,
        output_html=output_html,
        center=args.center,
        scale=args.scale,
    )

    print(f"[INFO] saved spans json to: {output_json}")
    print(f"[INFO] saved html to: {output_html}")


if __name__ == "__main__":
    main()