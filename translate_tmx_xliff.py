#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
translate_tmx_xliff.py

Translate TMX or XLIFF files in chunks using a system prompt from a separate file,
and write a bilingual output that can be imported back into CAT tools.

Outputs:
  - Translated TMX or XLIFF file with updated segments.

Usage example:
  export OPENAI_API_KEY=sk-...
  python translate_tmx_xliff.py \
      --input file.tmx \
      --sysprompt-file sysprompt.txt \
      --model gpt-5 \
      --chunk-size 6000
"""

import os
import re
import sys
import time
import random
import queue
import threading
import argparse
from typing import Dict, List, Tuple, Optional
import xml.etree.ElementTree as ET

# OpenAI SDK (v1.x). Install with: pip install openai
try:
    import openai
except Exception:
    openai = None  # we'll error out if you try to run without it

# ---------------------------
# Threading state / locks
# ---------------------------
translated_texts: Dict[int, str] = {}
translated_texts_lock = threading.Lock()
print_lock = threading.Lock()

# Will be set in main()
translation_queue: "queue.Queue[Optional[Tuple[int, str]]]" = None  # type: ignore

def test_openai_api_key(model: str = "gpt-4o-mini") -> bool:
    """
    Quick sanity check: tries a 1-token request to verify the API key works.
    Returns True if successful, False otherwise.
    """
    if openai is None:
        print("OpenAI SDK not installed. Run: pip install openai")
        return False
    try:
        resp = openai.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "ping"}],
            max_completion_tokens=10000,
        )
        print(f"API key test successful with model '{model}'.")
        return True
    except Exception as e:
        print(f"API key test failed: {e}")
        return False

# ---------------------------
# Helpers
# ---------------------------

def log(msg: str):
    with print_lock:
        print(msg, flush=True)

def read_text(path: str, encoding: str) -> str:
    with open(path, "r", encoding=encoding, errors="replace") as f:
        return f.read()

def parse_tmx_xliff(file_path: str) -> Tuple[ET.ElementTree, List[Tuple[ET.Element, str]]]:
    """
    Parse TMX or (M)XLIFF and extract segments for translation.

    Returns:
        tree: parsed ElementTree (so you can write back changes)
        segments: list of tuples (writable_element, source_text)
          - For TMX: (seg_element, seg_text)
          - For XLIFF/MXLIFF: (top-level <target> under <trans-unit>, source_text)
            (If <target> is missing, it is created directly under <trans-unit>.)
    """
    def local_name(tag: str) -> str:
        # '{ns}name' -> 'name' or 'name' -> 'name'
        return tag.split('}', 1)[1] if '}' in tag else tag

    def ns_tag_like(tag_in_doc: str, desired_local: str) -> str:
        # Use the same namespace as tag_in_doc, if any
        if '}' in tag_in_doc:
            ns = tag_in_doc.split('}', 1)[0][1:]
            return f'{{{ns}}}{desired_local}'
        return desired_local

    tree = ET.parse(file_path)
    root = tree.getroot()
    segments: List[Tuple[ET.Element, str]] = []

    # --- TMX path: collect <seg> (namespaced or not) ---
    for seg in root.iter():
        if local_name(seg.tag) == 'seg':
            seg_text = ''.join(seg.itertext())
            segments.append((seg, seg_text))

    # --- XLIFF / MXLIFF path: collect <trans-unit> ---
    # Only consider the top-level <target> directly under <trans-unit> (ignore <alt-trans> targets).
    for tu in root.iter():
        if local_name(tu.tag) != 'trans-unit':
            continue

        source_el = None
        target_el = None
        for child in list(tu):
            ln = local_name(child.tag)
            if ln == 'source' and source_el is None:
                source_el = child
            elif ln == 'target' and target_el is None:
                target_el = child

        if source_el is None:
            continue

        source_text = ''.join(source_el.itertext())

        if target_el is None:
            # Create a top-level <target> using the same namespace as <source>, if present
            target_tag = ns_tag_like(source_el.tag, 'target')
            target_el = ET.Element(target_tag)
            # Insert right after <source> when possible, otherwise append
            children = list(tu)
            try:
                src_idx = children.index(source_el)
                tu.insert(src_idx + 1, target_el)
            except ValueError:
                tu.append(target_el)

        segments.append((target_el, source_text))

    return tree, segments

def translate_text(text: str, sysprompt: str, model: str) -> str:
    """
    Translate 'text' using OpenAI Chat Completions with a system prompt loaded from file.
    """
    if openai is None:
        raise RuntimeError("OpenAI SDK not available. Install `pip install openai`.")

    # Using the Chat Completions API (python-openai v1.x)
    resp = openai.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": sysprompt},
            {"role": "user", "content": text},
        ],
    )
    out = resp.choices[0].message.content or ""
# Replace HTML entities with actual characters
    # Ensure HTML entities are correctly converted
    out = out.replace("<", "<").replace(">", ">").replace("&", "&")
    # Convert any other HTML-related special characters, if necessary
    # Add more replacements as needed for proper handling
    return out.replace("\r\n", "\n")

# ---------------------------
# Worker
# ---------------------------

def worker(sysprompt: str, model: str, max_retries: int = 5, retry_delay: float = 2.0):
    """
    Pulls (index, chunk_text) from queue, translates, and stores in memory.
    Retries on common transient errors (rate limit, timeouts).
    """
    while True:
        task = translation_queue.get()
        if task is None:
            translation_queue.task_done()
            break

        index, chunk_text = task  # chunk_text MUST be a str
        try:
            for attempt in range(max_retries):
                try:
                    translated = translate_text(chunk_text, sysprompt, model)

                    # in-memory store for final sorted outputs
                    with translated_texts_lock:
                        translated_texts[index] = translated

                    log(f"[OK] Chunk {index} ({len(chunk_text)} chars)")
                    break

                except Exception as e:
                    msg = str(e).lower()
                    if any(t in msg for t in ["rate limit", "overloaded", "timeout", "temporar"]):
                        wait_s = retry_delay * (2 ** attempt) + random.uniform(0, 1)
                        log(f"[Retry] {index}: {e} → waiting {wait_s:.2f}s")
                        time.sleep(wait_s)
                        continue
                    else:
                        log(f"[ERR] {index}: {e} (non-retryable)")
                        break
        finally:
            translation_queue.task_done()

# ---------------------------
# Main
# ---------------------------

def main():
    global translation_queue

    ap = argparse.ArgumentParser(description="Translate TMX or XLIFF files in chunks and write bilingual output.")
    ap.add_argument("--input", required=True, help="Path to source TMX or XLIFF")
    ap.add_argument("--sysprompt-file", required=True, help="Path to system prompt TXT")
    ap.add_argument("--encoding", default="utf-8", help="Source encoding (default: utf-8)")
    ap.add_argument("--chunk-size", type=int, default=6000, help="Max chars per chunk (default: 6000)")
    ap.add_argument("--threads", type=int, default=4, help="Worker threads (default: 4)")
    ap.add_argument("--model", default="gpt-5", help="OpenAI model name (default: gpt-5)")

    args = ap.parse_args()

    # Sanity check the API key
    if not test_openai_api_key(args.model):
        sys.exit(1)

    if not os.path.isfile(args.input):
        print(f"Input not found: {args.input}", file=sys.stderr)
        sys.exit(2)
    if not os.path.isfile(args.sysprompt_file):
        print(f"Sysprompt file not found: {args.sysprompt_file}", file=sys.stderr)
        sys.exit(2)

    sysprompt = read_text(args.sysprompt_file, "utf-8").strip()
    if not sysprompt:
        print("Sysprompt file is empty.", file=sys.stderr)
        sys.exit(2)

    # Parse segments from TMX or XLIFF, extracting target elements
    tree, segments = parse_tmx_xliff(args.input)

    if not segments:
        print("No segments found in the input file.", file=sys.stderr)
        sys.exit(1)

    # Translate and update the targets directly in the XML tree
    translation_queue = queue.Queue()
    workers: List[threading.Thread] = []
    for _ in range(max(1, args.threads)):
        t = threading.Thread(target=worker, args=(sysprompt, args.model), daemon=True)
        t.start()
        workers.append(t)

    # Enqueue tasks — ONLY pass the source text (string), not the Element!
    for index, (_, source_text) in enumerate(segments):
        # Optional: chunking by args.chunk_size if needed; here we send whole source_text.
        translation_queue.put((index, source_text))

    # Wait for all tasks
    translation_queue.join()

    # Stop workers
    for _ in workers:
        translation_queue.put(None)
    for t in workers:
        t.join()

    # Update the tree with translated segments
    for idx, (target_element, _) in enumerate(segments):
        if idx in translated_texts:
            translated_text = translated_texts[idx]
            if target_element is not None:
                target_element.text = translated_text

    # Save the modified XML back to a file
    # Preserve extension, add _translated before it; fallback to suffix if no known ext.
    m = re.search(r'\.(tmx|xlf|xliff)$', args.input, flags=re.IGNORECASE)
    if m:
        ext = m.group(1)
        out_path = re.sub(r'\.(tmx|xlf|xliff)$', fr'_translated.{ext}', args.input, flags=re.IGNORECASE)
    else:
        out_path = args.input + "_translated"

    tree.write(out_path, encoding='utf-8', xml_declaration=True)
    print(f"Translated file saved to: {out_path}")

if __name__ == "__main__":
    main()
