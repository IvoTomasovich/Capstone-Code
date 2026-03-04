"""
compare_only.py — Compare two transcripts and calculate accuracy metrics.
No correction algorithm is run. This purely diffs two files.

Usage:
    1. Set CORRECTED_FILE (system output) and GROUND_TRUTH_FILE below
    2. Optionally set ORIGINAL_FILE if you want to see what the errors were
    3. Run:  python compare_only.py

Output:
    A text file with precision, recall, accuracy, and detailed error lists.

Typical use case:
    - Run correct_only.py to produce a corrected transcript
    - Run compare_only.py to score that output against ground truth
"""

import json
import os
from datetime import datetime
from difflib import SequenceMatcher
from typing import Dict, List


# ============================================================================
# CONFIGURATION — SET THESE PATHS
# ============================================================================

# The original transcript WITH errors (before any corrections)
ORIGINAL_FILE = "combined_transcripts_incorrect.json"

# The corrected transcript (output from correct_only.py or any other system)
CORRECTED_FILE = "corrected_output.txt"

# The ground truth transcript (human-verified correct version)
GROUND_TRUTH_FILE = "combined_transcripts_correct.json"

# Where to write the comparison report
OUTPUT_REPORT = "comparison_report.txt"


# ============================================================================
# TRANSCRIPT LOADING
# ============================================================================

def load_transcript(file_path: str) -> str:
    """
    Load a transcript from a JSON or TXT file and return the full text.
    """
    if file_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    text_parts = []

    try:
        data = json.loads(content)
        if isinstance(data, list):
            for segment in data:
                if "text" in segment:
                    text_parts.append(segment["text"].strip())
        elif isinstance(data, dict) and "text" in data:
            text_parts.append(data["text"].strip())
    except json.JSONDecodeError:
        for line in content.split("\n"):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if "text" in obj:
                    text_parts.append(obj["text"].strip())
            except json.JSONDecodeError:
                continue

    if not text_parts:
        return content

    return " ".join(text_parts)


# ============================================================================
# DIFF COMPARISON
# ============================================================================

def compare_texts(original: str, other: str) -> List[Dict]:
    """Word-level diff between two texts."""
    original_words = original.split()
    other_words = other.split()

    matcher = SequenceMatcher(None, original_words, other_words)
    changes = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "replace":
            orig_text = " ".join(original_words[i1:i2])
            other_text = " ".join(other_words[j1:j2])

            if orig_text.strip(".,!?;:'\"()[]{}").lower() == other_text.strip(".,!?;:'\"()[]{}").lower():
                continue

            if len(orig_text.split()) > 2:
                sim = SequenceMatcher(
                    None,
                    orig_text.strip(".,!?;:'\"()[]{}").lower(),
                    other_text.strip(".,!?;:'\"()[]{}").lower(),
                ).ratio()
                if sim < 0.3:
                    continue

            ctx_start = max(0, j1 - 5)
            ctx_end = min(len(other_words), j2 + 5)

            changes.append({
                "original": orig_text,
                "corrected": other_text,
                "context": " ".join(other_words[ctx_start:ctx_end]),
                "position": i1,
            })

        elif tag == "delete":
            orig_text = " ".join(original_words[i1:i2])
            ctx_start = max(0, j1 - 5)
            ctx_end = min(len(other_words), j1 + 5)

            changes.append({
                "original": orig_text,
                "corrected": "[DELETED]",
                "context": " ".join(other_words[ctx_start:ctx_end]),
                "position": i1,
            })

        elif tag == "insert":
            other_text = " ".join(other_words[j1:j2])
            ctx_start = max(0, j1 - 5)
            ctx_end = min(len(other_words), j2 + 5)

            changes.append({
                "original": "[INSERTED]",
                "corrected": other_text,
                "context": " ".join(other_words[ctx_start:ctx_end]),
                "position": i1,
            })

    return changes


# ============================================================================
# ACCURACY CALCULATION
# ============================================================================

def calculate_accuracy(original: str, corrected: str, ground_truth: str) -> Dict:
    """Compare corrected output against ground truth."""

    true_corrections = compare_texts(original, ground_truth)
    system_corrections = compare_texts(original, corrected)

    true_positives = []
    false_negatives = []
    false_positives = []

    system_by_pos = {c["position"]: c for c in system_corrections}

    for true_change in true_corrections:
        pos = true_change["position"]

        if pos in system_by_pos:
            system_change = system_by_pos[pos]

            if system_change["corrected"].lower().strip() == true_change["corrected"].lower().strip():
                true_positives.append({
                    "position": pos,
                    "original": true_change["original"],
                    "ground_truth": true_change["corrected"],
                    "system": system_change["corrected"],
                    "context": system_change["context"],
                    "status": "CORRECT",
                })
            else:
                false_negatives.append({
                    "position": pos,
                    "original": true_change["original"],
                    "should_be": true_change["corrected"],
                    "system_said": system_change["corrected"],
                    "context": system_change["context"],
                    "status": "WRONG_FIX",
                })
        else:
            false_negatives.append({
                "position": pos,
                "original": true_change["original"],
                "should_be": true_change["corrected"],
                "system_said": true_change["original"],
                "context": true_change["context"],
                "status": "MISSED",
            })

    true_positions = {c["position"] for c in true_corrections}
    for system_change in system_corrections:
        if system_change["position"] not in true_positions:
            false_positives.append({
                "position": system_change["position"],
                "original": system_change["original"],
                "system_changed_to": system_change["corrected"],
                "context": system_change["context"],
                "status": "UNNECESSARY_CHANGE",
            })

    total_errors = len(true_corrections)
    correctly_fixed = len(true_positives)

    accuracy = (correctly_fixed / total_errors * 100) if total_errors > 0 else 0
    precision = (
        (correctly_fixed / (correctly_fixed + len(false_positives))) * 100
        if (correctly_fixed + len(false_positives)) > 0
        else 0
    )
    recall = (correctly_fixed / total_errors * 100) if total_errors > 0 else 0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "total_errors": total_errors,
        "correctly_fixed": correctly_fixed,
        "missed": len([fn for fn in false_negatives if fn["status"] == "MISSED"]),
        "wrong_fixes": len([fn for fn in false_negatives if fn["status"] == "WRONG_FIX"]),
        "false_positives": len(false_positives),
        "true_positives": true_positives,
        "false_negatives": false_negatives,
        "false_positives_list": false_positives,
    }


# ============================================================================
# REPORT GENERATION
# ============================================================================

def generate_report(metrics: Dict, original_path: str, corrected_path: str, truth_path: str) -> str:
    """Generate the comparison report."""

    lines = []
    lines.append("=" * 80)
    lines.append("TRANSCRIPT COMPARISON — ACCURACY REPORT")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"Date:               {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Original (errors):  {original_path}")
    lines.append(f"Corrected (system): {corrected_path}")
    lines.append(f"Ground truth:       {truth_path}")
    lines.append(f"Method:             Pure comparison (no correction algorithm run)")
    lines.append("")

    lines.append("ACCURACY METRICS")
    lines.append("-" * 80)
    lines.append(f"Total errors in original:  {metrics['total_errors']}")
    lines.append(f"Correctly fixed:           {metrics['correctly_fixed']}")
    lines.append(f"Missed:                    {metrics['missed']}")
    lines.append(f"Wrong fixes:               {metrics['wrong_fixes']}")
    lines.append(f"False positives:           {metrics['false_positives']}")
    lines.append("")
    lines.append(f"Accuracy (recall):         {metrics['accuracy']:.2f}%")
    lines.append(f"Precision:                 {metrics['precision']:.2f}%")
    lines.append(f"Recall:                    {metrics['recall']:.2f}%")
    lines.append("")

    if metrics["true_positives"]:
        lines.append("CORRECTLY FIXED ERRORS")
        lines.append("-" * 80)
        for i, tp in enumerate(metrics["true_positives"], 1):
            lines.append(f"  {i}. '{tp['original']}' -> '{tp['ground_truth']}'")
            if tp.get("context"):
                ctx = tp["context"]
                if len(ctx) > 120:
                    ctx = ctx[:60] + " ... " + ctx[-60:]
                lines.append(f"     Context: ...{ctx}...")
        lines.append("")

    if metrics["false_negatives"]:
        lines.append("MISSED ERRORS")
        lines.append("-" * 80)
        for i, fn in enumerate(metrics["false_negatives"], 1):
            status = "MISSED" if fn["status"] == "MISSED" else "WRONG FIX"
            lines.append(f"  {i}. [{status}] '{fn['original']}'")
            lines.append(f"     Should be:    '{fn['should_be']}'")
            lines.append(f"     System said:  '{fn['system_said']}'")
            if fn.get("context"):
                ctx = fn["context"]
                if len(ctx) > 120:
                    ctx = ctx[:60] + " ... " + ctx[-60:]
                lines.append(f"     Context: ...{ctx}...")
        lines.append("")

    if metrics["false_positives_list"]:
        filtered = [
            fp for fp in metrics["false_positives_list"]
            if fp["original"] != "[INSERTED]"
            and fp["system_changed_to"] != "[DELETED]"
            and len(fp["original"]) <= 200
            and len(fp["system_changed_to"]) <= 200
        ]
        if filtered:
            lines.append("FALSE POSITIVES (UNNECESSARY CHANGES)")
            lines.append("-" * 80)
            for i, fp in enumerate(filtered, 1):
                lines.append(f"  {i}. '{fp['original']}' (was correct)")
                lines.append(f"     Changed to: '{fp['system_changed_to']}'")
            lines.append("")

    lines.append("=" * 80)
    lines.append("END OF REPORT")
    lines.append("=" * 80)

    return "\n".join(lines)


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 60)
    print("TRANSCRIPT COMPARISON (NO CORRECTION)")
    print("=" * 60)

    # Load all three files
    print(f"\nLoading original (errors):  {ORIGINAL_FILE}")
    original_text = load_transcript(ORIGINAL_FILE)
    print(f"  -> {len(original_text.split())} words")

    print(f"Loading corrected (system): {CORRECTED_FILE}")
    corrected_text = load_transcript(CORRECTED_FILE)
    print(f"  -> {len(corrected_text.split())} words")

    print(f"Loading ground truth:       {GROUND_TRUTH_FILE}")
    ground_truth_text = load_transcript(GROUND_TRUTH_FILE)
    print(f"  -> {len(ground_truth_text.split())} words")

    # Calculate accuracy
    print("\nComparing corrected output against ground truth...")
    metrics = calculate_accuracy(original_text, corrected_text, ground_truth_text)

    print(f"\n  Accuracy:  {metrics['accuracy']:.2f}%")
    print(f"  Precision: {metrics['precision']:.2f}%")
    print(f"  Recall:    {metrics['recall']:.2f}%")
    print(f"  Fixed {metrics['correctly_fixed']} / {metrics['total_errors']} errors")

    # Write report
    report = generate_report(metrics, ORIGINAL_FILE, CORRECTED_FILE, GROUND_TRUTH_FILE)

    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\nReport written to: {OUTPUT_REPORT}")
    print("Done.")


if __name__ == "__main__":
    main()