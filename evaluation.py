"""
evaluate.py — Standalone accuracy evaluation script.

Runs the three-stage fuzzy correction pipeline on an error transcript,
compares the result against a ground truth transcript, and writes a
detailed accuracy report to a text file.

Usage:
    1. Set the three file paths below (ERROR_TRANSCRIPT, GROUND_TRUTH, OUTPUT_REPORT)
    2. Make sure correction.py and the dictionary JSON files are in the same directory
    3. Run:  python evaluate.py

Output:
    A text file (OUTPUT_REPORT) containing precision, recall, accuracy,
    and detailed breakdowns of every correction, miss, and false positive.
"""

import json
import os
from datetime import datetime
from difflib import SequenceMatcher
from typing import Dict, List

from correction import load_dictionaries, correct_transcript


# ============================================================================
# CONFIGURATION — SET THESE PATHS
# ============================================================================

# Path to the transcript WITH errors (what Whisper produced)
ERROR_TRANSCRIPT = "combined_transcripts_incorrect.json"

# Path to the ground truth transcript (human-verified correct version)
GROUND_TRUTH = "combined_transcripts_correct.json"

# Path where the accuracy report will be written
OUTPUT_REPORT = "accuracy_report.txt"

# Dictionary file paths (relative to this script)
ENGLISH_WORDS = "english_words.json"
NOLA_NAMES = "nola_names.json"
NOLA_STREETS = "nola_streets.json"


# ============================================================================
# TRANSCRIPT LOADING
# ============================================================================

def load_transcript(file_path: str) -> str:
    """
    Load a transcript from a JSON or TXT file and return the full text.

    Supports:
      - JSON array of segments with 'text' fields
      - JSON object with a 'text' field
      - JSONL (one JSON object per line, each with 'text')
      - Plain text (.txt)
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
        # Try JSONL format
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
        # Fall back to treating it as plain text
        return content

    return " ".join(text_parts)


# ============================================================================
# DIFF COMPARISON
# ============================================================================

def compare_texts_with_diff(original: str, corrected: str) -> List[Dict]:
    """
    Word-level diff between two texts.
    Returns a list of changes, each with: original, corrected, position, context.
    """
    original_words = original.split()
    corrected_words = corrected.split()

    matcher = SequenceMatcher(None, original_words, corrected_words)
    changes = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "replace":
            orig_text = " ".join(original_words[i1:i2])
            corr_text = " ".join(corrected_words[j1:j2])

            # Skip pure case differences
            if orig_text.strip(".,!?;:'\"()[]{}").lower() == corr_text.strip(".,!?;:'\"()[]{}").lower():
                continue

            # Skip wildly dissimilar multi-word replacements (alignment artifacts)
            if len(orig_text.split()) > 2:
                sim = SequenceMatcher(
                    None,
                    orig_text.strip(".,!?;:'\"()[]{}").lower(),
                    corr_text.strip(".,!?;:'\"()[]{}").lower(),
                ).ratio()
                if sim < 0.3:
                    continue

            ctx_start = max(0, j1 - 5)
            ctx_end = min(len(corrected_words), j2 + 5)
            context = " ".join(corrected_words[ctx_start:ctx_end])

            changes.append({
                "original": orig_text,
                "corrected": corr_text,
                "context": context,
                "position": i1,
            })

        elif tag == "delete":
            orig_text = " ".join(original_words[i1:i2])
            ctx_start = max(0, j1 - 5)
            ctx_end = min(len(corrected_words), j1 + 5)
            context = " ".join(corrected_words[ctx_start:ctx_end])

            changes.append({
                "original": orig_text,
                "corrected": "[DELETED]",
                "context": context,
                "position": i1,
            })

        elif tag == "insert":
            corr_text = " ".join(corrected_words[j1:j2])
            ctx_start = max(0, j1 - 5)
            ctx_end = min(len(corrected_words), j2 + 5)
            context = " ".join(corrected_words[ctx_start:ctx_end])

            changes.append({
                "original": "[INSERTED]",
                "corrected": corr_text,
                "context": context,
                "position": i1,
            })

    return changes


# ============================================================================
# ACCURACY CALCULATION
# ============================================================================

def calculate_accuracy(original: str, system_corrected: str, ground_truth: str) -> Dict:
    """
    Compare system output against ground truth to compute accuracy metrics.

    Returns dict with: accuracy, precision, recall, and detailed lists of
    true positives, false negatives, and false positives.
    """
    # The "answer key" — every error that should be fixed
    true_corrections = compare_texts_with_diff(original, ground_truth)

    # What the system actually changed
    system_corrections = compare_texts_with_diff(original, system_corrected)

    true_positives = []
    false_negatives = []
    false_positives = []

    # Index system corrections by position for fast lookup
    system_by_pos = {c["position"]: c for c in system_corrections}

    # For each real error, check what the system did
    for true_change in true_corrections:
        pos = true_change["position"]

        if pos in system_by_pos:
            system_change = system_by_pos[pos]

            if system_change["corrected"].lower().strip() == true_change["corrected"].lower().strip():
                # System found and correctly fixed this error
                true_positives.append({
                    "position": pos,
                    "original": true_change["original"],
                    "ground_truth": true_change["corrected"],
                    "system": system_change["corrected"],
                    "context": system_change["context"],
                    "status": "CORRECT",
                })
            else:
                # System changed this position but to the wrong value
                false_negatives.append({
                    "position": pos,
                    "original": true_change["original"],
                    "should_be": true_change["corrected"],
                    "system_said": system_change["corrected"],
                    "context": system_change["context"],
                    "status": "WRONG_FIX",
                })
        else:
            # System didn't touch this position — missed the error
            false_negatives.append({
                "position": pos,
                "original": true_change["original"],
                "should_be": true_change["corrected"],
                "system_said": true_change["original"],
                "context": true_change["context"],
                "status": "MISSED",
            })

    # Find false positives — system changes at positions with no real error
    true_positions = {c["position"] for c in true_corrections}
    for system_change in system_corrections:
        pos = system_change["position"]
        if pos not in true_positions:
            false_positives.append({
                "position": pos,
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

def generate_report(
    original_text: str,
    corrected_text: str,
    ground_truth_text: str,
    corrections: List[Dict],
    metrics: Dict,
    error_path: str,
    truth_path: str,
) -> str:
    """Generate the full accuracy report as a string."""

    lines = []
    lines.append("=" * 80)
    lines.append("TRANSCRIPT CORRECTION — ACCURACY EVALUATION REPORT")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"Date:               {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Error transcript:   {error_path}")
    lines.append(f"Ground truth:       {truth_path}")
    lines.append(f"Correction method:  Three-stage fuzzy pipeline (no LLM)")
    lines.append("")

    # --- Document stats ---
    lines.append("DOCUMENT STATISTICS")
    lines.append("-" * 80)
    lines.append(f"Original word count:     {len(original_text.split())}")
    lines.append(f"Corrected word count:    {len(corrected_text.split())}")
    lines.append(f"Ground truth word count: {len(ground_truth_text.split())}")
    lines.append("")

    # --- Correction summary ---
    hardcoded = [c for c in corrections if c.get("type", "").startswith("HARDCODED")]
    streets = [c for c in corrections if c.get("type", "").startswith("STREET")]
    names = [c for c in corrections if c.get("type", "").startswith("NAME")]

    lines.append("CORRECTIONS APPLIED")
    lines.append("-" * 80)
    lines.append(f"Stage 0 (Hardcoded):     {len(hardcoded)}")
    lines.append(f"Stage 1 (Streets):       {len(streets)}")
    lines.append(f"Stage 2 (Names):         {len(names)}")
    lines.append(f"Total corrections:       {len(corrections)}")
    lines.append("")

    if corrections:
        for i, c in enumerate(corrections, 1):
            lines.append(f"  {i}. [{c.get('type', '?')}] '{c['original']}' -> '{c['corrected']}'")
            if c.get("context"):
                ctx = c["context"]
                if len(ctx) > 120:
                    ctx = ctx[:60] + " ... " + ctx[-60:]
                lines.append(f"     Context: ...{ctx}...")
        lines.append("")

    # --- Accuracy metrics ---
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

    if metrics["accuracy"] > 80:
        lines.append(f"Assessment:                EXCELLENT")
    elif metrics["accuracy"] > 60:
        lines.append(f"Assessment:                GOOD")
    elif metrics["accuracy"] > 40:
        lines.append(f"Assessment:                FAIR")
    else:
        lines.append(f"Assessment:                NEEDS IMPROVEMENT")
    lines.append("")

    # --- True positives ---
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

    # --- False negatives (missed + wrong fixes) ---
    if metrics["false_negatives"]:
        lines.append("MISSED ERRORS")
        lines.append("-" * 80)
        for i, fn in enumerate(metrics["false_negatives"], 1):
            status_label = "MISSED" if fn["status"] == "MISSED" else "WRONG FIX"
            lines.append(f"  {i}. [{status_label}] '{fn['original']}'")
            lines.append(f"     Should be:    '{fn['should_be']}'")
            lines.append(f"     System said:  '{fn['system_said']}'")
            if fn.get("context"):
                ctx = fn["context"]
                if len(ctx) > 120:
                    ctx = ctx[:60] + " ... " + ctx[-60:]
                lines.append(f"     Context: ...{ctx}...")
        lines.append("")

    # --- False positives ---
    if metrics["false_positives_list"]:
        # Filter out noisy entries
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
                if fp.get("context"):
                    ctx = fp["context"]
                    if len(ctx) > 120:
                        ctx = ctx[:60] + " ... " + ctx[-60:]
                    lines.append(f"     Context: ...{ctx}...")
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
    print("TRANSCRIPT CORRECTION — ACCURACY EVALUATION")
    print("=" * 60)

    # Load dictionaries
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dicts = load_dictionaries(
        english_path=os.path.join(script_dir, ENGLISH_WORDS),
        names_path=os.path.join(script_dir, NOLA_NAMES),
        streets_path=os.path.join(script_dir, NOLA_STREETS),
    )

    # Load transcripts
    print(f"\nLoading error transcript: {ERROR_TRANSCRIPT}")
    original_text = load_transcript(ERROR_TRANSCRIPT)
    print(f"  -> {len(original_text.split())} words")

    print(f"Loading ground truth:     {GROUND_TRUTH}")
    ground_truth_text = load_transcript(GROUND_TRUTH)
    print(f"  -> {len(ground_truth_text.split())} words")

    # Run correction pipeline
    print("\nRunning correction pipeline...")
    corrected_text, corrections = correct_transcript(original_text, dicts)
    print(f"  -> {len(corrections)} corrections applied")

    # Calculate accuracy
    print("\nCalculating accuracy against ground truth...")
    metrics = calculate_accuracy(original_text, corrected_text, ground_truth_text)

    print(f"\n  Accuracy:  {metrics['accuracy']:.2f}%")
    print(f"  Precision: {metrics['precision']:.2f}%")
    print(f"  Recall:    {metrics['recall']:.2f}%")
    print(f"  Fixed {metrics['correctly_fixed']} / {metrics['total_errors']} errors")

    # Generate and write report
    report = generate_report(
        original_text,
        corrected_text,
        ground_truth_text,
        corrections,
        metrics,
        ERROR_TRANSCRIPT,
        GROUND_TRUTH,
    )

    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\nReport written to: {OUTPUT_REPORT}")
    print("Done.")


if __name__ == "__main__":
    main()