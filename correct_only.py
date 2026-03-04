"""
correct_only.py — Run the correction pipeline on a transcript and save
the corrected output. No evaluation, no ground truth needed.

Usage:
    1. Set INPUT_FILE and OUTPUT_FILE below
    2. Make sure correction.py and dictionary JSON files are in the same directory
    3. Run:  python correct_only.py

Output:
    A plain text file containing the corrected transcript.
"""

import json
import os
from correction import load_dictionaries, correct_transcript


# ============================================================================
# CONFIGURATION — SET THESE PATHS
# ============================================================================

# Path to the transcript to correct
INPUT_FILE = "combined_transcripts_incorrect.json"

# Where to save the corrected output
OUTPUT_FILE = "corrected_output_solo_script.txt"

# Dictionary file paths
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
# MAIN
# ============================================================================

def main():
    print("=" * 60)
    print("TRANSCRIPT CORRECTION (NO EVALUATION)")
    print("=" * 60)

    # Load dictionaries
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dicts = load_dictionaries(
        english_path=os.path.join(script_dir, ENGLISH_WORDS),
        names_path=os.path.join(script_dir, NOLA_NAMES),
        streets_path=os.path.join(script_dir, NOLA_STREETS),
    )

    # Load transcript
    print(f"\nLoading transcript: {INPUT_FILE}")
    original_text = load_transcript(INPUT_FILE)
    print(f"  -> {len(original_text.split())} words")

    # Run correction pipeline
    print("\nRunning correction pipeline...")
    corrected_text, corrections = correct_transcript(original_text, dicts)

    print(f"\n  {len(corrections)} corrections applied")

    # Summary of what was changed
    if corrections:
        print("\n  Corrections made:")
        for i, c in enumerate(corrections, 1):
            print(f"    {i}. [{c.get('type', '?')}] '{c['original']}' -> '{c['corrected']}'")

    # Save corrected output
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(corrected_text)

    print(f"\nCorrected transcript saved to: {OUTPUT_FILE}")
    print("Done.")


if __name__ == "__main__":
    main()