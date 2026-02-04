import gradio as gr
import json
import pandas as pd
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import re
from difflib import SequenceMatcher
from datetime import datetime
import os
import shutil
from typing import Dict, List, Tuple, Optional


# ============================================================================
# GLOBAL VARIABLES
# ============================================================================

# Global variables for model and dictionaries (lazy loading)
model_local = None
dicts = None
latest_report = {"report": "", "original": "", "corrected": ""}
latest_accuracy = {"summary": "", "report": "", "has_data": False}


# ============================================================================
# SYSTEM INITIALIZATION
# ============================================================================

def initialize_system():
    """Initialize model and dictionaries once (lazy loading)"""
    global model_local, dicts
    
    if model_local is None:
        print("🔄 Initializing ChatOllama model...")
        model_local = ChatOllama(
            model="llama3.1",
            temperature=0,
            num_ctx=8192,
            top_p=1.0,
        )
    
    if dicts is None:
        print("📚 Loading dictionaries...")
        dicts = load_dictionaries()
    
    return model_local, dicts


# ============================================================================
# DICTIONARY LOADING
# ============================================================================

def load_dictionaries():
    """Load 3 specialized dictionaries"""
    
    # 1. English common words
    try:
        with open('words_dictionary.json', 'r', encoding='utf-8') as f:
            english_words = set(word.lower() for word in json.load(f))
        print(f"✅ Loaded {len(english_words)} English words")
    except FileNotFoundError:
        print("⚠️ words_dictionary.json not found - using minimal set")
        english_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'has', 'have', 'had',
            'was', 'were', 'been', 'is', 'are', 'weekend', 'seconded',
            'representing', 'resident', 'community', 'on', 'behalf'
        }
    
    # 2. New Orleans names (format: {"first_names": [...], "last_names": [...]})
    try:
        with open('nola_names.json', 'r', encoding='utf-8') as f:
            names_data = json.load(f)
        
        first_names = names_data.get('first_names', [])
        last_names = names_data.get('last_names', [])
        names = first_names + last_names
        
        print(f"✅ Loaded {len(first_names)} first names and {len(last_names)} last names")
        print(f"   Total: {len(names)} name entries")
    except FileNotFoundError:
        print("⚠️ nola_names.json not found - using sample set")
        names = ["Moreno", "Palmer", "Cantrell", "Helena"]
        first_names = []
        last_names = []
    
    # 3. New Orleans streets (array format)
    try:
        with open('nola_streets.json', 'r', encoding='utf-8') as f:
            streets = json.load(f)
        print(f"✅ Loaded {len(streets)} street names")
    except FileNotFoundError:
        print("⚠️ nola_streets.json not found - using sample set")
        streets = ["Claiborne Avenue", "Canal Street", "Tchoupitoulas Street"]
    
    return {
        'english': english_words,
        'streets': streets,
        'names': names,
        'first_names': set(fn.lower() for fn in first_names),
        'last_names': set(ln.lower() for ln in last_names)
    }


# ============================================================================
# LLM CORRECTION WITH CHANGE TRACKING
# ============================================================================

def find_potential_name_errors(text: str, dicts: Dict) -> Tuple[List[str], List[str]]:
    """
    Find words that look like misspelled names by fuzzy matching
    
    Returns: (high_confidence_errors, low_confidence_errors)
    """
    from difflib import get_close_matches, SequenceMatcher
    
    words = text.split()
    high_confidence = []
    low_confidence = []
    
    # Look for words after titles that might be names
    title_words = ['mayor', 'councilmember', 'commissioner', 'dr.', 'mr.', 'ms.', 'mrs.', 'madam', 'miss', 'professor', 'teacher', 'senator', 'representative', 'director']
    
    # Get the original name lists (not the lowercase sets)
    all_names = dicts['names']
    
    for i, word in enumerate(words):
        if word.lower() in title_words and i + 1 < len(words):
            potential_name = words[i + 1].strip('.,!?;:\'"()[]{}')
            
            # Check if this name is NOT exactly in our reference (case-insensitive)
            if potential_name.lower() not in [n.lower() for n in all_names]:
                # Use fuzzy matching to find close matches
                close_matches = get_close_matches(
                    potential_name, 
                    all_names,
                    n=1, 
                    cutoff=0.70  # Lower threshold to catch more
                )
                
                if close_matches:
                    match = close_matches[0]
                    # Calculate similarity ratio
                    similarity = SequenceMatcher(None, potential_name.lower(), match.lower()).ratio()
                    
                    error_str = f"{potential_name} → {match}"
                    
                    # High confidence: >78% similar (1-2 letter difference)
                    if similarity > 0.78:
                        high_confidence.append(error_str)
                    else:
                        low_confidence.append(error_str)
    
    return high_confidence, low_confidence


def llm_correct_with_tracking(text: str, dicts: Dict, llm) -> Tuple[str, List[Dict]]:
    """
    
    Args:
        text: Original text to correct
        dicts: Dictionary containing names, streets, english words
        llm: LLM instance
    
    Returns:
        (corrected_text, list_of_changes)
    """
    
    print("\n🤖 LLM CORRECTION PIPELINE")
    print("="*80)
    
    # SAFETY CHECK: Don't process extremely large texts
    word_count = len(text.split())
    if word_count > 3000:
        print(f"   ⚠️ ERROR: Text too large ({word_count} words)")
        print(f"   Maximum supported: 3000 words")
        print(f"   Suggestion: The document is too large to correct in one pass")
        print(f"   Using first 3000 words only...")
        # Truncate to first 3000 words
        words = text.split()[:3000]
        text = ' '.join(words)
        word_count = 3000
    
    # Sample glossary terms to include in prompt
    names_sample = ", ".join(dicts['names'])  
    streets_sample = ", ".join(dicts['streets'])
    
    # Find potential name errors using fuzzy matching
    high_conf_errors, low_conf_errors = find_potential_name_errors(text, dicts)
    
    # DEBUG: Print what fuzzy search found
    print(f"\n🔍 FUZZY SEARCH RESULTS:")
    print(f"   High confidence errors found: {len(high_conf_errors)}")
    for error in high_conf_errors:
        print(f"      - {error}")
    print(f"   Low confidence errors found: {len(low_conf_errors)}")
    for error in low_conf_errors[:10]:  # Show first 10
        print(f"      - {error}")
    
    # DEBUG: Check for Rogers/Harrison in hints
    print(f"\n🔍 Checking hints for 'Rogers' or 'Harrison':")
    found_rogers = False
    found_harrison = False
    for error in high_conf_errors + low_conf_errors:
        if 'rogers' in error.lower():
            print(f"   ✓ Found: {error}")
            found_rogers = True
        if 'harrison' in error.lower():
            print(f"   ✓ Found: {error}")
            found_harrison = True
    if not found_rogers and not found_harrison:
        print(f"   ✗ Neither 'Rogers' nor 'Harrison' found in hints")
    
    # DEBUG: Check context around position 763
    words = text.split()
    if len(words) > 763:
        context_start = max(0, 763 - 30)
        context_end = min(len(words), 763 + 30)
        context_words = words[context_start:context_end]
        
        print(f"\n🔍 CONTEXT AROUND POSITION 763:")
        for i, word in enumerate(context_words, start=context_start):
            marker = " ← TARGET" if i == 763 else ""
            print(f"   [{i}] {word}{marker}")
        
        # Check if Harrison appears nearby
        context_str = ' '.join(context_words).lower()
        if 'harrison' in context_str:
            print(f"\n   ⚠️ WARNING: 'Harrison' appears in nearby context!")
            harrison_positions = [i for i, w in enumerate(context_words, start=context_start) 
                                 if 'harrison' in w.lower()]
            print(f"   Harrison found at positions: {harrison_positions}")
    
    # Build hint section based on confidence
    hints_section = ""
    
    if high_conf_errors:
        hints_section += f"""
🚨 DEFINITE TRANSCRIPTION ERRORS (MUST FIX):
{chr(10).join(f"- {error}" for error in high_conf_errors)}

These are clear misspellings. Replace the left side with the right side.
"""
    
    if low_conf_errors:
        hints_section += f"""
POSSIBLE TRANSCRIPTION ERRORS (CHECK CAREFULLY):
{chr(10).join(f"- {error}" for error in low_conf_errors[:10])}

These might be errors - verify against context before correcting.
"""
    
    # Build comprehensive prompt
    prompt = f"""You are correcting transcription errors in New Orleans City Council meeting transcripts. Audio-to-text systems often mishear names and street names.

REFERENCE - PEOPLE NAMES IN NEW ORLEANS:
{names_sample}

REFERENCE - STREET NAMES IN NEW ORLEANS:
{streets_sample}
{hints_section}
INSTRUCTIONS:
1. FIRST: Fix all "DEFINITE TRANSCRIPTION ERRORS" listed above - these are confirmed mistakes
2. Look for names after titles like "Mayor", "Councilmember", "Commissioner" 
3. Check "POSSIBLE TRANSCRIPTION ERRORS" and fix if they make sense in context
4. Only correct names when you're confident it's an error
5. Do NOT change legitimate names that aren't in the reference list
6. Do NOT change common English words
7. Do NOT rewrite sentences
8. Keep everything else EXACTLY as it appears
9. This is an English transcript. Do NOT assume foreign spellings are correct
10. Check reference list for proper double consonant usage
11. New Orleans has French street names - check reference list for correct spelling
CRITICAL: Output ONLY the corrected text. DO NOT add any explanations, notes, or comments. Just the corrected text.

TEXT TO CORRECT:
{text}

CORRECTED TEXT:"""

    # DEBUG: Save prompt for inspection
    with open('/tmp/llm_prompt_debug.txt', 'w', encoding='utf-8') as f:
        f.write(prompt)
    print(f"\n📝 Full prompt saved to /tmp/llm_prompt_debug.txt")

    word_count = len(text.split())
    print(f"   📝 Sending {word_count} words to LLM...")
    
    # Warn if too large
    if word_count > 1000:
        print(f"   ⚠️ WARNING: Context is large ({word_count} words). This may take 2-5 minutes.")
    
    try:
        import time
        start_time = time.time()
        response = llm.invoke(prompt)
        
        # Extract text content from AIMessage object
        if hasattr(response, 'content'):
            corrected_text = response.content.strip()
        else:
            corrected_text = str(response).strip()
        
        elapsed = time.time() - start_time
        print(f"   ✅ LLM responded in {elapsed:.1f} seconds")
        
        # Remove any markdown formatting the LLM might add
        corrected_text = corrected_text.replace("```", "").strip()
        if corrected_text.startswith("CORRECTED TEXT:"):
            corrected_text = corrected_text.replace("CORRECTED TEXT:", "").strip()
        
        # Remove common LLM explanation patterns at the end
        explanation_patterns = [
            "this is the corrected text",
            "these names have been corrected",
            "note:",
            "corrections made:",
            "i have corrected",
            "the following corrections",
            "summary of corrections",
            "potential transcription errors",
            "addressed:",
        ]
        
        # Find the earliest occurrence of any explanation pattern
        earliest_idx = len(corrected_text)
        for pattern in explanation_patterns:
            idx = corrected_text.lower().find(pattern.lower())
            if idx != -1 and idx < earliest_idx:
                earliest_idx = idx
        
        # If we found an explanation, truncate there
        if earliest_idx < len(corrected_text):
            # Look backwards to find the last sentence-ending punctuation
            last_period = corrected_text.rfind('.', 0, earliest_idx)
            last_question = corrected_text.rfind('?', 0, earliest_idx)
            last_exclaim = corrected_text.rfind('!', 0, earliest_idx)
            
            last_punct = max(last_period, last_question, last_exclaim)
            
            if last_punct > 0:
                corrected_text = corrected_text[:last_punct + 1].strip()
                print(f"   🔧 Stripped LLM explanation from output")
        
        # Track changes by comparing original and corrected with PROPER DIFF
        changes = compare_texts_with_diff(text, corrected_text)
        
        # If NO changes were made, the LLM might be too conservative - warn user
        if len(changes) == 0:
            print(f"   ⚠️ WARNING: LLM made 0 corrections!")
            print(f"   This might mean:")
            print(f"   1. The text was already perfect")
            print(f"   2. The LLM is being too conservative")
            print(f"   3. The LLM didn't understand the prompt")
            print(f"   ")
            print(f"   💡 Suggestion: Review the input text to verify if errors exist")

        
        print(f"   ✅ LLM made {len(changes)} corrections")
        
        return corrected_text, changes
        
    except Exception as e:
        print(f"   ❌ LLM correction failed: {e}")
        print("   Returning original text unchanged")
        return text, []


def compare_texts_with_diff(original: str, corrected: str) -> List[Dict]:
    """
    Compare two texts using proper diff algorithm to identify actual changes
    
    This fixes the bug where position-based comparison would get out of sync
    and report false positives.
    
    Args:
        original: Original text
        corrected: Corrected text
    
    Returns:
        List of dicts with: {original, corrected, context}
    """
    
    original_words = original.split()
    corrected_words = corrected.split()
    
    # Use SequenceMatcher to find actual differences
    matcher = SequenceMatcher(None, original_words, corrected_words)
    changes = []
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'replace':
            # Word(s) were changed
            orig_text = ' '.join(original_words[i1:i2])
            corr_text = ' '.join(corrected_words[j1:j2])
            
            # Filter out punctuation-only changes
            orig_clean = orig_text.strip('.,!?;:\'"()[]{}').lower()
            corr_clean = corr_text.strip('.,!?;:\'"()[]{}').lower()
            
            # Skip if only punctuation changed
            if orig_clean == corr_clean:
                continue
            
            # Skip if words are too different (likely a rewrite, not a correction)
            word_count = len(orig_clean.split())
            if word_count > 2:
                similarity = SequenceMatcher(None, orig_clean, corr_clean).ratio()
                if similarity < 0.3:
                    continue
            
            # Get context (5 words before and after)
            context_start = max(0, j1 - 5)
            context_end = min(len(corrected_words), j2 + 5)
            context = ' '.join(corrected_words[context_start:context_end])
            
            changes.append({
                'original': orig_text,
                'corrected': corr_text,
                'context': context,
                'position': i1  # Keep for sorting/reference
            })
        
        elif tag == 'delete':
            # Word(s) were removed
            orig_text = ' '.join(original_words[i1:i2])
            
            # Get context
            context_start = max(0, j1 - 5)
            context_end = min(len(corrected_words), j1 + 5)
            context = ' '.join(corrected_words[context_start:context_end])
            
            changes.append({
                'original': orig_text,
                'corrected': '[DELETED]',
                'context': context,
                'position': i1
            })
        
        elif tag == 'insert':
            # Word(s) were added
            corr_text = ' '.join(corrected_words[j1:j2])
            
            # Get context
            context_start = max(0, j1 - 5)
            context_end = min(len(corrected_words), j2 + 5)
            context = ' '.join(corrected_words[context_start:context_end])
            
            changes.append({
                'original': '[INSERTED]',
                'corrected': corr_text,
                'context': context,
                'position': i1
            })
    
    return changes


# ============================================================================
# CHANGE REPORTING
# ============================================================================

def generate_correction_report(original_text: str, corrected_text: str, 
                               changes: List[Dict]) -> str:
    """Generate detailed correction report"""
    
    report = []
    report.append("=" * 80)
    report.append("📊 LLM CORRECTION REPORT")
    report.append("=" * 80)
    report.append("")
    
    # Summary Statistics
    report.append("📈 SUMMARY STATISTICS")
    report.append("-" * 80)
    report.append(f"Original text length:    {len(original_text)} characters")
    report.append(f"Corrected text length:   {len(corrected_text)} characters")
    report.append(f"Total words processed:   {len(original_text.split())} words")
    report.append(f"")
    report.append(f"TOTAL CHANGES MADE:      {len(changes)}")
    report.append("")
    
    # Detailed Changes
    if changes:
        report.append("🔧 DETAILED CORRECTIONS")
        report.append("-" * 80)
        
        for i, change in enumerate(changes, 1):
            report.append(f"\n{i}. Change at position {change['position']}:")
            report.append(f"   Original:  '{change['original']}'")
            report.append(f"   Corrected: '{change['corrected']}'")
            report.append(f"   Context:   ...{change['context']}...")
        
        report.append("")
    else:
        report.append("✅ NO CORRECTIONS NEEDED")
        report.append("-" * 80)
        report.append("The text was already correctly transcribed!")
        report.append("")
    
    # Text comparison
    report.append("📝 TEXT COMPARISON")
    report.append("-" * 80)
    report.append("\nORIGINAL (first 500 chars):")
    report.append(original_text[:500] + ("..." if len(original_text) > 500 else ""))
    report.append("\nCORRECTED (first 500 chars):")
    report.append(corrected_text[:500] + ("..." if len(corrected_text) > 500 else ""))
    report.append("")
    
    report.append("=" * 80)
    
    return "\n".join(report)


# ============================================================================
# ACCURACY EVALUATION
# ============================================================================

def calculate_accuracy_from_diffs(original: str, system_corrected: str, ground_truth: str) -> Dict:
    """
    Calculate accuracy by reusing compare_texts_with_diff (DRY principle)
    
    Args:
        original: Original text with errors
        system_corrected: Your system's corrected version
        ground_truth: Manually corrected ground truth
    
    Returns:
        Dictionary with accuracy metrics
    """
    
    # What SHOULD have been corrected (ground truth)
    true_corrections = compare_texts_with_diff(original, ground_truth)
    
    # What YOUR SYSTEM actually corrected
    system_corrections = compare_texts_with_diff(original, system_corrected)
    
    # Now compare the two lists
    true_positives = []
    false_negatives = []
    false_positives = []
    
    # Create lookup of system corrections by position
    system_by_pos = {change['position']: change for change in system_corrections}
    
    # Check each true error
    for true_change in true_corrections:
        pos = true_change['position']
        
        if pos in system_by_pos:
            system_change = system_by_pos[pos]
            
            # Compare what system corrected vs what it should be
            if system_change['corrected'].lower().strip() == true_change['corrected'].lower().strip():
                # Correct!
                true_positives.append({
                    'position': pos,
                    'original': true_change['original'],
                    'ground_truth': true_change['corrected'],
                    'system': system_change['corrected'],
                    'context': system_change['context'],
                    'status': 'CORRECT'
                })
            else:
                # System tried but got it wrong
                false_negatives.append({
                    'position': pos,
                    'original': true_change['original'],
                    'should_be': true_change['corrected'],
                    'system_said': system_change['corrected'],
                    'context': system_change['context'],
                    'status': 'WRONG_FIX'
                })
        else:
            # System didn't correct this error
            false_negatives.append({
                'position': pos,
                'original': true_change['original'],
                'should_be': true_change['corrected'],
                'system_said': true_change['original'],  # Unchanged
                'context': true_change['context'],
                'status': 'MISSED'
            })
    
    # Check for false positives (system changed something that shouldn't be changed)
    true_positions = {change['position'] for change in true_corrections}
    
    for system_change in system_corrections:
        pos = system_change['position']
        if pos not in true_positions:
            false_positives.append({
                'position': pos,
                'original': system_change['original'],
                'system_changed_to': system_change['corrected'],
                'context': system_change['context'],
                'status': 'UNNECESSARY_CHANGE'
            })
    
    # Calculate metrics
    total_errors = len(true_corrections)
    correctly_fixed = len(true_positives)
    
    accuracy = (correctly_fixed / total_errors * 100) if total_errors > 0 else 0
    precision = (correctly_fixed / (correctly_fixed + len(false_positives))) * 100 if (correctly_fixed + len(false_positives)) > 0 else 0
    recall = (correctly_fixed / total_errors * 100) if total_errors > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'total_errors': total_errors,
        'correctly_fixed': correctly_fixed,
        'missed': len([fn for fn in false_negatives if fn['status'] == 'MISSED']),
        'wrong_fixes': len([fn for fn in false_negatives if fn['status'] == 'WRONG_FIX']),
        'false_positives': len(false_positives),
        'true_positives': true_positives,
        'false_negatives': false_negatives,
        'false_positives_list': false_positives
    }


def generate_dual_accuracy_report(baseline_metrics: Dict, system_metrics: Dict, improvement: float) -> str:
    """Generate detailed accuracy report comparing baseline vs system"""
    
    report = []
    report.append("=" * 80)
    report.append("DUAL ACCURACY EVALUATION REPORT")
    report.append("=" * 80)
    report.append("")
    
    # Baseline Section
    report.append("BASELINE (Original Transcript - No Corrections)")
    report.append("-" * 80)
    report.append(f"Total Errors Detected:  {baseline_metrics['total_errors']}")
    report.append(f"Baseline Accuracy:      0.00% (no corrections applied)")
    report.append("")
    report.append("This represents how many errors exist in the original transcript.")
    report.append("")
    
    # System Performance Section
    report.append("SYSTEM PERFORMANCE (After Correction Pipeline)")
    report.append("-" * 80)
    report.append(f"Accuracy:          {system_metrics['accuracy']:.2f}%")
    report.append(f"Precision:         {system_metrics['precision']:.2f}%")
    report.append(f"Recall:            {system_metrics['recall']:.2f}%")
    report.append("")
    report.append(f"Total errors:      {system_metrics['total_errors']}")
    report.append(f"Correctly fixed:   {system_metrics['correctly_fixed']}")
    report.append(f"Missed:            {system_metrics['missed']}")
    report.append(f"Wrong fixes:       {system_metrics['wrong_fixes']}")
    report.append(f"False positives:   {system_metrics['false_positives']}")
    report.append("")
    
    # Improvement Section
    report.append("IMPROVEMENT ANALYSIS")
    report.append("-" * 80)
    report.append(f"Improvement:       +{improvement:.2f}%")
    report.append(f"Error Reduction:   {system_metrics['correctly_fixed']} / {system_metrics['total_errors']} errors fixed")
    
    if improvement > 80:
        report.append(f"Assessment:        EXCELLENT - System fixes most errors")
    elif improvement > 60:
        report.append(f"Assessment:        GOOD - System fixes majority of errors")
    elif improvement > 40:
        report.append(f"Assessment:        FAIR - System fixes some errors")
    else:
        report.append(f"Assessment:        NEEDS IMPROVEMENT - Many errors missed")
    
    report.append("")
    
    # Correctly Fixed - SHOW ALL
    if system_metrics['true_positives']:
        report.append("CORRECTLY FIXED ERRORS")
        report.append("-" * 80)
        for i, tp in enumerate(system_metrics['true_positives'], 1):  # Show ALL
            report.append(f"{i}. Position {tp['position']}:")
            report.append(f"   Original:     '{tp['original']}'")
            report.append(f"   Ground Truth: '{tp['ground_truth']}'")
            report.append(f"   System:       '{tp['system']}' ✓")
        report.append("")
    
    # Missed Errors - SHOW ALL
    if system_metrics['false_negatives']:
        report.append("❌ MISSED OR WRONGLY FIXED ERRORS")
        report.append("-" * 80)
        for i, fn in enumerate(system_metrics['false_negatives'], 1):  # Show ALL
            report.append(f"{i}. Position {fn['position']} - {fn['status']}:")
            report.append(f"   Original:     '{fn['original']}'")
            report.append(f"   Should be:    '{fn['should_be']}'")
            report.append(f"   System said:  '{fn['system_said']}'")
        report.append("")
    
    # False Positives - SHOW ALL
    if system_metrics['false_positives_list']:
        report.append("FALSE POSITIVES (Unnecessary Changes)")
        report.append("-" * 80)
        for i, fp in enumerate(system_metrics['false_positives_list'], 1):  # Show ALL
            report.append(f"{i}. Position {fp['position']}:")
            report.append(f"   Original (correct): '{fp['original']}'")
            report.append(f"   System changed to:  '{fp['system_changed_to']}'")
        report.append("")
    
    report.append("=" * 80)
    
    return "\n".join(report)


def get_latest_accuracy():
    """Retrieve the latest accuracy evaluation"""
    if not latest_accuracy["has_data"]:
        return "No accuracy evaluation available yet.\n\nTo generate an accuracy report:\n1. Go to 'Ask Questions' tab\n2. Upload BOTH an error transcript AND a ground truth file\n3. Ask a question\n4. Return to this tab to see the accuracy results", ""
    return latest_accuracy["summary"], latest_accuracy["report"]


# ============================================================================
# METADATA EXTRACTION
# ============================================================================

def extract_meeting_metadata(text):
    """Extract key metadata from meeting transcripts"""
    metadata = {}
    
    date_patterns = [
        r'(?:Today is|Date:)\s*([A-Z][a-z]+\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4})',
        r'(\d{1,2}/\d{1,2}/\d{2,4})',
        r'([A-Z][a-z]+\s+\d{1,2},?\s+\d{4})'
    ]
    
    for pattern in date_patterns:
        match = re.search(pattern, text[:500])
        if match:
            metadata['date'] = match.group(1)
            break
    
    time_patterns = [
        r'(\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)?)',
        r'(\d{1,2}\s*o\'clock)',
    ]
    
    for pattern in time_patterns:
        match = re.search(pattern, text[:500])
        if match:
            metadata['time'] = match.group(1)
            break
    
    meeting_types = ['Budget Committee', 'City Council', 'Planning Commission', 'Board Meeting']
    for meeting_type in meeting_types:
        if meeting_type.lower() in text[:500].lower():
            metadata['meeting_type'] = meeting_type
            break
    
    return metadata


def load_text_file(file_path):
    """Load plain text file with metadata extraction"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        metadata = extract_meeting_metadata(content)
        
        header_parts = ["=== MEETING TRANSCRIPT ===\n"]
        if metadata.get('date'):
            header_parts.append(f"DATE: {metadata['date']}")
        if metadata.get('time'):
            header_parts.append(f"TIME: {metadata['time']}")
        if metadata.get('meeting_type'):
            header_parts.append(f"TYPE: {metadata['meeting_type']}")
        
        header_parts.append(f"SOURCE: {file_path}")
        header_parts.append("\n=== TRANSCRIPT CONTENT ===\n\n")
        
        structured_content = "\n".join(header_parts) + content
        
        return [Document(
            page_content=structured_content,
            metadata={"source": file_path, "type": "text_file", **metadata}
        )]
    except Exception as e:
        print(f"Error loading text file: {e}")
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return [Document(
            page_content=content,
            metadata={"source": file_path, "type": "text_file"}
        )]


def load_json_transcript(file_path):
    """Load JSON transcript and extract only the text content"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    text_parts = []
    
    try:
        # Try parsing as a complete JSON array first
        data = json.loads(content)
        
        if isinstance(data, list):
            # It's an array of segments
            for segment in data:
                if 'text' in segment:
                    text_parts.append(segment['text'].strip())
        else:
            # It's a single object, try to find text
            if 'text' in data:
                text_parts.append(data['text'].strip())
    
    except json.JSONDecodeError:
        # Fall back to line-by-line parsing (newline-delimited JSON)
        lines = content.split('\n')
        for line in lines:
            try:
                data = json.loads(line.strip())
                if 'text' in data:
                    text_parts.append(data['text'].strip())
            except json.JSONDecodeError:
                continue
    
    # Join all text segments with spaces
    full_text = ' '.join(text_parts)
    
    print(f"   📄 Extracted {len(text_parts)} segments")
    print(f"   📊 Total text: {len(full_text.split())} words")
    print(f"   🔍 First 100 chars: {full_text[:100]}...")
    
    return [Document(
        page_content=full_text,
        metadata={"source": file_path, "type": "json_transcript"}
    )]


# ============================================================================
# MAIN PROCESSING FUNCTION
# ============================================================================

def process_documents(files, ground_truth_file, question):
    """Process uploaded documents with LLM-only correction pipeline"""
    
    global latest_report, latest_accuracy
    
    # Use shared initialization instead of recreating
    model_local, dicts = initialize_system()
    
    # Load documents
    all_docs = []
    print(f"\n📁 Processing {len(files)} file(s)...")
    for file in files:
        file_path = file.name
        
        if file_path.endswith('.json'):
            docs = load_json_transcript(file_path)
            all_docs.extend(docs)
        elif file_path.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            all_docs.extend(docs)
        elif file_path.endswith('.txt'):
            docs = load_text_file(file_path)
            all_docs.extend(docs)
        else:
            print(f"⚠️ Skipping unsupported file type: {file_path}")
    
    if not all_docs:
        return "No valid documents uploaded. Please upload JSON, PDF, or TXT files.", "", "", ""
    
    full_text = "\n\n=== DOCUMENT SEPARATOR ===\n\n".join([doc.page_content for doc in all_docs])
    
    print(f"   📊 Full document: {len(full_text.split())} words")
    print(f"   🔍 First 200 chars: {full_text[:200]}...")
    
    print("✂️ Splitting into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ": ", " ", ""],
        keep_separator=True
    )
    doc_splits = text_splitter.split_documents(all_docs)
    
    print("🧠 Creating embeddings...")
    embeddings = OllamaEmbeddings(model='nomic-embed-text')
    
    print("💾 Creating vector store...")
    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name="rag-chroma",
        embedding=embeddings,
    )
    
    # Check if comprehensive search needed
    keyword_indicators = [
        'find all', 'list all', 'every instance', 'all occurrences', 
        'how many times', 'count', 'all mentions'
    ]
    
    is_comprehensive = any(indicator in question.lower() for indicator in keyword_indicators)
    
    if is_comprehensive:
        print("🔍 Using comprehensive search (full document)...")
        # Use full text but with size limit
        word_count = len(full_text.split())
        if word_count > 2000:
            print(f"   ⚠️ Document is large ({word_count} words)")
            print(f"   ⚠️ This may take 3-5 minutes to correct")
        context = full_text
    else:
        print("🔍 Using semantic retrieval...")
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 12, "fetch_k": 40, "lambda_mult": 0.6}
        )
        retrieved_docs = retriever.get_relevant_documents(question)
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    
    print("✅ Generating answer...")
    
    # CRITICAL FIX: Correct the CONTEXT before generating the answer
    print("\n" + "="*80)
    print("🔄 CORRECTING RETRIEVED CONTEXT (Before Answer Generation)")
    print("="*80)
    
    corrected_context, context_changes = llm_correct_with_tracking(context, dicts, model_local)
    
    print("\n✅ Now generating answer from corrected context...")
    
    template = """You are an expert document analyst. Answer based on the provided context.

CONTEXT:
{context}

QUESTION: {question}

INSTRUCTIONS:
- Answer directly and precisely
- Quote exact passages when possible
- If information is not in context, state clearly

ANSWER:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model_local | StrOutputParser()
    
    # Use CORRECTED context to generate answer
    final_answer = chain.invoke({"context": corrected_context, "question": question})
    
    # Generate report showing corrections made to the CONTEXT
    report = generate_correction_report(context, corrected_context, context_changes)
    
    # Store in global variable
    latest_report["report"] = report
    latest_report["original"] = context  # Show original context
    latest_report["corrected"] = final_answer  # Show final answer
    
    # Print to terminal
    print("\n" + report)
    
    # If ground truth file was provided, calculate accuracy
    if ground_truth_file is not None:
        print("\n" + "="*80)
        print("🎯 CALCULATING ACCURACY (Ground Truth Provided)")
        print("="*80)
        
        # Load ground truth
        if ground_truth_file.name.endswith('.json'):
            truth_docs = load_json_transcript(ground_truth_file.name)
        elif ground_truth_file.name.endswith('.txt'):
            truth_docs = load_text_file(ground_truth_file.name)
        else:
            print("⚠️ Ground truth file must be JSON or TXT - skipping accuracy evaluation")
            latest_accuracy["has_data"] = False
        
        if ground_truth_file.name.endswith('.json') or ground_truth_file.name.endswith('.txt'):
            ground_truth = truth_docs[0].page_content
            
            # Use the original full_text as the "error" version
            original_errors = full_text
            
            print(f"   📄 Error transcript: {len(original_errors.split())} words")
            print(f"   📄 Ground truth: {len(ground_truth.split())} words")
            
            # Calculate baseline (how many errors exist)
            print("   📊 Step 1: Calculating baseline...")
            baseline_metrics = calculate_accuracy_from_diffs(original_errors, original_errors, ground_truth)
            
            # Calculate system accuracy (how well we did)
            print("   📊 Step 2: Calculating system accuracy...")
            system_metrics = calculate_accuracy_from_diffs(original_errors, corrected_context, ground_truth)
            
            # Calculate improvement
            improvement = system_metrics['accuracy']
            
            # Generate reports
            accuracy_report = generate_dual_accuracy_report(baseline_metrics, system_metrics, improvement)
            
            accuracy_summary = f"""
ACCURACY EVALUATION COMPLETE

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 BASELINE (Original Transcript)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total Errors in Original: {baseline_metrics['total_errors']}
Baseline Accuracy: 0.00% (no corrections applied)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🤖 SYSTEM PERFORMANCE (After Your Corrections)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
System Accuracy:  {system_metrics['accuracy']:.2f}%
Precision:        {system_metrics['precision']:.2f}%
Recall:           {system_metrics['recall']:.2f}%

Correctly Fixed:  {system_metrics['correctly_fixed']} / {system_metrics['total_errors']}
Missed:           {system_metrics['missed']}
Wrong Fixes:      {system_metrics['wrong_fixes']}
False Positives:  {system_metrics['false_positives']}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📈 IMPROVEMENT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Improvement: +{improvement:.2f}% (from 0% to {system_metrics['accuracy']:.2f}%)

Switch to the '🎯 Accuracy Evaluation' tab to see detailed results.
"""
            
            # Store in global variable
            latest_accuracy["summary"] = accuracy_summary
            latest_accuracy["report"] = accuracy_report
            latest_accuracy["has_data"] = True
            
            print("\n Accuracy evaluation complete!")
    else:
        # No ground truth provided
        latest_accuracy["has_data"] = False
    
    # Return for display
    summary = f"\n\n{'='*80}\n📊 {len(context_changes)} corrections made to source context.\n{'='*80}"
    
    return final_answer + summary, report, context, final_answer


def get_latest_report():
    """Retrieve the latest correction report"""
    if not latest_report["report"]:
        return "No corrections have been run yet. Process a document first.", "", ""
    return latest_report["report"], latest_report["original"], latest_report["corrected"]


# ============================================================================
# GRADIO INTERFACE WITH TABS
# ============================================================================

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 📄 Meeting Transcript Analyzer with LLM Correction")
    
    with gr.Tabs():
        # TAB 1: Main Q&A Interface (NOW WITH GROUND TRUTH OPTION)
        with gr.Tab("📝 Ask Questions"):
            gr.Markdown("""
            ### Upload Transcripts and Ask Questions
            
            **Required:** Upload at least one transcript file.
            
            **Optional:** Upload a ground truth file to evaluate accuracy (see results in the "🎯 Accuracy Evaluation" tab).
            """)
            
            with gr.Row():
                with gr.Column():
                    file_input = gr.File(
                        label="📄 Transcript(s) to Process (JSON, PDF, TXT)",
                        file_count="multiple",
                        file_types=[".json", ".pdf", ".txt"]
                    )
                    ground_truth_input = gr.File(
                        label="✅ Ground Truth (Optional - for accuracy evaluation)",
                        file_count="single",
                        file_types=[".json", ".txt"]
                    )
                    question_input = gr.Textbox(
                        label="Ask a question about your documents",
                        lines=4,
                        placeholder="Examples:\n- When did this meeting start?\n- Find all mentions of 'budget'\n- What was discussed about Crescent Care?"
                    )
                    submit_btn = gr.Button("Submit", variant="primary")
                
                with gr.Column():
                    answer_output = gr.Textbox(
                        label="Answer (Corrected)",
                        lines=20,
                        show_copy_button=True
                    )
            
            # Hidden outputs for report tab
            report_store = gr.Textbox(visible=False)
            original_store = gr.Textbox(visible=False)
            corrected_store = gr.Textbox(visible=False)
            
            submit_btn.click(
                fn=process_documents,
                inputs=[file_input, ground_truth_input, question_input],
                outputs=[answer_output, report_store, original_store, corrected_store]
            )
        
        # TAB 2: Detailed Correction Report
        with gr.Tab("📊 Correction Report"):
            gr.Markdown("""
            ### Detailed Correction Report
            This tab shows:
            - **Total number of changes** made
            - **What changed** (before → after)
            - **Context** for each correction
            - **Side-by-side comparison** of original vs corrected
            
            Run a query first, then view the report here!
            """)
            
            refresh_btn = gr.Button("🔄 Refresh Report", variant="secondary")
            
            with gr.Row():
                with gr.Column():
                    report_display = gr.Textbox(
                        label="Correction Report",
                        lines=30,
                        show_copy_button=True
                    )
            
            with gr.Row():
                with gr.Column():
                    original_display = gr.Textbox(
                        label="Original Answer (Before Correction)",
                        lines=15,
                        show_copy_button=True
                    )
                with gr.Column():
                    corrected_display = gr.Textbox(
                        label="Corrected Answer (After Correction)",
                        lines=15,
                        show_copy_button=True
                    )
            
            refresh_btn.click(
                fn=get_latest_report,
                inputs=[],
                outputs=[report_display, original_display, corrected_display]
            )
            
            # Auto-update when new corrections are made
            report_store.change(
                fn=lambda x: x,
                inputs=[report_store],
                outputs=[report_display]
            )
            original_store.change(
                fn=lambda x: x,
                inputs=[original_store],
                outputs=[original_display]
            )
            corrected_store.change(
                fn=lambda x: x,
                inputs=[corrected_store],
                outputs=[corrected_display]
            )
        
        # TAB 3: Accuracy Evaluation (VIEW ONLY)
        with gr.Tab("🎯 Accuracy Evaluation"):
            gr.Markdown("""
            ### System Accuracy Report
            
            This tab shows how well your correction system performed compared to a manually corrected "ground truth" version.
            
            **To generate an accuracy report:**
            1. Go to the "📝 Ask Questions" tab
            2. Upload your transcript file(s) as normal
            3. **Also upload a ground truth file** (manually corrected version)
            4. Ask your question
            5. Return here to see detailed accuracy metrics
            
            **The report includes:**
            - Baseline accuracy (how many errors existed originally)
            - System accuracy (how many errors your system fixed)
            - Precision, recall, and overall improvement
            - List of correctly fixed errors
            - List of missed errors
            - List of false positives (unnecessary changes)
            """)
            
            refresh_accuracy_btn = gr.Button("🔄 Refresh Accuracy Report", variant="secondary")
            
            with gr.Row():
                with gr.Column():
                    accuracy_summary_display = gr.Textbox(
                        label="Accuracy Summary",
                        lines=18,
                        show_copy_button=True
                    )
            
            with gr.Row():
                with gr.Column():
                    accuracy_report_display = gr.Textbox(
                        label="📋 Detailed Accuracy Report",
                        lines=30,
                        show_copy_button=True
                    )
            
            refresh_accuracy_btn.click(
                fn=get_latest_accuracy,
                inputs=[],
                outputs=[accuracy_summary_display, accuracy_report_display]
            )
        

if __name__ == "__main__":
    demo.launch(share=False)
