import gradio as gr
import json
import pandas as pd
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import ChatOpenAI
import re
from difflib import SequenceMatcher, get_close_matches
from datetime import datetime
import os
import shutil
from typing import Dict, List, Tuple, Optional
from collections import Counter


# ============================================================================
# GLOBAL VARIABLES
# ============================================================================

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
        print("🔄 Initializing ChatOpenAI model (for Q&A only, not correction)...")
        model_local = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            api_key="redacted",
            max_tokens=4096
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
        with open('english_words.json', 'r', encoding='utf-8') as f:
            english_words = set(word.lower() for word in json.load(f))
        print(f"✅ Loaded {len(english_words)} English words")
    except FileNotFoundError:
        print("⚠️ english_words.json not found - using minimal set")
        english_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'has', 'have', 'had',
            'was', 'were', 'been', 'is', 'are', 'weekend', 'seconded',
            'representing', 'resident', 'community', 'on', 'behalf'
        }
    
    # 2. New Orleans names
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
    
    # 3. New Orleans streets
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
# STAGE 1: RULE-BASED STREET NAME CORRECTION
# ============================================================================

def apply_correction_preserving_punctuation(words_list, position, corrected_value):
    
    if position >= len(words_list):
        return words_list
    
    original_word = words_list[position]
    
    # Check for possessive ('s or s')
    possessive = ''
    temp_word = original_word
    
    if temp_word.endswith("'s") or temp_word.endswith("'s"):  # Handle both straight and curly quotes
        possessive = temp_word[-2:]
        temp_word = temp_word[:-2]
    elif temp_word.endswith("s'"):
        possessive = temp_word[-2:]
        temp_word = temp_word[:-2]
    
    # Extract trailing punctuation (after removing possessive)
    trailing_punct = ''
    for char in reversed(temp_word):
        if char in '.,!?;:\'"()[]{}':
            trailing_punct = char + trailing_punct
        else:
            break
    
    # Apply correction with possessive AND punctuation preserved
    words_list[position] = corrected_value + possessive + trailing_punct
    
    return words_list

def normalize_street_dictionary(streets: List[str]) -> List[str]:
    """
    Extract street names without suffixes.
    
    For multi-word streets, add BOTH the full name AND individual words
    to catch both complete and partial matches.
    
    Filters out common words that should never match alone (French articles, etc.)
    """
    
    normalized = []
    seen = set()  # Avoid duplicates
    
    blacklist = {'de', 'la', 'du', 'des', 'le', 'les', 'rue', 'port', 'fort', 'st', 'saint'}
    
    street_suffixes = ['Street', 'St.', 'Avenue', 'Ave.', 'Boulevard', 'Blvd.', 
                      'Road', 'Rd.', 'Drive', 'Dr.', 'Lane', 'Ln.', 'Court', 'Ct.',
                      'Circle', 'Cir.', 'Place', 'Pl.', 'Way', 'Parkway', 'Pkwy.',
                      'Terrace', 'Ter.', 'Trail', 'Alley']
    
    for street in streets:
        street_name = street
        has_suffix = False
        
        # Check if it has a standard suffix
        for suffix in street_suffixes:
            if street.endswith(suffix):
                street_name = street[:-len(suffix)].strip()
                has_suffix = True
                break
        
        # Add the full normalized name
        if street_name not in seen:
            normalized.append(street_name)
            seen.add(street_name)
        
        # For ALL multi-word streets (with or without suffix),
        # add each individual word AND all contiguous sub-phrases
        words = street_name.split()
        if len(words) > 1:
            # Add individual words (EXCEPT blacklisted ones)
            for word in words:
                if word.lower() not in blacklist and word not in seen:
                    normalized.append(word)
                    seen.add(word)
            
            for i in range(len(words)):
                for j in range(i + 1, len(words) + 1):
                    phrase = ' '.join(words[i:j])
                    if phrase not in seen and phrase != street_name:
                        # Skip if phrase starts with blacklisted word
                        first_word = words[i].lower()
                        if first_word not in blacklist:
                            normalized.append(phrase)
                            seen.add(phrase)
    
    return normalized


def fix_street_names_fuzzy(text: str, street_dictionary: List[str], english_words: set) -> Tuple[str, List[Dict], List[Dict]]:
    """
    Fix street names using deterministic pattern matching + fuzzy matching.
    FUZZY-ONLY VERSION - No LLM
    
    Handles:
    - Multi-word street names (e.g., "South Gayoso Street")
    - Address numbers (skips numeric prefixes like "1030")
    - Non-standard street names (e.g., "Rue Renée" without suffix)
    """
    
    print("\n  STAGE 1: RULE-BASED STREET CORRECTION (FUZZY ONLY)")
    print("="*80)
    
    # Normalize streets
    normalized_streets = normalize_street_dictionary(street_dictionary)
    
    print(f"   📚 Loaded {len(normalized_streets)} street names")
    print(f"   🔍 Sample: {normalized_streets[:5]}")
    
    words = text.split()
    corrections = []
    near_misses = []
    
    # Street indicators (lowercase)
    street_indicators = ['street', 'st.', 'avenue', 'ave.', 'boulevard', 'blvd.', 
                        'road', 'rd.', 'drive', 'dr.', 'lane', 'ln.', 'court', 'ct.',
                        'circle', 'cir.', 'place', 'pl.', 'way', 'parkway', 'pkwy.',
                        'terrace', 'ter.', 'trail', 'alley']
    
    # Build lowercase lookup
    street_dict_lower = {s.lower(): s for s in normalized_streets}
    
    # Track corrections we've already made to avoid duplicates
    corrected_positions = set()
    
    # Find potential street names
    for i, word in enumerate(words):
        word_clean = word.strip('.,!?;:\'"()[]{}').lower()
        
        if word_clean in street_indicators and i > 0:
            # NEW: Try to capture multi-word street names
            # Look at 1, 2, or 3 words before the indicator
            for num_words in [3, 2, 1]:  # Try longest first
                if i >= num_words:
                    # Get the potential street name (could be multi-word)
                    potential_words = []
                    for j in range(num_words):
                        word_stripped = words[i - num_words + j].strip('.,!?;:\'"()[]{}')
                        if word_stripped:  # Skip empty strings
                            potential_words.append(word_stripped)
                    
                    if len(potential_words) == 0:  # All were punctuation
                        continue
                    
                    potential_street = ' '.join(potential_words)
                    start_pos = i - num_words
                    
                    # CHECK 1: Skip if we've already corrected this position
                    if start_pos in corrected_positions:
                        break
                    
                    # CHECK 2: Skip if starts with a number (address number)
                    if potential_words[0].replace(',', '').replace('.', '').isdigit():
                        print(f"   ✓ SKIPPED (address number): '{potential_street} {word}'")
                        continue
                    
                    # CHECK 3: Skip if ANY word is a common English word
                    skip = False
                    for pw in potential_words:
                        if pw.lower() in english_words:
                            skip = True
                            break
                    
                    if skip:
                        continue
                    
                    # CHECK 4: Already correct?
                    if potential_street.lower() in street_dict_lower:
                        # Already correct!
                        break
                    
                    # ✅ IMPROVED: Fuzzy match with PREFIX-BASED scoring
                    best_match = None
                    best_score = 0
                    
                    potential_lower = potential_street.lower()
                    
                    for correct_street_lower, correct_street_proper in street_dict_lower.items():
                        # Use SequenceMatcher for base score
                        base_score = SequenceMatcher(
                            None, 
                            potential_lower, 
                            correct_street_lower
                        ).ratio()
                        
                        # ✅ BOOST: If error is a prefix of correct (common typo pattern)
                        # "Deleron" is prefix of "Delaronde" -> boost score
                        if correct_street_lower.startswith(potential_lower):
                            # Strong prefix match - boost significantly
                            score = base_score + 0.3  # Add 30% bonus
                        elif potential_lower.startswith(correct_street_lower[:len(potential_lower)//2]):
                            # Partial prefix match - small boost
                            score = base_score + 0.1  # Add 10% bonus
                        else:
                            score = base_score
                        
                        # Cap at 1.0
                        score = min(score, 1.0)
                        
                        if score > best_score:
                            best_score = score
                            best_match = correct_street_proper
                    
                    # CHECK 5: High enough confidence?
                    # For single-word matches, require higher threshold to avoid false positives
                    threshold = 0.75 if num_words == 1 else 0.65
                    
                    # CHECK 6: Sanity check - don't correct to very short words unless high confidence
                    if best_match and len(best_match) <= 2 and best_score < 0.90:
                        print(f"   ✗ REJECTED: '{potential_street}' -> '{best_match}' (too short, {len(best_match)} chars, {best_score:.0%})")
                        continue
                    
                    if best_score >= threshold:
                        corrections.append({
                            'original': potential_street,
                            'corrected': best_match,
                            'confidence': best_score,
                            'position': start_pos,
                            'num_words': num_words,
                            'context': ' '.join(words[max(0, i-6):min(len(words), i+3)])
                        })
                        
                        # Mark this position as corrected
                        for k in range(num_words):
                            corrected_positions.add(start_pos + k)
                        
                        print(f"   ✓ WILL FIX: '{potential_street} {word}' → '{best_match} {word}' ({best_score:.0%})")
                        break  # Found a match, don't try shorter versions
                    else:
                        if best_score > 0.5:  # Only track if somewhat close
                            near_misses.append({
                                'original': potential_street,
                                'best_candidate': best_match,
                                'score': best_score,
                                'threshold': threshold,
                                'position': start_pos,
                                'context': ' '.join(words[max(0, i-6):min(len(words), i+3)])
                            })
    
    # Apply corrections
    corrected_text = text
    actual_corrections = []

    for correction in sorted(corrections, key=lambda x: -x['position']):
        words_before = corrected_text.split()
        
        # Replace multi-word street names
        num_words = correction['num_words']
        if correction['position'] + num_words <= len(words_before):
            corrected_parts = correction['corrected'].split()
            
            # Store originals before correction
            originals = []
            for j in range(num_words):
                if correction['position'] + j < len(words_before):
                    originals.append(words_before[correction['position'] + j])
            
            words_after = words_before.copy()
            
            # Apply each word with punctuation preservation on last word
            for j in range(num_words):
                if correction['position'] + j < len(words_after) and j < len(corrected_parts):
                    if j == num_words - 1:
                        # Last word - preserve punctuation
                        words_after = apply_correction_preserving_punctuation(
                            words_after,
                            correction['position'] + j,
                            corrected_parts[j]
                        )
                    else:
                        # Not last word - just replace
                        words_after[correction['position'] + j] = corrected_parts[j]
            
            # Check if anything actually changed
            changed = False
            for j in range(num_words):
                if correction['position'] + j < len(words_before):
                    if words_before[correction['position'] + j] != words_after[correction['position'] + j]:
                        changed = True
                        break
            
            if changed:
                # Record the actual change
                actual_corrections.append({
                    'original': ' '.join(originals),
                    'corrected': ' '.join([words_after[correction['position'] + j] for j in range(num_words) if correction['position'] + j < len(words_after)]),
                    'position': correction['position'],
                    'num_words': num_words,
                    'context': correction['context']
                })
                print(f"   ✓ APPLIED: '{' '.join(originals)}' → '{correction['corrected']}'")
            else:
                print(f"   ⚪ SKIPPED: '{' '.join(originals)}' already correct after punctuation preservation")
            
            corrected_text = ' '.join(words_after)

    print(f"\n   📊 Fixed {len(actual_corrections)} street name(s)")
    print("="*80)

    return corrected_text, actual_corrections, near_misses


# ============================================================================
# STAGE 2: FUZZY-BASED NAME CORRECTION (NO LLM)
# ============================================================================

def fix_names_fuzzy(text: str, dicts: Dict) -> Tuple[str, List[Dict], List[Dict]]:
    """
    Fix person names using ONLY fuzzy matching - NO LLM
    """
    
    print("\n👤 STAGE 2: RULE-BASED NAME CORRECTION (FUZZY ONLY)")
    print("="*80)
    
    words = text.split()
    corrections = []
    near_misses = []
    
    # Title words that precede names
    title_words = ['mayor', 'councilmember', 'commissioner', 'dr.', 'mr.', 'ms.', 
                   'mrs.', 'madam', 'miss', 'professor', 'senator', 'representative', 
                   'director', 'councilmembers', 'member', 'president', 'councilwoman', 
                   'officer', 'members', 'councilor', 'councilman', 'vice']
    
    all_names = dicts['names']
    english_words = dicts['english']
    
    # Build frequency map - count ALL instances (capitalized OR lowercase)
    name_frequencies = Counter()
    for word in words:
        clean_word = word.strip('.,!?;:\'"()[]{}')
        # Count it regardless of capitalization
        if clean_word and len(clean_word) > 1:
            # Store in original form for precise counting
            name_frequencies[clean_word] += 1
    
    print(f"   Found {len(name_frequencies)} unique words")
    
    # Find potential name errors
    for i, word in enumerate(words):
        if word.lower() in title_words and i + 1 < len(words):
            potential_name = words[i + 1].strip('.,!?;:\'"()[]{}')
            
            if not potential_name:  # Empty after stripping
                continue
            
            # DEBUG: Track "Morel" specifically
            if 'morel' in potential_name.lower():
                print(f"\n   🔍 DEBUG: Found potential name '{potential_name}'")
                print(f"      After title: '{word}'")
                print(f"      In English dict? {potential_name.lower() in english_words}")
                print(f"      In names dict? {potential_name.lower() in [n.lower() for n in all_names]}")
                print(f"      Frequency count: {name_frequencies.get(potential_name, 0)}")
            
            # Skip English words
            if potential_name.lower() in english_words:
                if 'morel' in potential_name.lower():
                    print(f"      → SKIPPED: English word")
                continue
            
            # Skip if already in dictionary (case-insensitive)
            if potential_name.lower() in [n.lower() for n in all_names]:
                if 'morel' in potential_name.lower():
                    print(f"      → SKIPPED: Already in names dictionary")
                continue
            
            # Fuzzy match against names dictionary
            close_matches = get_close_matches(
                potential_name, 
                all_names,
                n=1, 
                cutoff=0.70
            )
            
            if close_matches:
                match = close_matches[0]
                
                # Skip if just case difference
                if potential_name.lower() == match.lower():
                    if 'morel' in potential_name.lower():
                        print(f"      → SKIPPED: Just case difference with '{match}'")
                    continue
                
                similarity = SequenceMatcher(None, potential_name.lower(), match.lower()).ratio()
                
                # DEBUG: Show fuzzy match result for Morel
                if 'morel' in potential_name.lower():
                    print(f"      Fuzzy match found: '{match}' (similarity: {similarity:.2%})")
                
                # ✅ IMPROVED: Apply confidence boosting based on frequency (case-insensitive check)
                confidence_boost = False
                boost_reasons = []
                
                # Count all case variations
                correct_count = (name_frequencies.get(match, 0) + 
                               name_frequencies.get(match.lower(), 0) + 
                               name_frequencies.get(match.upper(), 0) +
                               name_frequencies.get(match.capitalize(), 0))
                
                incorrect_count = (name_frequencies.get(potential_name, 0) + 
                                 name_frequencies.get(potential_name.lower(), 0) +
                                 name_frequencies.get(potential_name.upper(), 0) +
                                 name_frequencies.get(potential_name.capitalize(), 0))
                
                # BOOST 1: Correct form is MORE COMMON than incorrect form
                if correct_count > incorrect_count and correct_count > 0:
                    confidence_boost = True
                    boost_reasons.append(f"Correct '{match}' ({correct_count}x) > Error '{potential_name}' ({incorrect_count}x)")
                
                # BOOST 2: Incorrect form is rare (≤2 occurrences)
                if incorrect_count <= 2:
                    confidence_boost = True
                    boost_reasons.append(f"Error '{potential_name}' rare ({incorrect_count}x)")
                
                # BOOST 3: Correct form appears multiple times (≥3)
                if correct_count >= 3:
                    confidence_boost = True
                    boost_reasons.append(f"Correct '{match}' common ({correct_count}x)")
                
                if 'morel' in potential_name.lower() and boost_reasons:
                    print(f"      Confidence boosts: {', '.join(boost_reasons)}")
                
                # Only correct if high confidence OR boosted
                if similarity > 0.78 or (similarity > 0.70 and confidence_boost):
                    corrections.append({
                        'original': potential_name,
                        'corrected': match,
                        'confidence': similarity,
                        'position': i+1,
                        'context': ' '.join(words[max(0, i-3):min(len(words), i+8)])
                    })
                    
                    print(f"   ✓ WILL FIX: '{potential_name}' → '{match}' ({similarity:.0%}, after '{word}')")
                else:
                    # Track as near miss
                    near_misses.append({
                        'original': potential_name,
                        'best_candidate': match,
                        'score': similarity,
                        'threshold': 0.78 if not confidence_boost else 0.70,
                        'boosted': confidence_boost,
                        'position': i+1,
                        'context': ' '.join(words[max(0, i-3):min(len(words), i+8)])
                    })
                    if 'morel' in potential_name.lower():
                        print(f"      → NEAR MISS: Score {similarity:.2%}, boost={confidence_boost}, threshold={'0.78' if not confidence_boost else '0.70'}")
            else:
                if 'morel' in potential_name.lower():
                    print(f"      → NO FUZZY MATCH: No close matches found above 70% threshold")
    
    corrected_text = text
    actual_corrections = []  # Track what actually changed

    for correction in sorted(corrections, key=lambda x: -x['position']):
        words_before = corrected_text.split()
        
        # Store the original word before correction
        original_with_punct = words_before[correction['position']] if correction['position'] < len(words_before) else None
        
        # Apply correction with punctuation preservation
        words_after = apply_correction_preserving_punctuation(
            words_before.copy(), 
            correction['position'], 
            correction['corrected']
        )
        
        # Check if anything actually changed
        if original_with_punct and words_before[correction['position']] != words_after[correction['position']]:
            # Actually changed - record it
            actual_corrections.append({
                'original': words_before[correction['position']],
                'corrected': words_after[correction['position']],
                'position': correction['position'],
                'context': correction['context']
            })
            print(f"   ✓ APPLIED: '{words_before[correction['position']]}' → '{words_after[correction['position']]}'")
        else:
            # No actual change (probably due to punctuation preservation)
            print(f"   ⚪ SKIPPED: '{original_with_punct}' already correct after punctuation preservation")
        
        corrected_text = ' '.join(words_after)

    print(f"\n   📊 Fixed {len(actual_corrections)} name(s)")
    print("="*80)

    return corrected_text, actual_corrections, near_misses


# ============================================================================
# TWO-STAGE FUZZY CORRECTION PIPELINE
# ============================================================================

def fuzzy_correct_with_tracking(text: str, dicts: Dict) -> Tuple[str, List[Dict], Dict[str, List[Dict]]]:
    """
    Two-stage FUZZY-ONLY correction pipeline (NO LLM for correction)
    Returns: (corrected_text, corrections, near_misses_by_type)
    """
    
    print("\n🔧 TWO-STAGE FUZZY CORRECTION PIPELINE (NO LLM)")
    print("="*80)
    
    word_count = len(text.split())
    print(f"   📊 Total context: {word_count} words")
    
    # STAGE 1: Fix streets (returns 3 values now)
    text_streets_fixed, street_corrections, street_near_misses = fix_street_names_fuzzy(
        text, 
        dicts['streets'],
        dicts['english']
    )
    
    # STAGE 2: Fix names (returns 3 values now)
    final_text, name_corrections, name_near_misses = fix_names_fuzzy(
        text_streets_fixed,
        dicts
    )
    
    # Combine corrections
    all_corrections = []
    
    for corr in street_corrections:
        all_corrections.append({
            'original': corr['original'],
            'corrected': corr['corrected'],
            'context': corr['context'],
            'position': corr['position'],
            'type': 'STREET (Fuzzy)'
        })
    
    for corr in name_corrections:
        all_corrections.append({
            'original': corr['original'],
            'corrected': corr['corrected'],
            'context': corr['context'],
            'position': corr['position'],
            'type': 'NAME (Fuzzy)'
        })
    
    all_corrections.sort(key=lambda x: x['position'])
    
    # ✅ Package near misses
    near_misses_by_type = {
        'streets': street_near_misses,
        'names': name_near_misses
    }
    
    print(f"\n✅ PIPELINE COMPLETE:")
    print(f"   Streets fixed: {len(street_corrections)}")
    print(f"   Names fixed: {len(name_corrections)}")
    print(f"   Total changes: {len(all_corrections)}")
    print("="*80)
    
    return final_text, all_corrections, near_misses_by_type  # ✅ Return 3 values


# ============================================================================
# CHANGE REPORTING
# ============================================================================

def generate_correction_report(original_text: str, corrected_text: str, 
                               changes: List[Dict]) -> str:
    """Generate detailed correction report"""
    
    report = []
    report.append("=" * 80)
    report.append("FUZZY-ONLY CORRECTION REPORT")
    report.append("=" * 80)
    report.append("")
    
    report.append("  CORRECTION METHOD: Rule-Based Fuzzy Matching (No LLM)")
    report.append("")
    
    # Summary Statistics
    report.append(" SUMMARY STATISTICS")
    report.append("-" * 80)
    report.append(f"Original text length:    {len(original_text)} characters")
    report.append(f"Corrected text length:   {len(corrected_text)} characters")
    report.append(f"Total words processed:   {len(original_text.split())} words")
    report.append(f"")
    
    # Break down by type
    street_changes = [c for c in changes if c.get('type', '').startswith('STREET')]
    name_changes = [c for c in changes if c.get('type', '').startswith('NAME')]
    
    report.append(f"STAGE 1 (Streets):       {len(street_changes)} corrections")
    report.append(f"STAGE 2 (Names):         {len(name_changes)} corrections")
    report.append(f"TOTAL CHANGES MADE:      {len(changes)}")
    report.append("")
    
    # Detailed Changes
    if changes:
        report.append(" DETAILED CORRECTIONS")
        report.append("-" * 80)
        
        for i, change in enumerate(changes, 1):
            report.append(f"\n{i}. [{change.get('type', 'UNKNOWN')}]")
            report.append(f"   Original:  '{change['original']}'")
            report.append(f"   Corrected: '{change['corrected']}'")
            report.append(f"   Context:   ...{change['context']}...")
        
        report.append("")
    else:
        report.append(" NO CORRECTIONS NEEDED")
        report.append("-" * 80)
        report.append("The text was already correctly transcribed!")
        report.append("")
    
    report.append("=" * 80)
    
    return "\n".join(report)


# ============================================================================
# ACCURACY EVALUATION
# ============================================================================

def compare_texts_with_diff(original: str, corrected: str) -> List[Dict]:
    """Compare two texts using diff algorithm"""
    
    original_words = original.split()
    corrected_words = corrected.split()
    
    matcher = SequenceMatcher(None, original_words, corrected_words)
    changes = []
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'replace':
            orig_text = ' '.join(original_words[i1:i2])
            corr_text = ' '.join(corrected_words[j1:j2])
            
            orig_clean = orig_text.strip('.,!?;:\'"()[]{}').lower()
            corr_clean = corr_text.strip('.,!?;:\'"()[]{}').lower()
            
            if orig_clean == corr_clean:
                continue
            
            word_count = len(orig_clean.split())
            if word_count > 2:
                similarity = SequenceMatcher(None, orig_clean, corr_clean).ratio()
                if similarity < 0.3:
                    continue
            
            context_start = max(0, j1 - 5)
            context_end = min(len(corrected_words), j2 + 5)
            context = ' '.join(corrected_words[context_start:context_end])
            
            changes.append({
                'original': orig_text,
                'corrected': corr_text,
                'context': context,
                'position': i1
            })
        
        elif tag == 'delete':
            orig_text = ' '.join(original_words[i1:i2])
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
            corr_text = ' '.join(corrected_words[j1:j2])
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


def calculate_accuracy_from_diffs(original: str, system_corrected: str, ground_truth: str) -> Dict:
    """Calculate accuracy metrics by comparing system output against ground truth."""
    
    # Step 1: Build the "answer key" — what SHOULD change between original and ground truth.
    # Each entry is a word-level diff: {original, corrected, position, context}
    true_corrections = compare_texts_with_diff(original, ground_truth)
    
    # Step 2: Build the "system's work" — what DID change between original and system output.
    system_corrections = compare_texts_with_diff(original, system_corrected)
    
    # Three buckets for classification:
    true_positives = []    # System fixed an error correctly (matches ground truth)
    false_negatives = []   # System missed an error or fixed it wrong
    false_positives = []   # System changed something that didn't need changing
    
    # Index system corrections by word position for fast lookup.
    # This lets us quickly check "did the system do anything at position N?"
    system_by_pos = {change['position']: change for change in system_corrections}
    
    # Step 3: For each real error (from the answer key), check what the system did.
    for true_change in true_corrections:
        pos = true_change['position']
        
        if pos in system_by_pos:
            # The system DID change something at this position
            system_change = system_by_pos[pos]
            
            # Check: did it change it to the RIGHT value?
            # (case-insensitive comparison to avoid penalizing capitalization differences)
            if system_change['corrected'].lower().strip() == true_change['corrected'].lower().strip():
                # YES — system found the error AND fixed it correctly
                true_positives.append({
                    'position': pos,
                    'original': true_change['original'],
                    'ground_truth': true_change['corrected'],
                    'system': system_change['corrected'],
                    'context': system_change['context'],
                    'status': 'CORRECT'
                })
            else:
                # NO — system found the error but corrected it to the WRONG value
                # e.g., ground truth says "Morrell" but system said "Morano"
                false_negatives.append({
                    'position': pos,
                    'original': true_change['original'],
                    'should_be': true_change['corrected'],
                    'system_said': system_change['corrected'],
                    'context': system_change['context'],
                    'status': 'WRONG_FIX'
                })
        else:
            # The system did NOT touch this position at all — error was missed entirely.
            # system_said = original because the system left it unchanged.
            false_negatives.append({
                'position': pos,
                'original': true_change['original'],
                'should_be': true_change['corrected'],
                'system_said': true_change['original'],
                'context': true_change['context'],
                'status': 'MISSED'
            })
    
    # Step 4: Find false positives — changes the system made that WEREN'T needed.
    # Build a set of positions where real errors exist (from ground truth).
    true_positions = {change['position'] for change in true_corrections}
    
    # Any system change at a position NOT in the answer key is a false positive:
    # the system "corrected" something that was already right.
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
    
    # Step 5: Calculate metrics
    total_errors = len(true_corrections)       # Total real errors in the original
    correctly_fixed = len(true_positives)      # How many the system got right
    
    # Accuracy (= recall here): of all real errors, what % did we fix correctly?
    accuracy = (correctly_fixed / total_errors * 100) if total_errors > 0 else 0
    
    # Precision: of all changes the system made, what % were actually needed?
    # High precision = few unnecessary changes. Low precision = lots of false alarms.
    precision = (correctly_fixed / (correctly_fixed + len(false_positives))) * 100 if (correctly_fixed + len(false_positives)) > 0 else 0
    
    # Recall: same as accuracy — of all real errors, what % did we catch?
    recall = (correctly_fixed / total_errors * 100) if total_errors > 0 else 0
    
    # Step 6: Return everything — both summary metrics and detailed lists
    # for the report generator to format.
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'total_errors': total_errors,
        'correctly_fixed': correctly_fixed,
        'missed': len([fn for fn in false_negatives if fn['status'] == 'MISSED']),
        'wrong_fixes': len([fn for fn in false_negatives if fn['status'] == 'WRONG_FIX']),
        'false_positives': len(false_positives),
        'true_positives': true_positives,              # Detailed list of correct fixes
        'false_negatives': false_negatives,             # Detailed list of misses + wrong fixes
        'false_positives_list': false_positives         # Detailed list of unnecessary changes
    }

def generate_dual_accuracy_report(baseline_metrics: Dict, system_metrics: Dict, improvement: float, near_misses: Dict[str, List[Dict]] = None) -> str:
    """Generate detailed accuracy report with near-miss explanations"""
    
    report = []
    report.append("=" * 80)
    report.append("FUZZY-ONLY ACCURACY EVALUATION REPORT")
    report.append("=" * 80)
    report.append("")
    
    # Baseline Section
    report.append("BASELINE (Original Transcript - No Corrections)")
    report.append("-" * 80)
    report.append(f"Total Errors Detected:  {baseline_metrics['total_errors']}")
    report.append(f"Baseline Accuracy:      0.00% (no corrections applied)")
    report.append("")
    
    # System Performance Section
    report.append("SYSTEM PERFORMANCE (Fuzzy-Only Corrections)")
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
        report.append(f"Assessment:        EXCELLENT")
    elif improvement > 60:
        report.append(f"Assessment:        GOOD")
    elif improvement > 40:
        report.append(f"Assessment:        FAIR")
    else:
        report.append(f"Assessment:        NEEDS IMPROVEMENT")
    
    report.append("")
    
    # CORRECTLY FIXED ERRORS
    if system_metrics['true_positives']:
        report.append("CORRECTLY FIXED ERRORS")
        report.append("-" * 80)
        report.append("These errors were successfully identified and corrected:")
        report.append("")
        
        for i, tp in enumerate(system_metrics['true_positives'], 1):
            report.append(f"{i}. ERROR FIXED")
            
            if tp.get('context'):
                ctx = tp['context']
                if len(ctx) > 150:
                    ctx = ctx[:70] + " ... " + ctx[-70:]
                report.append(f"   Context:      \"...{ctx}...\"")
            
            report.append(f"   Error:        '{tp['original']}'")
            report.append(f"   Corrected to: '{tp['ground_truth']}'")
            report.append(f"   System fixed: '{tp['system']}' ✓")
            report.append("")
    
    # MISSED ERRORS (with near-miss explanations)
    if system_metrics['false_negatives']:
        report.append("MISSED ERRORS")
        report.append("-" * 80)
        report.append("These errors were not corrected by the system:")
        report.append("")
        
        for i, fn in enumerate(system_metrics['false_negatives'], 1):
            report.append(f"{i}. ERROR MISSED")
            
            if fn.get('context'):
                ctx = fn['context']
                if len(ctx) > 150:
                    ctx = ctx[:70] + " ... " + ctx[-70:]
                report.append(f"   Context:      \"...{ctx}...\"")
            
            report.append(f"   Error:        '{fn['original']}'")
            report.append(f"   Should be:    '{fn['should_be']}'")
            report.append(f"   System said:  '{fn['system_said']}'")
            
            # ✅ NEW: Add near-miss explanation
            if near_misses:
                # Look for this error in near misses
                original_clean = fn['original'].strip('.,!?;:\'"()[]{}').lower()
                
                found_near_miss = None
                for miss_type in ['streets', 'names']:
                    for miss in near_misses.get(miss_type, []):
                        miss_clean = miss['original'].strip('.,!?;:\'"()[]{}').lower()
                        if original_clean == miss_clean or original_clean in miss_clean:
                            found_near_miss = miss
                            break
                    if found_near_miss:
                        break
                
                if found_near_miss:
                    score = found_near_miss['score']
                    threshold = found_near_miss['threshold']
                    candidate = found_near_miss['best_candidate']
                    
                    report.append(f"   Why missed:   Fuzzy match found '{candidate}' ({score:.1%} similarity)")
                    report.append(f"                 but score was below threshold ({score:.1%} < {threshold:.0%})")
                elif fn['status'] == 'MISSED':
                    if fn['original'] == fn['system_said']:
                        report.append(f"   Why missed:   No similar match found in dictionary")
                    else:
                        report.append(f"   Why missed:   System attempted a different fix")
                elif fn['status'] == 'WRONG_FIX':
                    report.append(f"   Why missed:   System corrected to wrong value")
            else:
                # Fallback if no near misses provided
                if fn['status'] == 'MISSED':
                    if fn['original'] == fn['system_said']:
                        report.append(f"   Why missed:   System did not detect/change this error")
                    else:
                        report.append(f"   Why missed:   System attempted a different fix")
                elif fn['status'] == 'WRONG_FIX':
                    report.append(f"   Why missed:   System corrected to wrong value")
            
            report.append("")
    
    # ⚠️ FALSE POSITIVES
    if system_metrics['false_positives_list']:
        # Filter out confusing entries
        filtered_fps = []
        skipped_large_edits = 0
        skipped_inserted_deleted = 0
        
        for fp in system_metrics['false_positives_list']:
            if fp['original'] == '[INSERTED]' or fp['system_changed_to'] == '[DELETED]':
                skipped_inserted_deleted += 1
                continue
            
            if len(fp['system_changed_to']) > 200 or len(fp['original']) > 200:
                skipped_large_edits += 1
                continue
            
            filtered_fps.append(fp)
        
        if filtered_fps or skipped_large_edits > 0:
            report.append("FALSE POSITIVES (Unnecessary Changes)")
            report.append("-" * 80)
            report.append("These changes were made to text that was already correct:")
            report.append("")
        
        for i, fp in enumerate(filtered_fps, 1):
            report.append(f"{i}. UNNECESSARY CHANGE")
            
            if fp.get('context'):
                ctx = fp['context']
                if len(ctx) > 150:
                    ctx = ctx[:70] + " ... " + ctx[-70:]
                report.append(f"   Context:      \"...{ctx}...\"")
            
            report.append(f"   Original:     '{fp['original']}' (was correct)")
            report.append(f"   Changed to:   '{fp['system_changed_to']}'")
            
            orig_lower = fp['original'].lower().strip()
            changed_lower = fp['system_changed_to'].lower().strip()
            
            if orig_lower == changed_lower:
                report.append(f"   Type:         Capitalization/punctuation fix")
            elif len(fp['original'].split()) == 1 and len(fp['system_changed_to'].split()) == 1:
                report.append(f"   Type:         Single word substitution")
            elif abs(len(fp['original'].split()) - len(fp['system_changed_to'].split())) <= 2:
                report.append(f"   Type:         Minor text modification")
            else:
                report.append(f"   Type:         Text rewrite/restructuring")
            
            report.append("")
        
        if skipped_large_edits > 0:
            report.append(f"NOTE: {skipped_large_edits} large text edits were detected but not shown.")
            report.append("")
        
        if skipped_inserted_deleted > 0:
            report.append(f"NOTE: {skipped_inserted_deleted} [INSERTED]/[DELETED] entries were filtered out.")
            report.append("")
    
    report.append("=" * 80)
    report.append("")
    report.append("EXPLANATION:")
    report.append("- Position numbers reference word positions in the document")
    report.append("- Context shows surrounding text to help locate each error")
    report.append("- False positives may include legitimate grammar/style improvements")
    report.append("  that weren't present in the ground truth")
    report.append("")
    
    return "\n".join(report)


def get_latest_accuracy():
    """Retrieve the latest accuracy evaluation"""
    if not latest_accuracy["has_data"]:
        return "No accuracy evaluation available yet.", ""
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
    
    return metadata


def load_text_file(file_path):
    """Load plain text file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        metadata = extract_meeting_metadata(content)
        
        return [Document(
            page_content=content,
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
    """Load JSON transcript"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    text_parts = []
    
    try:
        data = json.loads(content)
        
        if isinstance(data, list):
            for segment in data:
                if 'text' in segment:
                    text_parts.append(segment['text'].strip())
        else:
            if 'text' in data:
                text_parts.append(data['text'].strip())
    
    except json.JSONDecodeError:
        lines = content.split('\n')
        for line in lines:
            try:
                data = json.loads(line.strip())
                if 'text' in data:
                    text_parts.append(data['text'].strip())
            except json.JSONDecodeError:
                continue
    
    full_text = ' '.join(text_parts)
    
    print(f"    Extracted {len(text_parts)} segments")
    print(f"    Total text: {len(full_text.split())} words")
    
    return [Document(
        page_content=full_text,
        metadata={"source": file_path, "type": "json_transcript"}
    )]


# ============================================================================
# MAIN PROCESSING FUNCTION
# ============================================================================

def process_documents(files, ground_truth_file, question):
    """Process documents with FUZZY-ONLY correction"""
    
    global latest_report, latest_accuracy
    
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
        return "No valid documents uploaded.", "", "", ""
    
    full_text = "\n\n=== DOCUMENT SEPARATOR ===\n\n".join([doc.page_content for doc in all_docs])
    
    print(f"   📊 Full document: {len(full_text.split())} words")
    
    # Check if accuracy testing mode
    if ground_truth_file is not None:
        print("\n" + "="*80)
        print("🎯 ACCURACY TESTING MODE DETECTED")
        print("="*80)
        print("   ⚙️ Processing FULL DOCUMENT with FUZZY-ONLY correction")
        
        context = full_text
        
        print("   ✅ Full document loaded")
        print("="*80)
        
    else:
        # Normal Q&A mode - use retrieval
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
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key="redacted"
        )
        
        print("💾 Creating vector store...")
        vectorstore = Chroma.from_documents(
            documents=doc_splits,
            collection_name="rag-chroma",
            embedding=embeddings,
        )
        
        keyword_indicators = [
            'find all', 'list all', 'every instance', 'all occurrences', 
            'how many times', 'count', 'all mentions'
        ]
        
        is_comprehensive = any(indicator in question.lower() for indicator in keyword_indicators)
        
        if is_comprehensive:
            print("🔍 Using comprehensive search (full document)...")
            context = full_text
        else:
            print("🔍 Using semantic retrieval...")
            retriever = vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 12, "fetch_k": 40, "lambda_mult": 0.6}
            )
            retrieved_docs = retriever.get_relevant_documents(question)
            context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    
    print(" Generating answer...")
    
    # FUZZY-ONLY CORRECTION
    print("\n" + "="*80)
    print(" CORRECTING CONTEXT (Fuzzy-Only Pipeline)")
    print("="*80)
    
    # UPDATED: Capture near_misses (3rd return value)
    corrected_context, context_changes, near_misses = fuzzy_correct_with_tracking(context, dicts)
    
    print("\n Now generating answer from corrected context...")
    
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
    
    final_answer = chain.invoke({"context": corrected_context, "question": question})
    
    # Generate report
    report = generate_correction_report(context, corrected_context, context_changes)
    
    latest_report["report"] = report
    latest_report["original"] = context
    latest_report["corrected"] = final_answer
    latest_report["corrected_context"] = corrected_context
    
    print("\n" + report)
    
    # Accuracy evaluation
    if ground_truth_file is not None:
        print("\n" + "="*80)
        print("🎯 CALCULATING ACCURACY")
        print("="*80)
        
        if ground_truth_file.name.endswith('.json'):
            truth_docs = load_json_transcript(ground_truth_file.name)
        elif ground_truth_file.name.endswith('.txt'):
            truth_docs = load_text_file(ground_truth_file.name)
        else:
            print("⚠️ Ground truth must be JSON or TXT")
            latest_accuracy["has_data"] = False
        
        if ground_truth_file.name.endswith('.json') or ground_truth_file.name.endswith('.txt'):
            ground_truth = truth_docs[0].page_content
            
            original_errors = full_text
            
            print(f"   📄 Error transcript: {len(original_errors.split())} words")
            print(f"   📄 Corrected transcript: {len(corrected_context.split())} words")
            print(f"   📄 Ground truth: {len(ground_truth.split())} words")
            
            print("   📊 Step 1: Calculating baseline...")
            baseline_metrics = calculate_accuracy_from_diffs(original_errors, original_errors, ground_truth)
            
            print(f"   ✅ Found {baseline_metrics['total_errors']} errors in original")
            
            print("   📊 Step 2: Calculating system accuracy...")
            system_metrics = calculate_accuracy_from_diffs(original_errors, corrected_context, ground_truth)
            
            print(f"   ✅ System fixed {system_metrics['correctly_fixed']} out of {system_metrics['total_errors']} errors")
            
            improvement = system_metrics['accuracy']
            
            # UPDATED: Pass near_misses to report generator
            accuracy_report = generate_dual_accuracy_report(
                baseline_metrics, 
                system_metrics, 
                improvement,
                near_misses  # NEW: Pass near misses
            )
            
            accuracy_summary = f"""
ACCURACY EVALUATION COMPLETE

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BASELINE (Original Transcript)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total Errors in Original: {baseline_metrics['total_errors']}
Baseline Accuracy: 0.00%

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SYSTEM PERFORMANCE (Fuzzy-Only Corrections)
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
Improvement: +{improvement:.2f}%

Processing Mode: Fuzzy-Only (No LLM)
                 Full Document ({len(corrected_context.split())} words)

Switch to the '🎯 Accuracy Evaluation' tab for details.
"""
            
            latest_accuracy["summary"] = accuracy_summary
            latest_accuracy["report"] = accuracy_report
            latest_accuracy["has_data"] = True
            
            print("\n✅ Accuracy evaluation complete!")
    else:
        latest_accuracy["has_data"] = False
    
    summary = f"\n\n{'='*80}\n📊 {len(context_changes)} corrections made (Fuzzy-Only).\n{'='*80}"
    
    return final_answer + summary, report, context, final_answer


def get_latest_report():
    """Retrieve the latest correction report"""
    if not latest_report["report"]:
        return "No corrections run yet.", "", ""
    return latest_report["report"], latest_report["original"], latest_report["corrected"]


# ============================================================================
# GRADIO INTERFACE
# ============================================================================

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Transcript Analyzer - FUZZY-ONLY VERSION")
    gr.Markdown("**Correction Method:** Rule-Based Fuzzy Matching (No LLM)")
    
    with gr.Tabs():
        with gr.Tab("Ask Questions"):
            gr.Markdown("""
            ### Upload Transcripts and Ask Questions
            
            **Correction Method:** Pure fuzzy matching (deterministic, no LLM)
            
            **Optional:** Upload ground truth for accuracy evaluation
            """)
            
            with gr.Row():
                with gr.Column():
                    file_input = gr.File(
                        label="Transcript(s) (JSON, PDF, TXT)",
                        file_count="multiple",
                        file_types=[".json", ".pdf", ".txt"]
                    )
                    ground_truth_input = gr.File(
                        label="Ground Truth (Optional)",
                        file_count="single",
                        file_types=[".json", ".txt"]
                    )
                    question_input = gr.Textbox(
                        label="Ask a question",
                        lines=4,
                        placeholder="What was discussed in the meeting?"
                    )
                    submit_btn = gr.Button("Submit", variant="primary")
                
                with gr.Column():
                    answer_output = gr.Textbox(
                        label="Answer (Corrected with Fuzzy-Only)",
                        lines=20,
                        show_copy_button=True
                    )
            
            report_store = gr.Textbox(visible=False)
            original_store = gr.Textbox(visible=False)
            corrected_store = gr.Textbox(visible=False)
            
            submit_btn.click(
                fn=process_documents,
                inputs=[file_input, ground_truth_input, question_input],
                outputs=[answer_output, report_store, original_store, corrected_store]
            )
        
        with gr.Tab("Correction Report"):
            gr.Markdown("""
            ### Fuzzy-Only Correction Report
            Shows what corrections were made using rule-based fuzzy matching
            """)
            
            refresh_btn = gr.Button("Refresh Report", variant="secondary")
            
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
                        label="Original",
                        lines=15,
                        show_copy_button=True
                    )
                with gr.Column():
                    corrected_display = gr.Textbox(
                        label="Corrected",
                        lines=15,
                        show_copy_button=True
                    )
            
            refresh_btn.click(
                fn=get_latest_report,
                inputs=[],
                outputs=[report_display, original_display, corrected_display]
            )
            
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
        
        with gr.Tab("Accuracy Evaluation"):
            gr.Markdown("""
            ### Fuzzy-Only Accuracy Report
            Shows performance using rule-based fuzzy matching only
            """)
            
            refresh_accuracy_btn = gr.Button("Refresh", variant="secondary")
            
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
                        label="Detailed Report",
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
