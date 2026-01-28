# Meeting Transcript Correction System

An AI-powered system for automatically correcting transcription errors in New Orleans City Council meeting transcripts using LLM-based correction and fuzzy string matching.

---

## 🎯 What It Does

This system corrects common transcription errors in meeting transcripts, specifically:
- **Name errors**: "Morino" → "Moreno", "Cantrel" → "Cantrell"
- **Street name errors**: "Tchopitoulous" → "Tchoupitoulas", "Burbon" → "Bourbon"
- **Title errors**: "Councilmember", "Commissioner", "Mayor"

**Current Accuracy**: ~85% on 96-error test transcript

---

## 📋 Requirements

### Software
- **Python 3.8+**
- **Ollama** (local AI runtime) - [Download here](https://ollama.ai/)

### Python Packages
```bash
pip install langchain langchain-community python-dotenv
```

### AI Models
```bash
ollama pull llama3.1
```

### Data Files
- `nola_names.json` - First and last names of council members and officials
- `nola_streets.json` - New Orleans street names database

---

## 🚀 Quick Start

### 1. Set Up Your Data Files

**Create `nola_names.json`:**
```json
{
  "first_names": ["Helena", "LaToya", "JP", ...],
  "last_names": ["Moreno", "Cantrell", "Morrell", ...]
}
```

**Create `nola_streets.json`:**
```json
[
  "Tchoupitoulas Street",
  "Magazine Street",
  "Napoleon Avenue",
  ...
]
```

### 2. Run the Correction System

```bash
python transcript_analyzer_fixed.py your_transcript.json
```

The system will:
1. Load reference names and streets
2. Apply fuzzy matching to find potential errors
3. Use LLM (llama3.1) to correct errors in context
4. Generate a detailed correction report
5. Save corrected transcript to `/mnt/user-data/outputs/`

---

## 📊 How It Works

### Two-Pass Correction System

**Pass 1: Algorithmic Correction (Optional)**
- Uses fuzzy string matching (>85% similarity)
- Catches obvious misspellings after titles
- Fast, deterministic corrections

**Pass 2: LLM Correction (Primary)**
- Context-aware correction using llama3.1
- Understands natural language context
- Follows explicit correction rules:
  - English spelling preferred
  - Check for double consonants
  - Validate French street names
  - >85% similarity = likely error

### Key Features

1. **Fuzzy Matching Hints**: Pre-identifies high-confidence errors (>78% similar)
2. **Context Preservation**: LLM maintains original meaning and structure
3. **Change Tracking**: Detailed report shows every correction made
4. **Language Rules**: Prevents over-correction of foreign spellings

---

## 📁 File Structure

```
project/
├── transcript_analyzer_fixed.py    # Main correction system
├── nola_names.json                 # Names reference data
├── nola_streets.json               # Streets reference data
├── test_transcript.json            # Your input transcript
└── /mnt/user-data/outputs/
    ├── corrected_transcript.json   # Corrected output
    └── correction_report.txt       # Detailed change log
```

---

## 🧪 Testing

### Create a Test Transcript

Your transcript should be in Whisper JSON format:
```json
[
  {
    "id": 0,
    "text": " Mayor Cantrel addressed the council.",
    "start": 0.0,
    "end": 3.5
  },
  ...
]
```

### Run Correction

```bash
python transcript_analyzer_fixed.py test_transcript.json
```

### Review Report

The system generates a detailed report showing:
- Total corrections made
- Each correction with context
- Before/after text comparison
- Character count changes

---

## 🎛️ Configuration

### Adjust Similarity Threshold

In `transcript_analyzer_fixed.py`, find the `find_potential_name_errors()` function:

```python
# High confidence threshold (default: 78%)
if similarity > 0.78:
    high_confidence.append(error_str)
```

Lower = catches more errors (but more false positives)
Higher = more conservative (fewer corrections)

### Change LLM Model

```python
llm = ChatOllama(
    model="llama3.1",  # Change to "mistral" or other model
    temperature=0.0
)
```

---

## 📈 Performance Tips

### For Better Accuracy:
1. **Expand reference lists**: Add more names/streets to JSON files
2. **Provide context**: Longer transcripts give LLM more context
3. **Clean data**: Remove duplicate entries from reference lists
4. **Test iteratively**: Run on small samples first

### For Faster Processing:
1. **Use smaller models**: Switch from llama3.1 to mistral
2. **Reduce transcript size**: Process in chunks of <1000 words
3. **Disable Pass 1**: Comment out algorithmic correction if not needed

---

## 🐛 Troubleshooting

**"Ollama not found"**
```bash
ollama list  # Verify Ollama is running
ollama pull llama3.1  # Re-download model
```

**"FileNotFoundError: nola_names.json"**
- Ensure JSON files are in the same directory as script
- Check file names match exactly (case-sensitive)
- Use absolute paths if needed

**"Low accuracy on corrections"**
- Review reference data for contamination (errors in the reference lists)
- Check that correct spellings are in the reference data
- Increase similarity threshold for more conservative corrections

**"LLM making wrong corrections"**
- Add explicit examples to the prompt (see CRITICAL CORRECTION RULES)
- Reduce temperature to 0.0 for deterministic output
- Check that reference data has proper capitalization

---

## 🔧 Advanced: Customizing Correction Rules

In `transcript_analyzer_fixed.py`, modify the LLM prompt to add your own rules:

```python
CRITICAL CORRECTION RULES:
1. ENGLISH SPELLING: This is an English transcript...
2. DOUBLE CONSONANTS: Check reference list...
3. FRENCH NAMES: New Orleans has French street names...
4. YOUR CUSTOM RULE HERE
```

---

## 📚 Example Output

**Input:**
```
Mayor Cantrel addressed Councilmember Morino about Tchopitoulous Street.
```

**Output:**
```
Mayor Cantrell addressed Councilmember Moreno about Tchoupitoulas Street.
```

**Report:**
```
Change 1: Cantrel → Cantrell (93.3% similar)
Change 2: Morino → Moreno (83.3% similar)
Change 3: Tchopitoulous → Tchoupitoulas (90% similar)
```

---

## 🎓 Academic Use

This system was developed as part of a capstone project studying automated transcript correction for municipal meetings. Current accuracy: **~85%** on a 96-error test corpus.

**Key Findings:**
- LLM-only approach outperforms hybrid algorithmic+LLM systems
- Dictionary quality is critical (contaminated reference data causes unfixable errors)
- Context-awareness helps with ambiguous corrections
- Explicit language rules prevent over-correction

---

## 📝 Citation

If you use this system in your research, please cite:

```
Tomasovich, I. (2025). AI-Powered Transcript Correction for Municipal Meetings. 
New Orleans City Council Transcript Analysis System. [Software].
```

---

## 🤝 Contributing

This is an academic project, but suggestions welcome! Key areas for improvement:
- Expanding reference dictionaries
- Adding more language-specific rules
- Supporting multiple languages
- Real-time correction streaming

---

## 📄 License

Educational/Academic use only. 

---
