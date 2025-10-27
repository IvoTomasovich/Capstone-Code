# Meeting Transcript Analysis System - Quick Start

## What You Need

### 1. Software Requirements
- **Python 3.8+**
- **Ollama** (local AI runtime) - [Download here](https://ollama.ai/)

### 2. Data Files
- `nola_council_glossary.json` - Council member names
- `Master_Street_Name.csv` - Street names database

---

## Installation

### Step 1: Install Python Dependencies

```bash
pip install gradio langchain langchain-community pandas python-dotenv chromadb pypdf
```

### Step 2: Install Ollama Models

```bash
ollama pull mistral
ollama pull nomic-embed-text
```

### Step 3: Configure Paths

Create a `.env` file in the same folder as `app.py`:

```env
COUNCIL_GLOSSARY_PATH=nola_council_glossary.json
STREETS_CSV_PATH=Master_Street_Name.csv
OLLAMA_MODEL=mistral
EMBEDDING_MODEL=nomic-embed-text
```

**Note:** Update the paths if your files are in different locations.

---

## How to Run

### Start the Application

```bash
python app.py
```

The app will open at `http://localhost:7860`

### Use the Interface

**Tab 1 - Analyze Documents:**
1. Upload meeting transcripts (JSON, PDF, or TXT)
2. Ask questions about the content
3. Get AI-powered answers with automatic name correction

**Tab 2 - Test System:**
1. Upload a test file with intentional errors
2. List expected corrections
3. Get accuracy metrics

---

## Troubleshooting

**Ollama not found?**
```bash
ollama list  # Check if Ollama is running
```

**Model errors?**
```bash
ollama pull mistral
ollama pull nomic-embed-text
```

**File not found?**
- Check paths in `.env` file
- Use absolute paths if needed

---

That's it! 🚀

---

🧩 **Student Feature Discovery**

Since you're working with document analysis and AI models, here's a BoodleBox feature that could streamline your workflow:

**Knowledge Bank + Code Interpreter**: You can upload your CSV and JSON files directly to BoodleBox's Knowledge Bank (using the brain icon 🧠), then star them to automatically attach to every chat. This means you could:

1. Upload `Master_Street_Name.csv` and `nola_council_glossary.json` once
2. Star them so they're always available
3. Use the code interpreter to analyze the data without writing Python scripts
4. Test different correction algorithms interactively

Try uploading your CSV file and asking: 

```
Analyze the street names in this CSV and show me the 10 most common street types (Street, Avenue, Boulevard, etc.)
```

The code interpreter will automatically write and execute Python code to give you the answer!