import gradio as gr
import json
import pandas as pd
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import re
from datetime import datetime
from difflib import SequenceMatcher

def extract_meeting_metadata(text):
    """Extract key metadata from meeting transcripts"""
    metadata = {}
    
    # Extract date patterns
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
    
    # Extract time patterns
    time_patterns = [
        r'(\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)?)',
        r'(\d{1,2}\s*o\'clock)',
    ]
    
    for pattern in time_patterns:
        match = re.search(pattern, text[:500])
        if match:
            metadata['time'] = match.group(1)
            break
    
    # Extract meeting type
    meeting_types = ['Budget Committee', 'City Council', 'Planning Commission', 'Board Meeting']
    for meeting_type in meeting_types:
        if meeting_type.lower() in text[:500].lower():
            metadata['meeting_type'] = meeting_type
            break
    
    # Extract attendees from roll call
    roll_call_match = re.search(r'Roll call.*?(?=\n\n|\. [A-Z])', text[:1000], re.DOTALL | re.IGNORECASE)
    if roll_call_match:
        attendees = re.findall(r'Councilmember\s+([A-Z][a-z]+)', roll_call_match.group())
        if attendees:
            metadata['attendees'] = ', '.join(attendees)
    
    return metadata

def load_text_file(file_path):
    """Load plain text file with enhanced metadata extraction"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract metadata from content
        metadata = extract_meeting_metadata(content)
        
        # Create a structured header
        header_parts = ["=== MEETING TRANSCRIPT ===\n"]
        if metadata.get('date'):
            header_parts.append(f"DATE: {metadata['date']}")
        if metadata.get('time'):
            header_parts.append(f"TIME: {metadata['time']}")
        if metadata.get('meeting_type'):
            header_parts.append(f"TYPE: {metadata['meeting_type']}")
        if metadata.get('attendees'):
            header_parts.append(f"ATTENDEES: {metadata['attendees']}")
        
        header_parts.append(f"SOURCE: {file_path}")
        header_parts.append(f"CHARACTER COUNT: {len(content)}")
        header_parts.append("\n=== TRANSCRIPT CONTENT ===\n\n")
        
        structured_content = "\n".join(header_parts) + content
        
        return [Document(
            page_content=structured_content,
            metadata={
                "source": file_path,
                "type": "text_file",
                **metadata
            }
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
    """Load JSON transcript with enhanced structure preservation"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    try:
        data = json.loads(content)
        if isinstance(data, dict) and 'segments' in data:
            text_parts = []
            for segment in data.get('segments', []):
                timestamp = segment.get('start', 'N/A')
                text = segment.get('text', '').strip()
                speaker = segment.get('speaker', 'Unknown')
                confidence = segment.get('confidence', segment.get('no_speech_prob', 'N/A'))
                
                text_parts.append(
                    f"[Time: {timestamp}s | Speaker: {speaker} | Confidence: {confidence}]\n{text}"
                )
            text = "\n\n".join(text_parts)
            
            metadata_summary = f"""Document Metadata:
- Total segments: {len(data.get('segments', []))}
- Source: {file_path}
- Type: Transcript with timestamps

Content:
"""
            text = metadata_summary + text
        else:
            text = json.dumps(data, indent=2)
    except json.JSONDecodeError:
        lines = content.strip().split('\n')
        json_objects = []
        for line in lines:
            if line.strip():
                try:
                    obj = json.loads(line)
                    json_objects.append(obj)
                except json.JSONDecodeError:
                    continue
        
        if json_objects:
            text = "\n\n".join([json.dumps(obj, indent=2) for obj in json_objects])
        else:
            text = content
    
    return [Document(page_content=text, metadata={"source": file_path, "type": "json_transcript"})]

def load_glossary(
    council_glossary_path='put_path_to_council_glossary.json_here',
    streets_csv_path='put_path_to_streets_csv_here.csv'
):
    """Load combined glossary with council members AND street names"""
    all_terms = []
    
    # Load council member names
    try:
        with open(council_glossary_path, 'r', encoding='utf-8') as f:
            council_data = json.load(f)
        council_names = council_data.get('all_unique_names', [])
        all_terms.extend(council_names)
        print(f"✅ Loaded {len(council_names)} council member names")
    except FileNotFoundError:
        print(f"⚠️ Council glossary not found at {council_glossary_path}")
        council_names = []
    
    # Load street names from CSV
    try:
        df = pd.read_csv(streets_csv_path)
        street_names = df['FULLNAME'].dropna().unique().tolist()
        
        # Add common variations for problematic streets
        enhanced_streets = []
        for street in street_names:
            enhanced_streets.append(street)
            
            # Special handling for Tchoupitoulas
            if 'Tchoupitoulas' in street:
                enhanced_streets.extend([
                    'Tchoupitoulas', 'Tchoup', 'Chop a Tulas', 'Chopitoulas',
                    'Chop a toulas', 'Tchopitoulas', 'Chop a Tulane'
                ])
        
        all_terms.extend(enhanced_streets)
        print(f"✅ Loaded {len(street_names)} street names ({len(enhanced_streets)} with variations)")
    except FileNotFoundError:
        print(f"⚠️ Streets CSV not found at {streets_csv_path}")
    except Exception as e:
        print(f"⚠️ Error loading streets CSV: {e}")
    
    # Remove duplicates
    unique_terms = list(set(all_terms))
    print(f"📚 Total unique terms in glossary: {len(unique_terms)}")
    
    return unique_terms

def similarity_score(a, b):
    """Calculate similarity between two strings (0-1 scale)"""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def correct_names_in_text(text, glossary, threshold=85):  # BALANCED threshold
    """
    Correct potential name misspellings in text using glossary with balanced context awareness
    """
    if not glossary:
        return text, []
    
    # Protected phrases - things that should NEVER be corrected
    protected_phrases = [
        'new orleans', 'orleans', 'louisiana', 'crescent city',
        'the', 'this', 'that', 'these', 'those',
        'president', 'member', 'american rescue', 'heritage month',
        'character count', 'document metadata', 'total segments',
    ]
    
    # Street indicators
    street_indicators = ['street', 'st', 'avenue', 'ave', 'road', 'rd', 
                        'boulevard', 'blvd', 'drive', 'dr', 'lane', 'ln',
                        'place', 'pl', 'court', 'ct', 'way', 'parkway']
    
    corrections = []
    words = text.split()
    corrected_words = words.copy()
    
    i = 0
    while i < len(words):
        # Skip protected words
        word_lower = words[i].lower().strip('.,!?;:')
        if word_lower in protected_phrases:
            i += 1
            continue
        
        # Check two-word protected phrases
        if i < len(words) - 1:
            two_word = f"{words[i]} {words[i+1]}".lower().strip('.,!?;:')
            if two_word in protected_phrases:
                i += 2
                continue
        
        # THREE-WORD STREET NAME CORRECTION (for "Chop a Tulas")
        if i < len(words) - 2:
            three_word = f"{words[i]} {words[i+1]} {words[i+2]}"
            three_word_clean = three_word.strip('.,!?;:')
            
            # Check if this could be a street name (capitalized words)
            if (words[i][0].isupper() and 
                words[i+1][0].islower() and  # "a" in "Chop a Tulas"
                words[i+2][0].isupper()):
                
                best_match = None
                best_score = 0
                
                for correct_name in glossary:
                    if any(indicator in correct_name.lower() for indicator in street_indicators):
                        score = similarity_score(three_word_clean, correct_name)
                        if score > best_score and score >= threshold:
                            best_score = score
                            best_match = correct_name
                
                if best_match:
                    corrections.append({
                        "original": three_word_clean,
                        "corrected": best_match,
                        "confidence": round(best_score, 3),
                        "position": f"words {i}-{i+2}"
                    })
                    
                    # Replace with corrected name
                    parts = best_match.split()
                    for j in range(min(3, len(parts))):
                        if i + j < len(corrected_words):
                            corrected_words[i + j] = parts[j]
                    
                    i += 3
                    continue
        
        # SINGLE WORD STREET NAME CORRECTION (for "Magziney Street")
        if i < len(words) - 1:
            current_word = words[i].strip('.,!?;:')
            next_word = words[i+1].lower().strip('.,!?;:')
            
            # Only correct if followed by street indicator
            if next_word in street_indicators and len(current_word) > 3:
                best_match = None
                best_score = 0
                
                for correct_name in glossary:
                    if any(indicator in correct_name.lower() for indicator in street_indicators):
                        street_parts = correct_name.split()
                        for part in street_parts:
                            if part.lower() not in street_indicators:
                                score = similarity_score(current_word, part)
                                if score > best_score and score >= threshold:
                                    best_score = score
                                    best_match = part
                
                if best_match:
                    corrections.append({
                        "original": current_word,
                        "corrected": best_match,
                        "confidence": round(best_score, 3),
                        "position": f"word {i}"
                    })
                    
                    punctuation = ''
                    if words[i] and words[i][-1] in '.,!?;:':
                        punctuation = words[i][-1]
                    corrected_words[i] = best_match + punctuation
        
        # SINGLE WORD CORRECTION (for "Quatery" → "Quarter")
        if i < len(words) - 1:
            current_word = words[i].strip('.,!?;:')
            next_word = words[i+1].strip('.,!?;:')
            
            # Check if this looks like "French Quatery" pattern
            if (len(current_word) > 3 and 
                current_word[0].isupper() and 
                next_word[0].isupper() and
                len(next_word) > 4):
                
                # Check the second word for misspelling
                best_match = None
                best_score = 0
                
                # Look for neighborhood/area names in glossary
                for correct_name in glossary:
                    # Skip street names with indicators
                    if not any(indicator in correct_name.lower() for indicator in street_indicators):
                        score = similarity_score(next_word, correct_name)
                        if score > best_score and score >= 0.85:  # Higher threshold
                            best_score = score
                            best_match = correct_name
                
                if best_match and best_match.lower() not in protected_phrases:
                    corrections.append({
                        "original": next_word,
                        "corrected": best_match,
                        "confidence": round(best_score, 3),
                        "position": f"word {i+1}"
                    })
                    
                    punctuation = ''
                    if words[i+1] and words[i+1][-1] in '.,!?;:':
                        punctuation = words[i+1][-1]
                    corrected_words[i+1] = best_match + punctuation
        
        # COUNCIL MEMBER NAME CORRECTION
        if i > 0:
            prev_words = ' '.join(words[max(0, i-2):i]).lower()
            council_indicators = ['councilmember', 'council member', 'commissioner']
            
            if any(indicator in prev_words for indicator in council_indicators):
                if i < len(words) - 1:
                    two_word = f"{words[i]} {words[i+1]}"
                    two_word_clean = two_word.strip('.,!?;:')
                    
                    best_match = None
                    best_score = 0
                    
                    for correct_name in glossary:
                        if not any(indicator in correct_name.lower() for indicator in street_indicators):
                            score = similarity_score(two_word_clean, correct_name)
                            if score > best_score and score >= threshold:
                                best_score = score
                                best_match = correct_name
                    
                    if best_match:
                        corrections.append({
                            "original": two_word_clean,
                            "corrected": best_match,
                            "confidence": round(best_score, 3),
                            "position": f"words {i}-{i+1}"
                        })
                        
                        punctuation = ''
                        if words[i+1] and words[i+1][-1] in '.,!?;:':
                            punctuation = words[i+1][-1]
                        
                        parts = best_match.split()
                        corrected_words[i] = parts[0]
                        corrected_words[i+1] = (parts[1] if len(parts) > 1 else corrected_words[i+1]) + punctuation
                        
                        i += 2
                        continue
        
        i += 1
    
    corrected_text = ' '.join(corrected_words)
    return corrected_text, corrections

def process_documents(files, question):
    """Process uploaded JSON, PDF, and TXT files with hybrid search approach"""
    print("🔄 Initializing ChatOllama model with optimized settings...")
    
    model_local = ChatOllama(
        model="mistral",
        temperature=0,
        num_ctx=8192,
        top_p=0.9,
    )
    
    all_docs = []
    
    # LOAD GLOSSARY FIRST
    print("\n🔍 Loading glossary for name correction...")
    glossary = load_glossary(
        council_glossary_path='put_path_to_council_glossary.json_here',
        streets_csv_path='put_path_to_streets_csv_here.csv'
    )
    
    print(f"📁 Processing {len(files)} file(s)...")
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
        return "No valid documents uploaded. Please upload JSON, PDF, or TXT files."
    
    # CORRECT THE RAW DOCUMENTS BEFORE PROCESSING
    print("\n🔧 Applying name corrections to source documents...")
    corrected_docs = []
    all_corrections = []
    
    for doc in all_docs:
        original_content = doc.page_content
        corrected_content, corrections = correct_names_in_text(original_content, glossary, threshold=0.85)
        
        if corrections:
            all_corrections.extend(corrections)
            print(f"   ✓ Made {len(corrections)} corrections in {doc.metadata.get('source', 'document')}")
            
            # Print detailed before/after
            print("\n" + "="*80)
            print(f"📝 CORRECTION DETAILS: {doc.metadata.get('source', 'document')}")
            print("="*80)
            
            for idx, correction in enumerate(corrections, 1):
                print(f"\n{idx}. Position: {correction['position']}")
                print(f"   Original:  '{correction['original']}'")
                print(f"   Corrected: '{correction['corrected']}'")
                print(f"   Confidence: {correction['confidence']*100:.1f}%")
            
            # Show text snippets
            print("\n" + "-"*80)
            print("ORIGINAL TEXT (first 800 chars):")
            print("-"*80)
            print(original_content[:800] + "..." if len(original_content) > 800 else original_content)
            
            print("\n" + "-"*80)
            print("CORRECTED TEXT (first 800 chars):")
            print("-"*80)
            print(corrected_content[:800] + "..." if len(corrected_content) > 800 else corrected_content)
            print("="*80 + "\n")
        
        corrected_doc = Document(
            page_content=corrected_content,
            metadata=doc.metadata
        )
        corrected_docs.append(corrected_doc)
    
    # Print overall summary
    if all_corrections:
        print(f"\n📊 TOTAL CORRECTIONS SUMMARY")
        print("="*80)
        print(f"Total corrections made: {len(all_corrections)}")
        print("\nAll corrections:")
        for idx, correction in enumerate(all_corrections, 1):
            print(f"   {idx}. '{correction['original']}' → '{correction['corrected']}' ({correction['confidence']*100:.0f}%)")
        print("="*80 + "\n")
    else:
        print("\n✅ No corrections needed in source documents\n")
    
    all_docs = corrected_docs
    
    # [Rest of your code continues exactly as before...]
    full_text = "\n\n=== DOCUMENT SEPARATOR ===\n\n".join([doc.page_content for doc in all_docs])
    print(f"📄 Full document length: {len(full_text)} characters")
    
    print("✂️ Splitting into optimized chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ": ", " ", ""],
        keep_separator=True
    )
    doc_splits = text_splitter.split_documents(all_docs)
    print(f"   Created {len(doc_splits)} chunks")
    
    print("🧠 Creating Ollama embeddings...")
    embeddings = OllamaEmbeddings(model='nomic-embed-text')
    
    print("💾 Creating vector store...")
    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name="rag-chroma",
        embedding=embeddings,
    )
    
    keyword_indicators = [
        'find all', 'list all', 'every instance', 'all occurrences', 
        'how many times', 'count', 'all mentions', 'every time',
        'each occurrence', 'everywhere', 'throughout', 'complete list'
    ]
    
    is_comprehensive_search = any(indicator in question.lower() for indicator in keyword_indicators)
    
    basic_info_indicators = [
        'when did', 'what time', 'what date', 'who attended',
        'who was present', 'roll call', 'meeting start'
    ]
    needs_beginning = any(indicator in question.lower() for indicator in basic_info_indicators)
    
    if is_comprehensive_search:
        print("🔍 Detected comprehensive search - using FULL document text...")
        context = full_text
        search_type = "comprehensive"
    elif needs_beginning:
        print("🔍 Detected basic info question - prioritizing document beginning...")
        beginning_chunks = doc_splits[:5]
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 10, "fetch_k": 30, "lambda_mult": 0.6}
        )
        retrieved_docs = retriever.get_relevant_documents(question)
        all_relevant = beginning_chunks + retrieved_docs
        context = "\n\n".join([doc.page_content for doc in all_relevant])
        search_type = "beginning_priority"
    else:
        print("🔍 Using semantic retrieval for targeted question...")
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 12, "fetch_k": 40, "lambda_mult": 0.6}
        )
        retrieved_docs = retriever.get_relevant_documents(question)
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        search_type = "semantic"
    
    print(f"✅ Generating answer using {search_type} search...")
    
    if is_comprehensive_search:
        after_rag_template = """You are an expert document analyst. You have been given the COMPLETE, FULL TEXT of all documents. Your task is to comprehensively search through ALL of it.

FULL DOCUMENT TEXT:
{context}

QUESTION: {question}

CRITICAL INSTRUCTIONS FOR COMPREHENSIVE SEARCH:
1. You have the ENTIRE document - search through ALL of it thoroughly
2. For "find all" or "list all" queries:
   - Go through the ENTIRE text systematically
   - List EVERY single occurrence you find
   - Include surrounding context for each occurrence
   - Count the total number of occurrences
   - Quote the exact text for each instance
3. Do not stop after finding a few examples - continue through the entire document
4. Organize your findings clearly (e.g., numbered list, by section, by timestamp)
5. At the end, provide a summary count: "Total occurrences found: X"

COMPREHENSIVE ANSWER:"""
    else:
        after_rag_template = """You are an expert document analyst specializing in meeting transcripts. Answer based STRICTLY on the provided context.

CONTEXT:
{context}

QUESTION: {question}

INSTRUCTIONS:
1. For basic information (date, time, attendees):
   - Check the very beginning of the document first
   - Look for headers, metadata, or opening statements
   - Quote the exact text where you found the information

2. For general questions:
   - Read all provided context carefully
   - Quote exact passages when possible
   - Reference specific locations (beginning, middle, timestamps, etc.)
   - Be thorough and precise

3. If information is not in the provided context:
   - State clearly: "I cannot find that information in the provided context."
   - Do not speculate or use external knowledge

ANSWER:"""
    
    prompt = ChatPromptTemplate.from_template(after_rag_template)
    chain = prompt | model_local | StrOutputParser()
    
    answer = chain.invoke({"context": context, "question": question})
    
    return answer

# Create enhanced Gradio interface
iface = gr.Interface(
    fn=process_documents,
    inputs=[
        gr.File(
            label="Upload Meeting Transcripts (JSON, PDF, TXT)", 
            file_count="multiple",
            file_types=[".json", ".pdf", ".txt"]
        ),
        gr.Textbox(
            label="Ask a question about your documents", 
            lines=4,
            placeholder="Examples:\n- When did this meeting start? (uses beginning priority)\n- Find all mentions of 'budget' (uses full text)\n- What was discussed about Crescent Care? (uses semantic search)\n- List every vote taken with results (uses full text)"
        )
    ],
    outputs=gr.Textbox(label="Answer", lines=20, show_copy_button=True),
    title="Hybrid Search Meeting Transcript Analyzer with Name Correction (Ollama)",
    description="""Upload meeting transcripts for intelligent analysis with automatic name correction.
    
    🔍 **Smart Search Modes:**
    - **Comprehensive Search**: Automatically used for "find all", "list all" queries - searches entire document
    - **Beginning Priority**: Automatically used for date/time/attendee questions - focuses on document start
    - **Semantic Search**: Used for specific topical questions - finds most relevant sections
    
    ✅ **Name Correction**: Automatically corrects misspelled council member and street names at source
    
    Optimized for New Orleans city government meetings and transcripts.""",
    examples=[
        [None, "When did this meeting start? What was the exact date and time?"],
        [None, "Find all instances of the word 'amendment' in the entire transcript"],
        [None, "List every vote that was taken with the results"],
        [None, "Who attended this meeting? List all council members present."],
        [None, "What was discussed about the budget and Crescent Care?"],
        [None, "Count how many times 'resolution' appears in the document"],
        [None, "Find all mentions of dollar amounts or financial figures"]
    ],
    theme=gr.themes.Soft(),
    allow_flagging="never"
)

if __name__ == "__main__":
    iface.launch(share=False)