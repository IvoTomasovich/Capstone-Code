from langchain_ollama import OllamaEmbeddings

print("Testing OllamaEmbeddings connection...")

try:
    embeddings = OllamaEmbeddings(
        model='nomic-embed-text',
        base_url="http://localhost:11434"
    )
    
    # Try to embed a simple test
    result = embeddings.embed_query("test")
    print(f"✅ Success! Embedding dimension: {len(result)}")
    print(f"First 5 values: {result[:5]}")
    
except Exception as e:
    print(f"❌ Error: {e}")