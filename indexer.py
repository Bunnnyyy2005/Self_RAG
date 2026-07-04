import os
import shutil
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Paths
MD_DIR = "./data/markdown_context"
CHROMA_PATH = "./data/chroma_db"

def build_smart_vector_db():
    print("🧱 STEP 1: Initiating Semantic Data Ingestion...")
    
    # 1. Clear out any old, broken databases
    if os.path.exists(CHROMA_PATH):
        print("   🗑️ Wiping old database to ensure a clean slate...")
        shutil.rmtree(CHROMA_PATH)
    
    # 2. Load the Markdown papers
    print(f"   📚 Loading documents from {MD_DIR}...")
    loader = DirectoryLoader(MD_DIR, glob="**/*.md", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
    raw_documents = loader.load()
    
    if not raw_documents:
        print("   ❌ No markdown files found! Make sure you converted your PDFs.")
        return

    # 3. SMART CHUNKING: Split by Markdown Headers (Keeps tables/sections together)
    print("   ✂️ Applying Structure-Aware Chunking...")
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    
    # We still use a character splitter, but ONLY as a safety net if a single section is 10 pages long.
    # A massive chunk size of 3000 ensures tables are never split in half.
    safety_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=300)
    
    final_chunks = []
    
    for doc in raw_documents:
        # Extract the filename to use as a Citation Tag
        source_file = os.path.basename(doc.metadata.get("source", "Unknown_Paper"))
        
        # Split the document by its headers
        md_splits = markdown_splitter.split_text(doc.page_content)
        
        # Inject metadata into every single chunk so the AI knows where it came from
        for split in md_splits:
            split.metadata["Source_Paper"] = source_file
            
        # Apply the safety net (in case a section is too large)
        safe_splits = safety_splitter.split_documents(md_splits)
        final_chunks.extend(safe_splits)

    print(f"   🧩 Successfully created {len(final_chunks)} semantic chunks.")
    
    # Let's peek at the first chunk to see our beautiful metadata!
    if final_chunks:
        print("\n   👀 PEEK AT CHUNK #1 METADATA:")
        print(f"       {final_chunks[0].metadata}\n")

    # 4. Create local embeddings (Runs on CPU, totally free)
    print("   🧠 Initializing BAAI/bge-small Embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={'device': 'cpu'}
    )

    # 5. Save to ChromaDB
    print("   💾 Saving to Vector Database...")
    Chroma.from_documents(final_chunks, embeddings, persist_directory=CHROMA_PATH)
    
    print("✅ SUCCESS! Foundation built. Vector database is ready.")

if __name__ == "__main__":
    build_smart_vector_db()