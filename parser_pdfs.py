import os
from pathlib import Path
import concurrent.futures
import pymupdf4llm

def convert_pdf_to_markdown(pdf_path: Path, output_dir: Path):
    """
    Converts a single PDF file into a clean Markdown file.
    Preserves structural headers, tables, and mathematical notations.
    """
    try:
        print(f"[PROCESSING] {pdf_path.name}...")
        
        # Perform the high-speed Markdown extraction
        markdown_text = pymupdf4llm.to_markdown(str(pdf_path))
        
        # Define the output file path
        output_file_path = output_dir / f"{pdf_path.stem}.md"
        
        # Save the structured text
        with open(output_file_path, "w", encoding="utf-8") as f:
            f.write(markdown_text)
            
        print(f"[SUCCESS] Saved to {output_file_path.name}")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to process {pdf_path.name}: {str(e)}")
        return False

def batch_process_documents(source_folder: str, output_folder: str):
    """
    Scans the source folder for PDFs and processes them concurrently.
    """
    source_path = Path(source_folder)
    output_path = Path(output_folder)
    
    # Create output directory if it does not exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all PDF files in the source directory
    pdf_files = list(source_path.glob("*.pdf"))
    
    if not pdf_files:
        print(f"No PDF files found in '{source_folder}'. Please add some documents.")
        return

    print(f"Found {len(pdf_files)} documents. Starting high-speed batch extraction...\n")

    # Use a ThreadPoolExecutor for concurrent file I/O operations
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(convert_pdf_to_markdown, pdf, output_path) 
            for pdf in pdf_files
        ]
        
        # Wait for all files to finish processing
        results = [future.result() for future in concurrent.futures.as_completed(futures)]
    
    successful_runs = sum(1 for r in results if r)
    print(f"\n[COMPLETED] Successfully processed {successful_runs}/{len(pdf_files)} documents.")

if __name__ == "__main__":
    # Define your local data directories
    SOURCE_DIR = "./data/raw_pdfs"
    OUTPUT_DIR = "./data/markdown_context"
    
    # Create sample directory structure if running for the first time
    Path(SOURCE_DIR).mkdir(parents=True, exist_ok=True)
    
    batch_process_documents(SOURCE_DIR, OUTPUT_DIR)