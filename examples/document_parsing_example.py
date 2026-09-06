"""
Example demonstrating document parsing from various file formats
"""
from pipeline import RAGPipeline
from config import Settings
from components.parsers import ParserFactory


def main():
    """Demonstrate document parsing capabilities"""
    
    # Initialize settings
    settings = Settings()
    
    # Initialize RAG pipeline
    print("="*80)
    print("DOCUMENT PARSING EXAMPLE")
    print("="*80)
    
    rag_pipeline = RAGPipeline(settings=settings)
    
    # Show available parsers
    print("\n📋 Available Parsers:")
    for parser_name in rag_pipeline.parser_factory.list_parsers():
        print(f"  - {parser_name}")
    
    print("\n📄 Supported File Extensions:")
    extensions = rag_pipeline.parser_factory.list_supported_extensions()
    print(f"  {', '.join(extensions)}")
    
    # Example 1: Ingest a single PDF file
    print("\n" + "="*80)
    print("EXAMPLE 1: Ingest Single PDF File")
    print("="*80)
    
    try:
        chunks = rag_pipeline.ingest_document_from_file(
            file_path="./documents/sample.pdf",
            additional_metadata={"source": "user_upload", "category": "research"}
        )
        print(f"\n✓ Successfully ingested PDF with {len(chunks)} chunks")
    except Exception as e:
        print(f"\n✗ PDF ingestion failed: {e}")
        print("  (Create a ./documents/sample.pdf file to test this)")
    
    # Example 2: Ingest a DOCX file
    print("\n" + "="*80)
    print("EXAMPLE 2: Ingest DOCX File")
    print("="*80)
    
    try:
        chunks = rag_pipeline.ingest_document_from_file(
            file_path="./documents/report.docx",
            additional_metadata={"source": "reports", "category": "business"}
        )
        print(f"\n✓ Successfully ingested DOCX with {len(chunks)} chunks")
    except Exception as e:
        print(f"\n✗ DOCX ingestion failed: {e}")
        print("  (Create a ./documents/report.docx file to test this)")
    
    # Example 3: Ingest all documents from a directory
    print("\n" + "="*80)
    print("EXAMPLE 3: Bulk Ingest from Directory")
    print("="*80)
    
    try:
        results = rag_pipeline.ingest_documents_from_directory(
            directory_path="./documents",
            recursive=True,
            additional_metadata={"batch": "202410", "source": "local"}
        )
        
        print("\nDetailed Results:")
        for file_path, chunks in results.items():
            status = "✓" if chunks else "✗"
            print(f"  {status} {file_path}: {len(chunks)} chunks")
            
    except Exception as e:
        print(f"\n✗ Directory ingestion failed: {e}")
        print("  (Create a ./documents directory with files to test this)")
    
    # Example 4: Filter by file type
    print("\n" + "="*80)
    print("EXAMPLE 4: Ingest Only PDF Files")
    print("="*80)
    
    try:
        results = rag_pipeline.ingest_documents_from_directory(
            directory_path="./documents",
            recursive=True,
            file_pattern="*.pdf",
            additional_metadata={"type": "pdf_only"}
        )
        print(f"\n✓ Ingested {len(results)} PDF files")
    except Exception as e:
        print(f"\n✗ Failed: {e}")
    
    # Example 5: Parse without ingesting (just to see the content)
    print("\n" + "="*80)
    print("EXAMPLE 5: Parse File Without Ingesting")
    print("="*80)
    
    parser_factory = ParserFactory()
    
    # Create sample text file for demonstration
    import os
    os.makedirs("./temp_docs", exist_ok=True)
    sample_file = "./temp_docs/sample.txt"
    
    with open(sample_file, 'w') as f:
        f.write("""
Machine Learning Introduction

Machine learning is a subset of artificial intelligence that enables systems to learn from data.
It has three main categories:
1. Supervised Learning
2. Unsupervised Learning
3. Reinforcement Learning

Applications include image recognition, natural language processing, and predictive analytics.
""")
    
    # Parse the file
    text = parser_factory.parse_file(sample_file)
    metadata = parser_factory.get_metadata(sample_file)
    
    print(f"\nParsed Text ({len(text)} characters):")
    print(text[:200] + "..." if len(text) > 200 else text)
    
    print(f"\nMetadata:")
    for key, value in metadata.items():
        print(f"  {key}: {value}")
    
    # Cleanup
    os.remove(sample_file)
    os.rmdir("./temp_docs")
    
    # Example 6: Query the ingested documents
    print("\n" + "="*80)
    print("EXAMPLE 6: Query Ingested Documents")
    print("="*80)
    
    if rag_pipeline.vector_db.count() > 0:
        query = "What is machine learning?"
        print(f"\nQuery: {query}")
        
        results, metadata = rag_pipeline.retrieve(query, top_k=3)
        
        print(f"\n✓ Found {len(results)} results")
        for i, result in enumerate(results, 1):
            print(f"\n[Result {i}]")
            print(f"  Document: {result.chunk.doc_title}")
            print(f"  Score: {result.score:.4f}")
            print(f"  Content: {result.chunk.content[:150]}...")
    else:
        print("\n⚠ No documents ingested yet. Ingest some documents first!")
    
    # Show parser-specific features
    print("\n" + "="*80)
    print("PARSER-SPECIFIC FEATURES")
    print("="*80)
    
    print("""
PDF Parser (PyMuPDF or unstructured):
  - Extracts text from all pages
  - Preserves page structure
  - Optional: Extract images
  - Optional: Extract tables (with unstructured)
  - Extracts metadata (title, author, etc.)

DOCX Parser:
  - Extracts paragraphs
  - Extracts tables
  - Preserves document structure
  - Extracts properties (author, created date, etc.)

Markdown Parser:
  - Preserves or strips markdown formatting
  - Counts headers
  - Extracts YAML frontmatter

Image Parser (OCR):
  - Uses Tesseract OCR
  - Supports multiple languages
  - Extracts text from images
  - Works with PNG, JPG, GIF, etc.

Text Parser:
  - Handles various encodings (UTF-8, Latin-1)
  - Supports TXT, CSV, JSON, XML, HTML
  - Fast and simple
    """)


if __name__ == "__main__":
    main()

