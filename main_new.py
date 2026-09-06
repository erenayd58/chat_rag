"""
CLI Chat application with automatic document ingestion
"""
import os
import sys
from typing import Optional
from pipeline import RAGPipeline
from config import Settings
from utils import DocumentTracker


def print_banner():
    """Print application banner"""
    print("\n" + "="*80)
    print("  RAG CONVERSATIONAL CHAT")
    print("  Retrieval-Augmented Generation with Document Ingestion")
    print("="*80 + "\n")


def ingest_documents_from_folder(
    pipeline: RAGPipeline,
    tracker: DocumentTracker,
    settings: Settings
) -> int:
    """
    Ingest new documents from the configured folder
    
    Args:
        pipeline: RAG pipeline instance
        tracker: Document tracker instance
        settings: Settings instance
    
    Returns:
        Number of new documents ingested
    """
    input_path = settings.documents_input_path
    
    if not os.path.exists(input_path):
        print(f"📁 Input folder not found: {input_path}")
        print(f"   Creating folder...")
        os.makedirs(input_path, exist_ok=True)
        print(f"   ✓ Folder created. Add documents to {input_path} and restart.")
        return 0
    
    if not os.path.isdir(input_path):
        print(f"⚠️  {input_path} is not a directory")
        return 0
    
    print(f"📂 Scanning for documents in: {input_path}")
    print(f"   Recursive: {settings.documents_recursive}")
    
    # Find all files
    import glob
    if settings.documents_recursive:
        pattern = os.path.join(input_path, '**', '*')
        all_files = glob.glob(pattern, recursive=True)
    else:
        all_files = [
            os.path.join(input_path, f)
            for f in os.listdir(input_path)
            if os.path.isfile(os.path.join(input_path, f))
        ]
    
    # Filter to supported files and check if already ingested
    new_files = []
    skipped_files = []
    
    for file_path in all_files:
        if not os.path.isfile(file_path):
            continue
        
        # Check if parser supports this file
        if pipeline.parser_factory.get_parser(file_path) is None:
            continue
        
        # Check if already ingested
        if tracker.is_document_ingested(file_path):
            skipped_files.append(file_path)
        else:
            new_files.append(file_path)
    
    print(f"   Found {len(new_files)} new document(s)")
    print(f"   Skipping {len(skipped_files)} already ingested document(s)")
    
    if not new_files:
        return 0
    
    # Ingest new documents
    print(f"\n📥 Ingesting new documents...")
    ingested_count = 0
    
    for i, file_path in enumerate(new_files, 1):
        filename = os.path.basename(file_path)
        print(f"\n[{i}/{len(new_files)}] Processing: {filename}")
        
        try:
            chunks = pipeline.ingest_document_from_file(file_path)
            
            # Only mark as ingested if we got chunks
            if chunks:
                # Mark as ingested in tracker
                doc_id = os.path.basename(file_path).replace('.', '_')
                tracker.mark_as_ingested(
                    file_path=file_path,
                    doc_id=doc_id,
                    chunk_count=len(chunks),
                    metadata={'filename': filename}
                )
                
                print(f"  ✓ Success: {len(chunks)} chunks created")
                ingested_count += 1
            else:
                print(f"  ⚠️  Skipped: Document too short or empty")
            
        except Exception as e:
            print(f"  ✗ Failed: {e}")
    
    return ingested_count


def chat_loop(pipeline: RAGPipeline):
    """
    Main chat loop for conversational interaction
    
    Args:
        pipeline: RAG pipeline instance
    """
    print("\n" + "="*80)
    print("💬 CHAT MODE")
    print("="*80)
    print("\nCommands:")
    print("  - Type your question and press Enter")
    print("  - 'help' - Show this help message")
    print("  - 'stats' - Show document statistics")
    print("  - 'clear' - Clear conversation history")
    print("  - 'exit' or 'quit' - Exit the application")
    print("\n" + "-"*80 + "\n")
    
    while True:
        try:
            # Get user input
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.lower() in ['exit', 'quit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            elif user_input.lower() == 'help':
                print("\nAvailable commands:")
                print("  help  - Show this help message")
                print("  stats - Show document statistics")
                print("  clear - Clear conversation history")
                print("  exit  - Exit the application")
                continue
            
            elif user_input.lower() == 'stats':
                show_statistics(pipeline)
                continue
            
            elif user_input.lower() == 'clear':
                pipeline.clear_conversation()
                print("✓ Conversation history cleared\n")
                continue
            
            # Process query
            print("\n🔍 Processing your question...", end='', flush=True)
            
            try:
                # Use the complete query method that generates an answer
                result = pipeline.query(
                    question=user_input,
                    top_k=5,
                    use_query_expansion=True,
                    use_reranking=True,
                    retrieval_method='hybrid',
                    temperature=0.3,
                    max_tokens=500
                )
                
                print("\r" + " "*40 + "\r", end='')  # Clear "Processing..." message
                
                if not result['sources']:
                    print("Assistant: I couldn't find any relevant information in the documents.")
                    print("           Please try rephrasing your question.\n")
                    continue
                
                # Display the generated answer
                print(f"Assistant: {result['answer']}\n")
                
                # Optionally show sources
                show_sources = input("📚 Show sources? (y/n): ").strip().lower()
                if show_sources == 'y':
                    print("\n📄 Sources:")
                    for i, source in enumerate(result['sources'][:3], 1):
                        print(f"\n{i}. {source['document']}")
                        if source['section']:
                            print(f"   Section: {source['section']}")
                        print(f"   Relevance Score: {source['score']:.3f}")
                        print(f"   Preview: {source['content_preview']}")
                    print()
                
            except Exception as e:
                print(f"\n⚠️  Error processing query: {e}\n")
        
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except EOFError:
            print("\n\n👋 Goodbye!")
            break


def show_statistics(pipeline: RAGPipeline):
    """
    Show statistics about ingested documents
    
    Args:
        pipeline: RAG pipeline instance
    """
    from utils import DocumentTracker
    
    tracker = DocumentTracker()
    stats = tracker.get_statistics()
    
    print("\n" + "="*80)
    print("📊 DOCUMENT STATISTICS")
    print("="*80)
    
    if stats['total_documents'] == 0:
        print("\nNo documents ingested yet.")
        print(f"Add documents to '{pipeline.settings.documents_input_path}' and restart.\n")
        return
    
    print(f"\nTotal Documents: {stats['total_documents']}")
    print(f"Total Chunks: {stats['total_chunks']}")
    print(f"Total Size: {stats['total_size_bytes'] / 1024 / 1024:.2f} MB")
    
    if 'oldest_ingestion' in stats:
        print(f"Oldest Ingestion: {stats['oldest_ingestion'][:19]}")
        print(f"Latest Ingestion: {stats['latest_ingestion'][:19]}")
    
    # Vector DB stats
    db_count = pipeline.vector_db.count()
    print(f"\nVector Database:")
    print(f"  Chunks stored: {db_count}")
    
    print()


def main():
    """Main application"""
    
    print_banner()
    
    # Initialize settings (reads from .env)
    print("⚙️  Loading configuration...")
    settings = Settings()
    
    try:
        settings.validate()
    except ValueError as e:
        print(f"\n❌ Configuration error: {e}")
        print("   Please set AZURE_ENDPOINT and AZURE_API_KEY in .env file")
        print("\nTo get started:")
        print("  1. Copy env.example to .env")
        print("  2. Edit .env and add your Azure OpenAI credentials")
        print("  3. Run the application again\n")
        return
    
    print("✓ Configuration loaded\n")
    
    # Initialize RAG pipeline
    print("🚀 Initializing RAG pipeline...")
    try:
        rag_pipeline = RAGPipeline(settings=settings)
        print("✓ Pipeline initialized\n")
    except Exception as e:
        print(f"\n❌ Failed to initialize pipeline: {e}\n")
        return
    
    # Initialize document tracker
    tracker = DocumentTracker()
    
    # Check and ingest documents from input folder
    print("="*80)
    print("DOCUMENT INGESTION")
    print("="*80 + "\n")
    
    new_docs_count = ingest_documents_from_folder(rag_pipeline, tracker, settings)
    
    if new_docs_count > 0:
        print(f"\n✓ Successfully ingested {new_docs_count} new document(s)")
    
    # Show statistics
    show_statistics(rag_pipeline)
    
    # Check if any documents are loaded
    if rag_pipeline.vector_db.count() == 0:
        print("⚠️  No documents found in the knowledge base.")
        print(f"   Add documents to '{settings.documents_input_path}' and restart the application.\n")
        return
    
    # Start chat loop
    chat_loop(rag_pipeline)


if __name__ == "__main__":
    main()

