from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pathlib import Path
import pandas as pd
import logging
import hashlib
import json
import os
from typing import Optional, Dict, List, Union
from config import get_chroma_client, get_or_create_collection
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self, persist_directory: str = "chroma_db"):
        """Initialize document processor with tracking and caching"""
        self.client = get_chroma_client(persist_directory)
        self.collection = get_or_create_collection(self.client)
        self.processed_files_path = "processed_files.json"
        self.cache_dir = Path("document_cache")
        
        # Create cache directory if it doesn't exist
        self.cache_dir.mkdir(exist_ok=True)
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Load processed files history
        self.processed_files = self.load_processed_files()
        logger.info("✓ Document processor initialized")

    def load_processed_files(self) -> Dict[str, str]:
        """Load record of processed files"""
        if os.path.exists(self.processed_files_path):
            with open(self.processed_files_path, 'r') as f:
                return json.load(f)
        return {}

    def save_processed_files(self):
        """Save record of processed files"""
        with open(self.processed_files_path, 'w') as f:
            json.dump(self.processed_files, f, indent=2)

    def get_file_hash(self, file_path: Path) -> str:
        """Get hash of file content with caching"""
        cache_file = self.cache_dir / f"{file_path.name}.hash"
        
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                return f.read()
        
        with open(file_path, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
            
        with open(cache_file, 'w') as f:
            f.write(file_hash)
            
        return file_hash

    def should_process_file(self, file_path: Path) -> bool:
        """Check if file needs processing using cached hash"""
        try:
            current_hash = self.get_file_hash(file_path)
            file_key = str(file_path)
            
            if file_key not in self.processed_files:
                return True
            
            return self.processed_files[file_key] != current_hash
        except Exception as e:
            logger.error(f"Error checking file {file_path}: {e}")
            return False

    def mark_file_processed(self, file_path: Path):
        """Mark file as processed and update cache"""
        self.processed_files[str(file_path)] = self.get_file_hash(file_path)
        self.save_processed_files()

    def process_pdf(self, file_path: Path) -> bool:
        """Process a PDF file with progress tracking"""
        try:
            if not self.should_process_file(file_path):
                logger.info(f"Skipping unchanged file: {file_path}")
                return False

            logger.info(f"Processing PDF: {file_path}")
            loader = PyPDFLoader(str(file_path))
            pages = loader.load()
            
            # Show progress bar for chunking
            chunks = []
            for page in tqdm(pages, desc="Splitting pages"):
                page_chunks = self.text_splitter.split_documents([page])
                chunks.extend(page_chunks)
            
            # Delete existing chunks for this file
            self.delete_file_chunks(str(file_path))
            
            texts = [chunk.page_content for chunk in chunks]
            metadatas = [{
                "source": str(file_path),
                "page": chunk.metadata.get("page", 0),
                "type": "pdf",
                "chunk_number": i,
                "total_chunks": len(chunks)
            } for i, chunk in enumerate(chunks)]
            
            ids = [f"pdf_{file_path.stem}_{i}" for i in range(len(chunks))]

            if texts:
                # Process in batches with progress bar
                batch_size = 100
                for i in tqdm(range(0, len(texts), batch_size), desc="Storing chunks"):
                    end_idx = min(i + batch_size, len(texts))
                    self.collection.add(
                        documents=texts[i:end_idx],
                        metadatas=metadatas[i:end_idx],
                        ids=ids[i:end_idx]
                    )
                logger.info(f"✓ Added {len(texts)} chunks from {file_path}")

            self.mark_file_processed(file_path)
            return True

        except Exception as e:
            logger.error(f"Error processing PDF {file_path}: {e}")
            return False

    def process_csv(self, file_path: Path) -> bool:
        """Process a CSV file with validation and progress tracking"""
        try:
            if not self.should_process_file(file_path):
                logger.info(f"Skipping unchanged file: {file_path}")
                return False

            logger.info(f"Processing CSV: {file_path}")
            
            # Read CSV with progress feedback
            chunks = []
            metadatas = []
            ids = []
            chunk_size = 1000  # Process CSV in chunks
            row_counter = 0
            
            for chunk_df in tqdm(
                pd.read_csv(file_path, chunksize=chunk_size), 
                desc="Reading CSV"
            ):
                # Clean and validate data
                chunk_df = chunk_df.fillna("")
                chunk_df = chunk_df.astype(str)
                
                # Convert each row to a structured string
                for _, row in chunk_df.iterrows():
                    # Format row as key-value pairs
                    row_text = "\n".join([
                        f"{col}: {val}" for col, val in row.items() 
                        if val.strip() != ''
                    ])
                    
                    if row_text.strip():
                        text_chunks = self.text_splitter.split_text(row_text)
                        for chunk_idx, chunk in enumerate(text_chunks):
                            chunks.append(chunk)
                            metadatas.append({
                                "source": str(file_path),
                                "row": row_counter,
                                "type": "csv",
                                "chunk_number": chunk_idx,
                                "total_chunks": len(text_chunks)
                            })
                            ids.append(f"csv_{file_path.stem}_{row_counter}_{chunk_idx}")
                    row_counter += 1

            if chunks:
                # Delete existing chunks for this file
                self.delete_file_chunks(str(file_path))
                
                # Process in batches with progress bar
                batch_size = 100
                for i in tqdm(range(0, len(chunks), batch_size), desc="Storing chunks"):
                    end_idx = min(i + batch_size, len(chunks))
                    self.collection.add(
                        documents=chunks[i:end_idx],
                        metadatas=metadatas[i:end_idx],
                        ids=ids[i:end_idx]
                    )
                logger.info(f"✓ Added {len(chunks)} chunks from {file_path}")

            self.mark_file_processed(file_path)
            return True

        except Exception as e:
            logger.error(f"Error processing CSV {file_path}: {e}")
            return False

    def delete_file_chunks(self, file_path: str):
        """Delete all chunks for a given file"""
        try:
            # Get all chunks for this file
            results = self.collection.get(
                where={"source": file_path}
            )
            if results and results['ids']:
                self.collection.delete(
                    ids=results['ids']
                )
                logger.info(f"Deleted existing chunks for {file_path}")
        except Exception as e:
            logger.error(f"Error deleting chunks for {file_path}: {e}")

    def process_file(self, file_path: Union[str, Path]) -> bool:
        """Process a single file (PDF or CSV) with type detection"""
        try:
            path = Path(file_path)
            file_extension = path.suffix.lower()
            
            if not path.exists():
                logger.error(f"File not found: {path}")
                return False
                
            if file_extension == '.pdf':
                return self.process_pdf(path)
            elif file_extension == '.csv':
                return self.process_csv(path)
            else:
                logger.error(f"Unsupported file type: {file_extension}")
                return False
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            return False

    def process_directory(self, directory: Union[str, Path] = "data/raw"):
        """Process all supported files in directory with progress tracking"""
        try:
            base_dir = Path(directory)
            if not base_dir.exists():
                logger.error(f"Directory not found: {base_dir}")
                return
            
            # Collect all supported files
            files_to_process = []
            for ext in ['.pdf', '.csv']:
                files_to_process.extend(base_dir.rglob(f"*{ext}"))
            
            if not files_to_process:
                logger.info(f"No supported files found in {directory}")
                return
            
            # Process files with progress bar
            for file_path in tqdm(files_to_process, desc="Processing files"):
                try:
                    self.process_file(file_path)
                except Exception as e:
                    logger.error(f"Failed to process {file_path}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error processing directory {directory}: {e}")
            raise

if __name__ == "__main__":
    processor = DocumentProcessor()
    processor.process_directory()