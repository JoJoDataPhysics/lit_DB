import os
import json
import logging
import hashlib
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

import pdfplumber
from PyPDF2 import PdfReader

from src.models import AnalysisResult, AppConfig, TopicKeywords
from src.ollama_client import OllamaClient
from src.config_manager import ConfigManager
from src.database_manager import DatabaseManager


class PDFAnalyzer:
    def __init__(self, config_path: str = "config.yaml"):
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.get_config()
        self.ollama_client = OllamaClient(self.config.ollama)
        self.logger = logging.getLogger(__name__)
        
        # Initialize database manager if persistence is enabled
        self.db_manager = None
        if self.config.database.enable_persistence:
            try:
                from src.database_manager import DatabaseManager
                self.db_manager = DatabaseManager(self.config.database.path)
                self.logger.info("Database persistence enabled")
            except ImportError as e:
                self.logger.warning(f"Could not import DatabaseManager: {e}")
            except Exception as e:
                self.logger.error(f"Failed to initialize database: {e}")
        
        self._setup_logging()
        self._ensure_directories()
        
    def _setup_logging(self):
        logging.basicConfig(
            level=getattr(logging, self.config.logging.level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.config.logging.file),
                logging.StreamHandler()
            ]
        )
    
    def _ensure_directories(self):
        Path(self.config.pdf.folder_path).mkdir(parents=True, exist_ok=True)
        Path(self.config.output.results_folder).mkdir(parents=True, exist_ok=True)
        Path(self.config.logging.file).parent.mkdir(parents=True, exist_ok=True)
    
    def extract_text_from_pdf(self, pdf_path: Path) -> Dict[str, Any]:
        try:
            with pdfplumber.open(pdf_path) as pdf:
                text_chunks = []
                total_text = ""
                
                for page_num, page in enumerate(pdf.pages):
                    page_text = page.extract_text()
                    if page_text:
                        total_text += page_text + "\n"
                
                text_chunks = self._chunk_text(total_text)
                word_count = len(total_text.split())
                page_count = len(pdf.pages)
                
                return {
                    "text_chunks": text_chunks,
                    "word_count": word_count,
                    "page_count": page_count,
                    "full_text": total_text
                }
                
        except Exception as e:
            self.logger.error(f"Failed to extract text from {pdf_path}: {e}")
            raise
    
    def _extract_pdf_metadata(self, pdf_path: Path) -> Dict[str, Any]:
        """Extract PDF metadata using PyPDF2."""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PdfReader(file)
                metadata = pdf_reader.metadata
                
                if metadata:
                    return {
                        'author': metadata.get('/Author', '').strip() if metadata.get('/Author') else None,
                        'title': metadata.get('/Title', '').strip() if metadata.get('/Title') else None,
                        'subject': metadata.get('/Subject', '').strip() if metadata.get('/Subject') else None,
                        'creator': metadata.get('/Creator', '').strip() if metadata.get('/Creator') else None,
                        'producer': metadata.get('/Producer', '').strip() if metadata.get('/Producer') else None,
                        'creation_date': str(metadata.get('/CreationDate', '')) if metadata.get('/CreationDate') else None,
                        'modification_date': str(metadata.get('/ModDate', '')) if metadata.get('/ModDate') else None,
                        'file_size': pdf_path.stat().st_size,
                    }
                else:
                    return {
                        'author': None,
                        'title': None,
                        'subject': None,
                        'creator': None,
                        'producer': None,
                        'creation_date': None,
                        'modification_date': None,
                        'file_size': pdf_path.stat().st_size,
                    }
                    
        except Exception as e:
            self.logger.warning(f"Failed to extract metadata from {pdf_path}: {e}")
            return {
                'author': None,
                'title': None,
                'subject': None,
                'creator': None,
                'producer': None,
                'creation_date': None,
                'modification_date': None,
                'file_size': pdf_path.stat().st_size,
            }
    
    def _chunk_text(self, text: str) -> List[str]:
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        chunk_size = self.config.analysis.chunk_size
        overlap = self.config.analysis.context_overlap
        
        for word in words:
            current_chunk.append(word)
            current_length += len(word) + 1
            
            if current_length >= chunk_size:
                chunk_text = " ".join(current_chunk)
                chunks.append(chunk_text)
                
                overlap_words = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                current_chunk = overlap_words
                current_length = sum(len(w) + 1 for w in overlap_words)
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
            
        return chunks
    
    def _calculate_file_hash(self, pdf_path: Path) -> str:
        """Calculate SHA-256 hash of PDF file content only (for internal use)"""
        try:
            sha256_hash = hashlib.sha256()
            with open(pdf_path, "rb") as f:
                # Read file in chunks to handle large files efficiently
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(chunk)
            return sha256_hash.hexdigest()
        except Exception as e:
            self.logger.error(f"Failed to calculate hash for {pdf_path}: {e}")
            # Return a fallback hash based on filename and size
            return hashlib.sha256(f"{pdf_path.name}_{pdf_path.stat().st_size}".encode()).hexdigest()
    
    def _calculate_analysis_hash(self, file_hash: str, model: str) -> str:
        """Calculate combined hash of file content + model used for analysis uniqueness"""
        combined = f"{file_hash}_{model}"
        analysis_hash = hashlib.sha256(combined.encode()).hexdigest()
        self.logger.debug(f"Analysis hash for model {model}: {analysis_hash[:16]}...")
        return analysis_hash
    
    def _find_existing_analysis(self, analysis_hash: str) -> Optional[AnalysisResult]:
        """Check if an analysis already exists for this file + model combination"""
        results_folder = Path(self.config.output.results_folder)
        if not results_folder.exists():
            return None
        
        try:
            # Search through existing result files
            for result_file in results_folder.glob("*.json"):
                try:
                    with open(result_file, 'r') as f:
                        data = json.load(f)
                        
                    # Check if this result has the same analysis hash (file + model)
                    # For backward compatibility, also check old-style file_hash field
                    stored_hash = data.get('file_hash', '')
                    if stored_hash == analysis_hash:
                        self.logger.info(f"Found existing analysis in {result_file.name}")
                        # Convert back to AnalysisResult object, handling missing analysis_model for backward compatibility
                        if 'analysis_model' not in data and 'model_used' not in data:
                            data['analysis_model'] = 'unknown'  # Backward compatibility
                        elif 'model_used' in data:
                            data['analysis_model'] = data['model_used']  # Migrate from old field name
                        return AnalysisResult(**data)
                        
                except (json.JSONDecodeError, TypeError, KeyError) as e:
                    self.logger.warning(f"Could not parse result file {result_file}: {e}")
                    continue
                    
        except Exception as e:
            self.logger.error(f"Error searching for existing analysis: {e}")
            
        return None
    
    def analyze_pdf(self, pdf_path: Path, force_reanalysis: bool = False) -> AnalysisResult:
        self.logger.info(f"Starting analysis of {pdf_path.name}")
        
        # Get model first for hash calculation
        model = self.ollama_client.ensure_model_available()
        
        # Calculate file hash for deduplication
        file_hash = self._calculate_file_hash(pdf_path)
        analysis_hash = self._calculate_analysis_hash(file_hash, model)
        
        # Check for existing analysis (database first, then JSON fallback)
        if not force_reanalysis:
            # Database checking (primary)
            if self.db_manager:
                self.logger.debug(f"Checking database for analysis hash: {analysis_hash[:16]}...")
                if self.db_manager.file_already_analyzed(analysis_hash):
                    existing_result = self.db_manager.get_existing_analysis(analysis_hash)
                    if existing_result:
                        self.logger.info(f"✅ Found existing analysis in database for {pdf_path.name} with {model}, skipping")
                        return existing_result
                else:
                    self.logger.debug(f"No existing analysis found in database for {pdf_path.name}")
            
            # JSON fallback checking (only if database is disabled)
            else:
                existing_result = self._find_existing_analysis(analysis_hash)
                if existing_result:
                    self.logger.info(f"Found existing analysis in JSON for {pdf_path.name} with {model}, skipping")
                    return existing_result
        
        # Extract text and metadata for new analysis
        pdf_data = self.extract_text_from_pdf(pdf_path)
        metadata = self._extract_pdf_metadata(pdf_path)
        
        # Extract multiple topics
        topics_data = self._extract_topics(pdf_data["text_chunks"], model)
        
        # For each topic, extract specific keywords
        topic_keywords_list = []
        for topic_name in topics_data:
            keywords = self._extract_keywords_for_topic(
                pdf_data["text_chunks"], 
                topic_name, 
                model
            )
            topic_keywords = TopicKeywords(
                topic=topic_name,
                keywords=keywords[:self.config.analysis.max_keywords_per_topic],
                confidence_score=0.8
            )
            topic_keywords_list.append(topic_keywords)
        
        # Create result with enhanced metadata
        result = AnalysisResult(
            filename=pdf_path.name,
            file_path=str(pdf_path.absolute()),
            file_hash=analysis_hash,  # Store analysis hash for deduplication
            analysis_model=model,
            topics=topic_keywords_list,
            confidence_score=0.85,
            page_count=pdf_data["page_count"],
            word_count=pdf_data["word_count"],
            author=metadata.get('author'),
            title=metadata.get('title'),
            subject=metadata.get('subject'),
            creation_date=metadata.get('creation_date'),
            modification_date=metadata.get('modification_date'),
            # Backward compatibility
            topic=topics_data[0] if topics_data else "Unknown Topic",
            keywords=[kw for topic in topic_keywords_list for kw in topic.keywords][:self.config.analysis.max_keywords]
        )
        
        # Save to database if enabled
        if self.db_manager:
            try:
                analysis_id = self.db_manager.save_analysis_result(result, metadata, file_hash)
                self.logger.info(f"Saved analysis to database with ID: {analysis_id}")
            except Exception as e:
                self.logger.error(f"Failed to save to database: {e}")
        
        # Always save JSON as backup if configured
        if self.config.database.backup_json:
            self._save_result(result)
        
        return result
    
    def _extract_topics(self, text_chunks: List[str], model: str) -> List[str]:
        combined_text = " ".join(text_chunks[:5])[:3000]
        max_topics = self.config.analysis.max_topics
        
        prompt = f"""Analyze this document text and identify the {max_topics} DISTINCT main topics discussed. 
Each topic must be unique and non-overlapping. Avoid sub-topics when main topics exist.
Return ONLY the topic names, one per line, with 2-4 words per topic maximum.
Do not include explanations, numbering, or bullet points - just the topic names.
Do not repeat or rephrase topics.

Text: {combined_text}

Topics:"""
        
        try:
            response = self.ollama_client.generate_response(prompt, model)
            topics = []
            
            self.logger.debug(f"Raw topic response: {response}")
            
            for line in response.strip().split('\n'):
                line = line.strip()
                if line:
                    # Clean up prefixes: numbers, bullets, dashes, asterisks
                    cleaned_line = re.sub(r'^[\d\.\-\*•\s\)\]]+', '', line).strip()
                    # Remove any trailing punctuation
                    cleaned_line = re.sub(r'[:\.,;]+$', '', cleaned_line).strip()
                    
                    if cleaned_line and len(cleaned_line) > 1:  # Ensure meaningful content
                        topics.append(cleaned_line[:50])
                        self.logger.debug(f"Extracted topic: '{cleaned_line}'")
            
            # Remove duplicate topics before limiting count
            unique_topics = []
            seen_normalized = set()
            
            # Sort by length to prioritize shorter (more general) topics
            sorted_topics = sorted(topics, key=len)
            
            for topic in sorted_topics:
                # Normalize for comparison: lowercase, remove extra spaces, strip punctuation
                normalized = re.sub(r'[^\w\s]', '', topic.lower().strip())
                normalized = ' '.join(normalized.split())  # Remove extra whitespace
                
                # Skip invalid topics
                if (not normalized or 
                    len(normalized) < 3 or 
                    topic.lower().startswith(('here are', 'the following', 'these are', 'topics:', 'here is')) or
                    topic.lower() in ['introduction', 'acknowledgment', 'conclusion', 'summary', 'abstract', 'contents']):
                    self.logger.debug(f"Skipping invalid topic: '{topic}'")
                    continue
                    
                # Check for exact matches
                if normalized in seen_normalized:
                    self.logger.debug(f"Skipping duplicate topic: '{topic}'")
                    continue
                
                # Check for semantic similarity with existing topics
                is_similar = False
                for existing_topic in unique_topics:
                    # Normalize both topics for comparison
                    norm1 = set(re.sub(r'[^\w\s]', '', topic.lower()).split())
                    norm2 = set(re.sub(r'[^\w\s]', '', existing_topic.lower()).split())
                    
                    if len(norm1) > 0 and len(norm2) > 0:
                        # Calculate word overlap
                        common_words = norm1.intersection(norm2)
                        total_words = norm1.union(norm2)
                        overlap_ratio = len(common_words) / len(total_words) if total_words else 0
                        
                        # Check if topics are similar (high overlap or substring)
                        topic_clean = topic.lower().strip()
                        existing_clean = existing_topic.lower().strip()
                        
                        if (overlap_ratio >= 0.6 or 
                            topic_clean in existing_clean or 
                            existing_clean in topic_clean):
                            self.logger.debug(f"Skipping similar topic: '{topic}' (similar to '{existing_topic}')")
                            is_similar = True
                            break
                
                if not is_similar:
                    unique_topics.append(topic)
                    seen_normalized.add(normalized)
            
            topics = unique_topics
            self.logger.info(f"After deduplication: {len(topics)} unique topics")
            
            # Ensure we have at least one topic
            if not topics:
                self.logger.warning("No topics extracted, using fallback")
                topics = ["General Topic"]
            
            self.logger.info(f"Final topics: {topics}")
            return topics[:max_topics]
            
        except Exception as e:
            self.logger.error(f"Failed to extract topics: {e}")
            return ["Unknown Topic"]
    
    def _extract_keywords_for_topic(self, text_chunks: List[str], topic: str, model: str) -> List[str]:
        combined_text = " ".join(text_chunks[:5])[:3000]
        max_keywords = self.config.analysis.max_keywords_per_topic
        
        prompt = f"""Extract {max_keywords} important keywords specifically related to the topic "{topic}" from this text. 
Focus only on keywords that are directly relevant to this topic. Return only the keywords separated by commas, no explanations.

Text: {combined_text}

Keywords for topic "{topic}":"""
        
        try:
            response = self.ollama_client.generate_response(prompt, model)
            keywords_text = response.strip().split('\n')[0]
            # Clean up any explanatory text that might come before keywords
            if ':' in keywords_text:
                keywords_text = keywords_text.split(':', 1)[1]
            keywords = [kw.strip() for kw in keywords_text.split(',') if kw.strip()]
            return keywords[:max_keywords]
        except Exception as e:
            self.logger.error(f"Failed to extract keywords for topic {topic}: {e}")
            return []
    
    def _extract_keywords(self, text_chunks: List[str], model: str) -> List[str]:
        """Legacy method for backward compatibility"""
        combined_text = " ".join(text_chunks[:5])[:3000]
        max_keywords = self.config.analysis.max_keywords
        
        prompt = f"""Extract {max_keywords} important keywords from this text. Return only the keywords separated by commas, no explanations.

Text: {combined_text}

Keywords: """
        
        try:
            response = self.ollama_client.generate_response(prompt, model)
            keywords_text = response.strip().split('\n')[0]
            # Clean up any explanatory text that might come before keywords
            if ':' in keywords_text:
                keywords_text = keywords_text.split(':', 1)[1]
            keywords = [kw.strip() for kw in keywords_text.split(',') if kw.strip()]
            return keywords[:max_keywords]
        except Exception as e:
            self.logger.error(f"Failed to extract keywords: {e}")
            return []
    
    def _save_result(self, result: AnalysisResult):
        if not self.config.output.save_results:
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{result.filename}_{timestamp}.json"
        filepath = Path(self.config.output.results_folder) / filename
        
        with open(filepath, 'w') as f:
            json.dump(result.model_dump(), f, indent=2, default=str)
        
        self.logger.info(f"Results saved to {filepath}")
    
    def scan_pdf_folder(self) -> List[Path]:
        pdf_folder = Path(self.config.pdf.folder_path)
        if not pdf_folder.exists():
            self.logger.warning(f"PDF folder not found: {pdf_folder}")
            return []
        
        pdf_files = []
        for ext in self.config.pdf.extensions:
            pdf_files.extend(pdf_folder.glob(f"*{ext}"))
        
        self.logger.info(f"Found {len(pdf_files)} PDF files in {pdf_folder}")
        return pdf_files
    
    def analyze_all_pdfs(self, force_reanalysis: bool = False) -> List[AnalysisResult]:
        pdf_files = self.scan_pdf_folder()
        results = []
        
        for pdf_file in pdf_files:
            try:
                result = self.analyze_pdf(pdf_file, force_reanalysis=force_reanalysis)
                results.append(result)
                self.logger.info(f"Successfully analyzed {pdf_file.name}")
            except Exception as e:
                self.logger.error(f"Failed to analyze {pdf_file.name}: {e}")
        
        return results