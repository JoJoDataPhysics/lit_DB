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
        
        # Initialize vector database manager
        self.vector_db_manager = None
        try:
            from src.vector_db_manager import VectorDatabaseManager
            self.vector_db_manager = VectorDatabaseManager(self.config)
            self.logger.info("Vector database initialized")
        except ImportError as e:
            self.logger.warning(f"Vector database dependencies not available: {e}")
        except Exception as e:
            self.logger.error(f"Failed to initialize vector database: {e}")
        
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
        all_keywords = []
        
        for topic_name in topics_data:
            keywords = self._extract_keywords_for_topic(
                pdf_data["text_chunks"], 
                topic_name, 
                model
            )
            # Calculate topic-specific confidence (will be updated later with overall score)
            topic_keywords = TopicKeywords(
                topic=topic_name,
                keywords=keywords[:self.config.analysis.max_keywords_per_topic],
                confidence_score=0.8  # Placeholder, will be updated
            )
            topic_keywords_list.append(topic_keywords)
            all_keywords.extend(keywords[:self.config.analysis.max_keywords_per_topic])
        
        # Calculate dynamic confidence based on text and analysis quality
        text_quality = self._assess_text_quality(pdf_data["full_text"])
        dynamic_confidence = self._calculate_dynamic_confidence(
            pdf_data["full_text"], 
            topic_keywords_list, 
            all_keywords
        )
        
        # Update topic confidence scores with overall confidence
        for topic_keywords in topic_keywords_list:
            topic_keywords.confidence_score = dynamic_confidence
        
        # Create result with enhanced metadata and dynamic confidence
        result = AnalysisResult(
            filename=pdf_path.name,
            file_path=str(pdf_path.absolute()),  
            file_hash=analysis_hash,  # Store analysis hash for deduplication
            analysis_model=model,
            topics=topic_keywords_list,
            confidence_score=dynamic_confidence,  # Use calculated confidence
            page_count=pdf_data["page_count"],
            word_count=pdf_data["word_count"],
            author=metadata.get('author'),
            title=metadata.get('title'),
            subject=metadata.get('subject'),
            creation_date=metadata.get('creation_date'),
            modification_date=metadata.get('modification_date'),
            text_quality_score=text_quality,  # Include text quality assessment
            # Backward compatibility
            topic=topics_data[0] if topics_data else "Unknown Topic",
            keywords=all_keywords[:self.config.analysis.max_keywords]
        )
        
        # Save to database if enabled
        if self.db_manager:
            try:
                analysis_id = self.db_manager.save_analysis_result(result, metadata, file_hash)
                self.logger.info(f"Saved analysis to database with ID: {analysis_id}")
            except Exception as e:
                self.logger.error(f"Failed to save to database: {e}")
        
        # Save to vector database if enabled
        if self.vector_db_manager:
            try:
                success = self.vector_db_manager.add_document(result, pdf_data["full_text"])
                if success:
                    self.logger.info(f"Saved document embeddings to vector database: {result.filename}")
                else:
                    self.logger.warning(f"Failed to save embeddings for: {result.filename}")
            except Exception as e:
                self.logger.error(f"Failed to save to vector database: {e}")
        
        # Always save JSON as backup if configured
        if self.config.database.backup_json:
            self._save_result(result)
        
        return result
    
    def _extract_topics(self, text_chunks: List[str], model: str) -> List[str]:
        combined_text = " ".join(text_chunks[:5])[:3000]
        max_topics = self.config.analysis.max_topics
        
        # Enhanced prompt with chain-of-thought reasoning
        prompt = f"""Analyze this document and extract the main topics. Think step by step:

1. First, identify the document type (academic paper, technical manual, general text)
2. Look for key themes, subject areas, and main concepts discussed
3. Focus on specific, meaningful topics (avoid generic terms like "introduction" or "overview")
4. Extract {max_topics} distinct, non-overlapping topics
5. Each topic should be 2-4 words, specific and descriptive

Example good topics: "Machine Learning Algorithms", "Climate Change Impact", "Software Architecture"
Example bad topics: "Introduction", "Overview", "General Discussion"

Text to analyze:
{combined_text}

Analysis:
Document type: [Identify the type first]
Main themes: [List key themes you observe]

Final topics (exactly {max_topics}, one per line):"""
        
        try:
            response = self.ollama_client.generate_response(prompt, model)
            topics = []
            
            self.logger.debug(f"Raw topic response: {response}")
            
            # Parse structured response - look for "Final topics" section or fall back to all lines
            lines = response.strip().split('\n')
            final_topics_section = False
            found_final_section = False
            
            for line in lines:
                line = line.strip()
                
                # Check if we found the "Final topics" section
                if "final topics" in line.lower():
                    final_topics_section = True
                    found_final_section = True
                    continue
                
                # Skip analysis sections when in structured mode
                if found_final_section and line.lower().startswith(('document type:', 'main themes:', 'analysis:')):
                    continue
                    
                # Process topic lines (either in final section or fallback to all lines)
                if (final_topics_section or not found_final_section) and line:
                    # Clean up prefixes: numbers, bullets, dashes, asterisks
                    cleaned_line = re.sub(r'^[\d\.\-\*•\s\)\]]+', '', line).strip()
                    # Remove any trailing punctuation
                    cleaned_line = re.sub(r'[:\.,;]+$', '', cleaned_line).strip()
                    
                    # Skip obvious non-topic lines
                    if (cleaned_line and len(cleaned_line) > 1 and 
                        not line.lower().startswith(('document type:', 'main themes:', 'analysis:', 'final topics'))):
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
        
        # Enhanced keyword extraction prompt with examples
        prompt = f"""Extract the most relevant keywords for the topic "{topic}" from this text. Follow these steps:

1. Scan the text for terms directly related to "{topic}"
2. Look for technical terms, proper nouns, and specific concepts
3. Prioritize terms that appear multiple times or in important contexts
4. Avoid generic words like "the", "and", "important", "study"
5. Extract exactly {max_keywords} keywords

Good keywords examples:
- For "Machine Learning": "neural networks", "algorithms", "training data", "classification"
- For "Climate Change": "greenhouse gases", "global warming", "carbon emissions", "sea level"

Text to analyze:
{combined_text}

Extract {max_keywords} keywords for topic "{topic}" (comma-separated, no explanations):"""
        
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
    
    def _assess_text_quality(self, extracted_text: str) -> float:
        """Assess the quality of extracted text from PDF."""
        if not extracted_text or len(extracted_text.strip()) == 0:
            return 0.0
        
        # Metrics for text quality assessment
        metrics = {}
        
        # 1. Length check - ensure substantial content
        text_length = len(extracted_text.strip())
        metrics['length_score'] = min(1.0, text_length / 500)  # Normalize to 500 chars
        
        # 2. Word-to-character ratio - detect garbled text
        words = extracted_text.split()
        if text_length > 0:
            word_char_ratio = len(' '.join(words)) / text_length
            metrics['word_ratio'] = min(1.0, word_char_ratio * 1.2)  # Good ratio ~0.8
        else:
            metrics['word_ratio'] = 0.0
        
        # 3. Sentence structure detection
        sentence_endings = extracted_text.count('.') + extracted_text.count('!') + extracted_text.count('?')
        if len(words) > 0:
            sentences_per_100_words = (sentence_endings / len(words)) * 100
            # Ideal: 5-15 sentences per 100 words
            metrics['sentence_structure'] = 1.0 - abs(sentences_per_100_words - 10) / 20
            metrics['sentence_structure'] = max(0.0, min(1.0, metrics['sentence_structure']))
        else:
            metrics['sentence_structure'] = 0.0
        
        # 4. Special character ratio - too many might indicate OCR issues
        special_chars = sum(1 for c in extracted_text if not c.isalnum() and not c.isspace())
        if text_length > 0:
            special_ratio = special_chars / text_length
            # Good range: 5-15% special characters
            metrics['special_char_ratio'] = 1.0 - abs(special_ratio - 0.10) / 0.10
            metrics['special_char_ratio'] = max(0.0, min(1.0, metrics['special_char_ratio']))
        else:
            metrics['special_char_ratio'] = 0.0
        
        # 5. Repetitive pattern detection (OCR artifacts)
        lines = extracted_text.split('\n')
        unique_lines = set(line.strip() for line in lines if line.strip())
        if len(lines) > 0:
            uniqueness_ratio = len(unique_lines) / len(lines)
            metrics['uniqueness'] = uniqueness_ratio
        else:
            metrics['uniqueness'] = 1.0
        
        # Weighted average of metrics
        weights = {
            'length_score': 0.2,
            'word_ratio': 0.25,
            'sentence_structure': 0.25,
            'special_char_ratio': 0.15,
            'uniqueness': 0.15
        }
        
        quality_score = sum(metrics[key] * weights[key] for key in metrics)
        
        self.logger.debug(f"Text quality metrics: {metrics}")
        self.logger.debug(f"Overall text quality score: {quality_score:.3f}")
        
        return quality_score
    
    def _score_topic_specificity(self, topics: List[str]) -> float:
        """Score how specific and meaningful the extracted topics are."""
        if not topics:
            return 0.0
        
        specificity_scores = []
        generic_terms = {
            'introduction', 'overview', 'summary', 'conclusion', 'abstract',
            'general', 'basic', 'important', 'main', 'key', 'discussion',
            'analysis', 'study', 'research', 'paper', 'document', 'text'
        }
        
        for topic in topics:
            topic_lower = topic.lower().strip()
            
            # Penalize very short topics
            length_score = min(1.0, len(topic.split()) / 3)  # 3+ words is ideal
            
            # Penalize generic terms
            generic_penalty = 0.0
            topic_words = set(topic_lower.split())
            generic_overlap = len(topic_words.intersection(generic_terms))
            if generic_overlap > 0:
                generic_penalty = generic_overlap / len(topic_words)
            
            # Reward specific, technical, or proper noun terms
            specificity_bonus = 0.0
            if any(word[0].isupper() for word in topic.split()):  # Proper nouns
                specificity_bonus += 0.2
            if len(topic_words) >= 2:  # Multi-word topics are more specific
                specificity_bonus += 0.3
            
            topic_score = length_score - generic_penalty + specificity_bonus
            topic_score = max(0.0, min(1.0, topic_score))
            specificity_scores.append(topic_score)
        
        return sum(specificity_scores) / len(specificity_scores)
    
    def _score_keyword_relevance(self, keywords: List[str], text: str) -> float:
        """Score how relevant keywords are to the actual text content."""
        if not keywords or not text:
            return 0.0
        
        text_lower = text.lower()
        text_words = set(text_lower.split())
        
        relevance_scores = []
        
        for keyword in keywords:
            keyword_lower = keyword.lower().strip()
            
            # Direct presence in text
            direct_presence = 1.0 if keyword_lower in text_lower else 0.0
            
            # Word-level presence (for multi-word keywords)
            keyword_words = keyword_lower.split()
            word_presence = sum(1 for word in keyword_words if word in text_words) / len(keyword_words)
            
            # Frequency bonus
            frequency = text_lower.count(keyword_lower)
            frequency_score = min(1.0, frequency / 3)  # Cap at 3 mentions
            
            # Length penalty for very short keywords (less meaningful)
            length_penalty = 0.5 if len(keyword_lower) < 4 else 0.0
            
            keyword_score = max(direct_presence, word_presence) + frequency_score - length_penalty
            keyword_score = max(0.0, min(1.0, keyword_score))
            relevance_scores.append(keyword_score)
        
        return sum(relevance_scores) / len(relevance_scores)
    
    def _assess_response_quality(self, topics: List[str], keywords: List[str]) -> float:
        """Assess the overall quality of LLM response."""
        metrics = {}
        
        # Topic count - should have reasonable number of topics
        expected_topics = self.config.analysis.max_topics
        topic_count_score = min(1.0, len(topics) / expected_topics) if topics else 0.0
        metrics['topic_count'] = topic_count_score
        
        # Keyword count - should have reasonable number of keywords per topic
        total_expected_keywords = expected_topics * self.config.analysis.max_keywords_per_topic
        keyword_count_score = min(1.0, len(keywords) / total_expected_keywords) if keywords else 0.0
        metrics['keyword_count'] = keyword_count_score
        
        # Response completeness
        has_topics = len(topics) > 0
        has_keywords = len(keywords) > 0
        completeness = (has_topics + has_keywords) / 2
        metrics['completeness'] = completeness
        
        # Weighted average
        weights = {'topic_count': 0.4, 'keyword_count': 0.4, 'completeness': 0.2}
        return sum(metrics[key] * weights[key] for key in metrics)
    
    def _calculate_dynamic_confidence(self, text: str, topics: List[TopicKeywords], all_keywords: List[str]) -> float:
        """Calculate dynamic confidence score based on multiple quality factors."""
        
        # Individual quality assessments
        text_quality = self._assess_text_quality(text)
        topic_specificity = self._score_topic_specificity([t.topic for t in topics])
        keyword_relevance = self._score_keyword_relevance(all_keywords, text)
        response_quality = self._assess_response_quality([t.topic for t in topics], all_keywords)
        
        # Weighted combination of quality factors
        weights = {
            'text_quality': 0.3,      # How well was the text extracted
            'topic_specificity': 0.3, # How specific and meaningful are topics
            'keyword_relevance': 0.25, # How relevant are keywords to content
            'response_quality': 0.15   # How complete was the LLM response
        }
        
        quality_scores = {
            'text_quality': text_quality,
            'topic_specificity': topic_specificity,
            'keyword_relevance': keyword_relevance,
            'response_quality': response_quality
        }
        
        # Calculate weighted confidence
        dynamic_confidence = sum(quality_scores[key] * weights[key] for key in weights)
        
        # Ensure confidence is in valid range
        dynamic_confidence = max(0.1, min(0.99, dynamic_confidence))
        
        self.logger.info(f"Dynamic confidence calculation:")
        self.logger.info(f"  Text quality: {text_quality:.3f}")
        self.logger.info(f"  Topic specificity: {topic_specificity:.3f}")
        self.logger.info(f"  Keyword relevance: {keyword_relevance:.3f}")
        self.logger.info(f"  Response quality: {response_quality:.3f}")
        self.logger.info(f"  Final confidence: {dynamic_confidence:.3f}")
        
        return dynamic_confidence