import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

import pdfplumber
from PyPDF2 import PdfReader

from src.models import AnalysisResult, AppConfig, TopicKeywords
from src.ollama_client import OllamaClient
from src.config_manager import ConfigManager


class PDFAnalyzer:
    def __init__(self, config_path: str = "config.yaml"):
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.get_config()
        self.ollama_client = OllamaClient(self.config.ollama)
        self.logger = logging.getLogger(__name__)
        
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
    
    def analyze_pdf(self, pdf_path: Path) -> AnalysisResult:
        self.logger.info(f"Starting analysis of {pdf_path.name}")
        
        pdf_data = self.extract_text_from_pdf(pdf_path)
        model = self.ollama_client.ensure_model_available()
        
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
        
        # Create result with new multi-topic structure
        result = AnalysisResult(
            filename=pdf_path.name,
            topics=topic_keywords_list,
            confidence_score=0.85,
            page_count=pdf_data["page_count"],
            word_count=pdf_data["word_count"],
            # Backward compatibility
            topic=topics_data[0] if topics_data else "Unknown Topic",
            keywords=[kw for topic in topic_keywords_list for kw in topic.keywords][:self.config.analysis.max_keywords]
        )
        
        self._save_result(result)
        return result
    
    def _extract_topics(self, text_chunks: List[str], model: str) -> List[str]:
        combined_text = " ".join(text_chunks[:5])[:3000]
        max_topics = self.config.analysis.max_topics
        
        prompt = f"""Analyze this document text and identify the {max_topics} main topics discussed. 
Return only the topic names, one per line, with 2-4 words per topic maximum.

Text: {combined_text}

Topics:"""
        
        try:
            response = self.ollama_client.generate_response(prompt, model)
            topics = []
            for line in response.strip().split('\n'):
                line = line.strip()
                # Clean up numbering or bullet points
                if line and not line.startswith(('•', '-', '*')):
                    # Remove numbers like "1.", "2.", etc.
                    if line[0].isdigit() and '.' in line[:3]:
                        line = line.split('.', 1)[1].strip()
                    if line:
                        topics.append(line[:50])
            
            # Ensure we have at least one topic
            if not topics:
                topics = ["General Topic"]
            
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
    
    def analyze_all_pdfs(self) -> List[AnalysisResult]:
        pdf_files = self.scan_pdf_folder()
        results = []
        
        for pdf_file in pdf_files:
            try:
                result = self.analyze_pdf(pdf_file)
                results.append(result)
                self.logger.info(f"Successfully analyzed {pdf_file.name}")
            except Exception as e:
                self.logger.error(f"Failed to analyze {pdf_file.name}: {e}")
        
        return results