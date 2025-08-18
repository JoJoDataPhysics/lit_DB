import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

import pdfplumber
from PyPDF2 import PdfReader

from src.models import AnalysisResult, AppConfig
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
        
        topic = self._extract_topic(pdf_data["text_chunks"], model)
        keywords = self._extract_keywords(pdf_data["text_chunks"], model)
        
        result = AnalysisResult(
            filename=pdf_path.name,
            topic=topic,
            keywords=keywords[:self.config.analysis.max_keywords],
            confidence_score=0.85,
            page_count=pdf_data["page_count"],
            word_count=pdf_data["word_count"]
        )
        
        self._save_result(result)
        return result
    
    def _extract_topic(self, text_chunks: List[str], model: str) -> str:
        combined_text = " ".join(text_chunks[:3])[:2000]
        
        prompt = f"""Analyze this ebook text and identify the main topic. Give only the topic name in 2-3 words maximum.

Text: {combined_text}

Topic:"""
        
        try:
            response = self.ollama_client.generate_response(prompt, model)
            topic = response.strip().split('\n')[0][:50]
            return topic
        except Exception as e:
            self.logger.error(f"Failed to extract topic: {e}")
            return "Unknown Topic"
    
    def _extract_keywords(self, text_chunks: List[str], model: str) -> List[str]:
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