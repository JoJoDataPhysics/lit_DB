from typing import List, Optional
from pydantic import BaseModel, Field
from datetime import datetime


class TopicKeywords(BaseModel):
    topic: str
    keywords: List[str] = Field(default_factory=list)
    confidence_score: float = Field(ge=0.0, le=1.0, default=0.8)


class AnalysisResult(BaseModel):
    filename: str
    file_path: Optional[str] = None  # Full path to the PDF file
    file_hash: str  # SHA-256 hash combining file content + model used
    analysis_model: str  # Model that generated this analysis
    topics: List[TopicKeywords] = Field(default_factory=list)
    confidence_score: float = Field(ge=0.0, le=1.0)
    timestamp: datetime = Field(default_factory=datetime.now)
    page_count: int
    word_count: int
    
    # PDF metadata fields
    author: Optional[str] = None
    title: Optional[str] = None
    subject: Optional[str] = None
    creation_date: Optional[str] = None
    modification_date: Optional[str] = None
    
    # Backward compatibility fields (deprecated)
    topic: Optional[str] = None
    keywords: List[str] = Field(default_factory=list)


class OllamaConfig(BaseModel):
    url: str = "http://localhost:11434"
    primary_model: str = "llama3.2:3b"
    fallback_models: List[str] = ["mistral:7b", "phi3:mini"]
    auto_install: bool = True


class PDFConfig(BaseModel):
    folder_path: str = "./test_literature"
    extensions: List[str] = [".pdf"]
    recursive_scan: bool = False


class AnalysisConfig(BaseModel):
    max_topics: int = 3
    max_keywords_per_topic: int = 8
    max_keywords: int = 10  # Backward compatibility
    chunk_size: int = 4000
    context_overlap: int = 200


class OutputConfig(BaseModel):
    format: str = "json"
    save_results: bool = True
    results_folder: str = "./results"


class LoggingConfig(BaseModel):
    level: str = "INFO"
    file: str = "./logs/app.log"


class DatabaseConfig(BaseModel):
    path: str = "./data/lit_db.sqlite"
    enable_persistence: bool = True
    backup_json: bool = True


class AppConfig(BaseModel):
    pdf: PDFConfig
    ollama: OllamaConfig
    analysis: AnalysisConfig
    output: OutputConfig
    logging: LoggingConfig = LoggingConfig()
    database: DatabaseConfig = DatabaseConfig()


class ModelStatus(BaseModel):
    name: str
    available: bool = False
    size: Optional[str] = None
    download_progress: Optional[float] = None