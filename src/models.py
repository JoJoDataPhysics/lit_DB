from typing import List, Optional
from pydantic import BaseModel, Field
from datetime import datetime


class AnalysisResult(BaseModel):
    filename: str
    topic: str
    keywords: List[str] = Field(default_factory=list)
    confidence_score: float = Field(ge=0.0, le=1.0)
    timestamp: datetime = Field(default_factory=datetime.now)
    page_count: int
    word_count: int


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
    max_keywords: int = 10
    chunk_size: int = 4000
    context_overlap: int = 200


class OutputConfig(BaseModel):
    format: str = "json"
    save_results: bool = True
    results_folder: str = "./results"


class LoggingConfig(BaseModel):
    level: str = "INFO"
    file: str = "./logs/app.log"


class AppConfig(BaseModel):
    pdf: PDFConfig
    ollama: OllamaConfig
    analysis: AnalysisConfig
    output: OutputConfig
    logging: LoggingConfig = LoggingConfig()


class ModelStatus(BaseModel):
    name: str
    available: bool = False
    size: Optional[str] = None
    download_progress: Optional[float] = None