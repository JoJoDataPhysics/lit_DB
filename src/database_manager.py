import sqlite3
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from src.models import AnalysisResult, TopicKeywords


class DatabaseManager:
    """SQLite database manager for PDF analysis results and file metadata."""
    
    def __init__(self, db_path: str = "./data/lit_db.sqlite"):
        self.db_path = Path(db_path)
        self.logger = logging.getLogger(__name__)
        
        # Ensure database directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database schema
        self._init_database()
    
    def _init_database(self):
        """Initialize database schema if it doesn't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS files (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_path TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    file_hash TEXT NOT NULL,
                    file_size INTEGER,
                    page_count INTEGER,
                    word_count INTEGER,
                    author TEXT,
                    title TEXT,
                    subject TEXT,
                    creator TEXT,
                    producer TEXT,
                    creation_date TEXT,
                    modification_date TEXT,
                    first_analyzed DATETIME DEFAULT CURRENT_TIMESTAMP,
                    last_analyzed DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(file_path, file_hash)
                );
                
                CREATE TABLE IF NOT EXISTS analysis_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_id INTEGER,
                    analysis_hash TEXT NOT NULL UNIQUE,
                    model_name TEXT NOT NULL,
                    confidence_score REAL,
                    analysis_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (file_id) REFERENCES files (id)
                );
                
                CREATE TABLE IF NOT EXISTS topics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    analysis_id INTEGER,
                    topic TEXT NOT NULL,
                    confidence_score REAL,
                    keywords TEXT,
                    FOREIGN KEY (analysis_id) REFERENCES analysis_results (id)
                );
                
                CREATE INDEX IF NOT EXISTS idx_files_hash ON files (file_hash);
                CREATE INDEX IF NOT EXISTS idx_analysis_hash ON analysis_results (analysis_hash);
                CREATE INDEX IF NOT EXISTS idx_files_path ON files (file_path);
            """)
    
    def file_already_analyzed(self, analysis_hash: str) -> bool:
        """Check if a file with the given analysis hash has already been analyzed."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT 1 FROM analysis_results WHERE analysis_hash = ? LIMIT 1",
                (analysis_hash,)
            )
            return cursor.fetchone() is not None
    
    def get_existing_analysis(self, analysis_hash: str) -> Optional[AnalysisResult]:
        """Retrieve existing analysis result by analysis hash."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            # Get analysis result with file info
            cursor = conn.execute("""
                SELECT ar.*, f.filename, f.file_path, f.page_count, f.word_count,
                       f.author, f.title, f.subject, f.creation_date, f.modification_date
                FROM analysis_results ar
                JOIN files f ON ar.file_id = f.id
                WHERE ar.analysis_hash = ?
            """, (analysis_hash,))
            
            row = cursor.fetchone()
            if not row:
                return None
            
            # Get topics for this analysis
            topics_cursor = conn.execute(
                "SELECT topic, confidence_score, keywords FROM topics WHERE analysis_id = ?",
                (row['id'],)
            )
            
            topics = []
            for topic_row in topics_cursor.fetchall():
                keywords = json.loads(topic_row['keywords']) if topic_row['keywords'] else []
                topics.append(TopicKeywords(
                    topic=topic_row['topic'],
                    keywords=keywords,
                    confidence_score=topic_row['confidence_score']
                ))
            
            return AnalysisResult(
                filename=row['filename'],
                file_path=row['file_path'],
                file_hash=analysis_hash.split('_')[0],  # Extract file hash from analysis hash
                analysis_model=row['model_name'],
                topics=topics,
                confidence_score=row['confidence_score'],
                timestamp=datetime.fromisoformat(row['analysis_timestamp']),
                page_count=row['page_count'],
                word_count=row['word_count'],
                author=row['author'],
                title=row['title'],
                subject=row['subject'],
                creation_date=row['creation_date'],
                modification_date=row['modification_date']
            )
    
    def save_analysis_result(self, result: AnalysisResult, file_metadata: Dict[str, Any]) -> int:
        """Save analysis result to database."""
        with sqlite3.connect(self.db_path) as conn:
            # Insert or update file record
            file_id = self._upsert_file(conn, result, file_metadata)
            
            # Create analysis hash
            analysis_hash = self._create_analysis_hash(result.file_hash, result.analysis_model)
            
            # Insert analysis result
            cursor = conn.execute("""
                INSERT INTO analysis_results (file_id, analysis_hash, model_name, confidence_score, analysis_timestamp)
                VALUES (?, ?, ?, ?, ?)
            """, (
                file_id,
                analysis_hash,
                result.analysis_model,
                result.confidence_score,
                result.timestamp.isoformat()
            ))
            
            analysis_id = cursor.lastrowid
            
            # Insert topics
            for topic_data in result.topics:
                conn.execute("""
                    INSERT INTO topics (analysis_id, topic, confidence_score, keywords)
                    VALUES (?, ?, ?, ?)
                """, (
                    analysis_id,
                    topic_data.topic,
                    topic_data.confidence_score,
                    json.dumps(topic_data.keywords)
                ))
            
            if analysis_id is None:
                raise ValueError("Failed to insert analysis result")
            return analysis_id
    
    def _upsert_file(self, conn: sqlite3.Connection, result: AnalysisResult, metadata: Dict[str, Any]) -> int:
        """Insert or update file record, return file_id."""
        # Check if file already exists
        cursor = conn.execute(
            "SELECT id FROM files WHERE file_path = ? AND file_hash = ?",
            (result.file_path, result.file_hash)
        )
        
        existing = cursor.fetchone()
        if existing:
            # Update last_analyzed timestamp
            conn.execute(
                "UPDATE files SET last_analyzed = CURRENT_TIMESTAMP WHERE id = ?",
                (existing[0],)
            )
            return int(existing[0])
        
        # Insert new file record
        cursor = conn.execute("""
            INSERT INTO files (
                file_path, filename, file_hash, file_size, page_count, word_count,
                author, title, subject, creator, producer, creation_date, modification_date
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            result.file_path,
            result.filename,
            result.file_hash,
            metadata.get('file_size'),
            result.page_count,
            result.word_count,
            result.author,
            result.title,
            result.subject,
            metadata.get('creator'),
            metadata.get('producer'),
            result.creation_date,
            result.modification_date
        ))
        
        if cursor.lastrowid is None:
            raise ValueError("Failed to insert file record")
        return cursor.lastrowid
    
    def _create_analysis_hash(self, file_hash: str, model_name: str) -> str:
        """Create analysis hash from file hash and model name."""
        import hashlib
        combined = f"{file_hash}_{model_name}"
        return hashlib.sha256(combined.encode()).hexdigest()
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM files")
            file_count = cursor.fetchone()[0]
            
            cursor = conn.execute("SELECT COUNT(*) FROM analysis_results")
            analysis_count = cursor.fetchone()[0]
            
            cursor = conn.execute("SELECT COUNT(*) FROM topics")
            topic_count = cursor.fetchone()[0]
            
            # Get unique models used
            cursor = conn.execute("SELECT DISTINCT model_name FROM analysis_results")
            models = [row[0] for row in cursor.fetchall()]
            
            return {
                'file_count': file_count,
                'analysis_count': analysis_count,
                'topic_count': topic_count,
                'models_used': models,
                'database_path': str(self.db_path),
                'database_size_mb': self.db_path.stat().st_size / (1024 * 1024) if self.db_path.exists() else 0
            }
    
    def get_analyzed_files(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get list of analyzed files with their analysis info."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            cursor = conn.execute("""
                SELECT f.filename, f.file_path, f.page_count, f.word_count, f.author, f.title,
                       f.first_analyzed, f.last_analyzed,
                       COUNT(ar.id) as analysis_count,
                       GROUP_CONCAT(DISTINCT ar.model_name) as models_used
                FROM files f
                LEFT JOIN analysis_results ar ON f.id = ar.file_id
                GROUP BY f.id
                ORDER BY f.last_analyzed DESC
                LIMIT ?
            """, (limit,))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def reset_database(self):
        """Reset (drop and recreate) the entire database. Use with caution!"""
        if self.db_path.exists():
            self.db_path.unlink()
        self._init_database()
        self.logger.info("Database reset successfully")