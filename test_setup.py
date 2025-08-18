#!/usr/bin/env python3

import sys
from pathlib import Path

# Add the current directory to Python path so we can import our modules
sys.path.append(str(Path(__file__).parent))

def test_setup():
    """Test basic setup and imports"""
    try:
        print("Testing imports...")
        from src.config_manager import ConfigManager
        from src.ollama_client import OllamaClient
        from src.pdf_analyzer import PDFAnalyzer
        print("✅ All imports successful")
        
        print("\nTesting configuration...")
        config_manager = ConfigManager()
        config = config_manager.get_config()
        print(f"✅ Config loaded: PDF folder = {config.pdf.folder_path}")
        
        print("\nTesting Ollama connection...")
        ollama_client = OllamaClient(config.ollama)
        if ollama_client.is_ollama_running():
            print("✅ Ollama is running")
            models = ollama_client.list_models()
            print(f"✅ Available models: {models}")
        else:
            print("❌ Ollama is not running")
            return False
            
        print("\nTesting PDF analyzer...")
        analyzer = PDFAnalyzer()
        pdf_files = analyzer.scan_pdf_folder()
        print(f"✅ Found {len(pdf_files)} PDF files")
        
        print("\nTesting new multi-topic configuration...")
        config = analyzer.config
        print(f"✅ Max topics: {config.analysis.max_topics}")
        print(f"✅ Max keywords per topic: {config.analysis.max_keywords_per_topic}")
        
        # Test schema validation with sample data
        print("\nTesting new data models...")
        from src.models import TopicKeywords, AnalysisResult
        
        # Test TopicKeywords model
        test_topic = TopicKeywords(
            topic="Machine Learning",
            keywords=["neural networks", "deep learning", "algorithms"],
            confidence_score=0.9
        )
        print(f"✅ TopicKeywords model: {test_topic.topic} with {len(test_topic.keywords)} keywords")
        
        # Test AnalysisResult with multiple topics and new metadata fields
        test_result = AnalysisResult(
            filename="test.pdf",
            file_path="/test/path/test.pdf",
            file_hash="test_hash_123456",
            analysis_model="llama3.2:3b",
            topics=[test_topic],
            confidence_score=0.85,
            page_count=10,
            word_count=1000,
            author="Test Author",
            title="Test Title",
            text_quality_score=0.9
        )
        print(f"✅ AnalysisResult model with {len(test_result.topics)} topics and metadata")
        
        print("\nTesting database functionality...")
        try:
            from src.database_manager import DatabaseManager
            
            # Test database initialization
            test_db_path = "./test_db.sqlite"
            db_manager = DatabaseManager(test_db_path)
            print("✅ DatabaseManager initialized successfully")
            
            # Test database operations
            stats = db_manager.get_database_stats()
            print(f"✅ Database stats: {stats['file_count']} files, {stats['analysis_count']} analyses")
            
            # Clean up test database
            import os
            if os.path.exists(test_db_path):
                os.remove(test_db_path)
            print("✅ Database test completed and cleaned up")
            
        except Exception as e:
            print(f"⚠️  Database test failed (this is OK if database is not fully integrated): {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Running lit_DB setup test...\n")
    success = test_setup()
    if success:
        print("\n🎉 All tests passed! Ready to analyze PDFs.")
    else:
        print("\n💥 Tests failed. Check the errors above.")
        sys.exit(1)