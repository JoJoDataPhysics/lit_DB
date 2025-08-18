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