# lit_DB: PDF Ebook Analysis Tool

A Python application that analyzes PDF ebooks using a local Ollama server to detect keywords and topics.

## Features

- 📖 PDF text extraction and analysis
- 🤖 Local LLM analysis via Ollama
- 🔑 Configurable keyword extraction (max number via config)
- 📊 Topic detection and classification
- 📁 Batch processing of PDF folders
- 💾 JSON result output
- 🛠️ Automatic model installation

## Requirements

- Python 3.8+
- Ollama server running locally
- PDF files in the `test_literature/` folder

## Installation

1. **Clone and setup virtual environment:**
   ```bash
   cd lit_DB
   python3 -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   venv\Scripts\activate     # Windows
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Start Ollama server:**
   ```bash
   ollama serve
   ```

4. **Setup and install models:**
   ```bash
   python main.py setup
   ```

## Usage

### Analyze all PDFs in test_literature folder:
```bash
python main.py analyze
```

### Analyze a specific PDF:
```bash
python main.py analyze sample_book.pdf
```

### List available PDFs:
```bash
python main.py list-pdfs
```

### Check model status:  
```bash
python main.py check-model
```

### Install a specific model:
```bash
python main.py install-model llama3.2:3b
```

## Configuration

Edit `config.yaml` to customize:
- PDF folder location
- Maximum keywords per document
- Ollama server settings
- Output format and location

## Project Structure

```
lit_DB/
├── config.yaml          # Configuration file
├── main.py              # CLI interface
├── requirements.txt     # Python dependencies
├── src/                 # Source code
├── test_literature/     # PDF files for analysis
├── results/             # Analysis output
└── logs/               # Application logs
```

## Example Output

```json
{
  "filename": "sample_ml_book.pdf",
  "topic": "Machine Learning Basics",
  "keywords": [
    "machine learning",
    "artificial intelligence", 
    "neural networks",
    "deep learning",
    "data science",
    "computer vision",
    "algorithms"
  ],
  "confidence_score": 0.85,
  "page_count": 1,
  "word_count": 139,
  "timestamp": "2025-08-18 23:12:25"
}
```

## Quick Start

1. **Place PDF files** in the `test_literature/` folder
2. **Start Ollama**: `ollama serve` 
3. **Setup**: `python main.py setup`
4. **Analyze**: `python main.py analyze`

Results are saved as JSON files in the `results/` folder.