# Agent Guidelines for lit_DB

## Build/Test Commands
- Run app: `python main.py [command]` - Click-based CLI with setup, analyze, list-pdfs commands
- Test setup: `python test_setup.py` - Basic functionality and Ollama connectivity test
- Install deps: `pip install -r requirements.txt` - PyPDF2, pdfplumber, pydantic, click, rich
- No formal test suite - use test_setup.py to verify configuration and Ollama connection

## Code Style Guidelines (Python-focused)
- **Imports**: stdlib, third-party, local imports grouped with blank lines (see pdf_analyzer.py:1-14)
- **Naming**: snake_case for functions/variables, PascalCase for classes (PDFAnalyzer, OllamaClient)
- **Types**: Pydantic models for data validation, type hints throughout (see models.py)
- **Error Handling**: Try/except with logging, return meaningful error messages to user
- **Functions**: Private methods prefixed with underscore, descriptive names
- **Logging**: Use self.logger with appropriate levels (INFO, ERROR, WARNING)
- **CLI**: Rich console for formatted output, Click for command structure
- **Config**: YAML-based config with Pydantic validation (config.yaml + models.py)

## Project Structure
- `src/`: Core modules (pdf_analyzer, ollama_client, config_manager, models)  
- `main.py`: Click CLI entry point with Rich formatting
- `config.yaml`: Application configuration
- `requirements.txt`: Python dependencies (no pyproject.toml)
- Follow existing patterns: Pydantic models, logging setup, path handling with pathlib