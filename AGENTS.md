# Agent Guidelines for lit_DB

## Build/Test Commands
- **Run app**: `python main.py [command]` - Click-based CLI with commands: setup, analyze, list-pdfs, db-status, list-topics
- **Test setup**: `python test_setup.py` - Comprehensive test of imports, config, Ollama connectivity, and data models
- **Install deps**: `pip install -r requirements.txt` - Core deps: PyPDF2, pdfplumber, pydantic, click, rich
- **Single test**: No formal test suite - use `test_setup.py` to verify all functionality and dependencies
- **Database**: SQLite-based with optional persistence (configurable via config.yaml)

## Code Style Guidelines (Python-focused)
- **Imports**: stdlib, third-party, local imports grouped with blank lines (see pdf_analyzer.py:1-14)
- **Naming**: snake_case for functions/variables, PascalCase for classes (PDFAnalyzer, OllamaClient)
- **Types**: Pydantic models for data validation with Field constraints, full type hints (see models.py)
- **Error Handling**: Try/except with Rich console error display, meaningful user messages
- **Functions**: Private methods prefixed with underscore, descriptive names, docstrings for public methods
- **Logging**: Use self.logger with INFO/ERROR/WARNING levels, structured logging patterns
- **CLI**: Rich console for formatted output, Click decorators, confirmation prompts for destructive operations
- **Config**: YAML-based config with Pydantic validation and nested models (AppConfig structure)
- **File Paths**: Use pathlib.Path consistently, absolute paths for file operations

## Project Structure
- `src/`: Core modules (pdf_analyzer, ollama_client, config_manager, database_manager, models)
- `main.py`: Click CLI with @cli.command() decorators and Rich formatting
- `config.yaml`: Nested YAML config (pdf, ollama, analysis, database sections)
- `requirements.txt`: Python dependencies (no pyproject.toml in use)
- Follow patterns: Pydantic validation, console.print() for output, pathlib for file ops

## Development Workflow
- **Testing**: Always run `python test_setup.py` after code changes to verify functionality
- **Commit Rule**: After each major change, create descriptive commits explaining purpose and impact
- **Dependencies**: Check existing imports before adding new libraries, prefer built-in solutions