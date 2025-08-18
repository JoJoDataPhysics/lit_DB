#!/usr/bin/env python3

import sys
import json
from pathlib import Path
from typing import List

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

sys.path.append(str(Path(__file__).parent))

from src.pdf_analyzer import PDFAnalyzer
from src.ollama_client import OllamaClient
from src.config_manager import ConfigManager


console = Console()


@click.group()
@click.option('--config', default='config.yaml', help='Path to configuration file')
@click.pass_context
def cli(ctx, config):
    ctx.ensure_object(dict)
    ctx.obj['config_path'] = config


@cli.command()
@click.pass_context
def setup(ctx):
    """Setup and install required Ollama model"""
    config_path = ctx.obj['config_path']
    
    try:
        config_manager = ConfigManager(config_path)
        config = config_manager.get_config()
        ollama_client = OllamaClient(config.ollama)
        
        console.print(Panel("🔧 Setting up lit_DB PDF Analyzer", style="bold blue"))
        
        if not ollama_client.is_ollama_running():
            console.print("[red]❌ Ollama is not running. Please start Ollama first:[/red]")
            console.print("[yellow]   ollama serve[/yellow]")
            sys.exit(1)
        
        console.print("✅ Ollama is running")
        
        model = ollama_client.ensure_model_available()
        console.print(f"✅ Model ready: {model}")
        
        console.print("\n[green]🎉 Setup complete! You can now analyze PDFs.[/green]")
        
    except Exception as e:
        console.print(f"[red]❌ Setup failed: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.pass_context
def check_model(ctx):
    """Check if required models are available"""
    config_path = ctx.obj['config_path']
    
    try:
        config_manager = ConfigManager(config_path)
        config = config_manager.get_config()
        ollama_client = OllamaClient(config.ollama)
        
        console.print(Panel("📋 Model Status Check", style="bold cyan"))
        
        if not ollama_client.is_ollama_running():
            console.print("[red]❌ Ollama is not running[/red]")
            return
        
        available_models = ollama_client.list_models()
        
        table = Table(title="Available Models")
        table.add_column("Model", style="cyan")
        table.add_column("Status", style="green")
        
        all_models = [config.ollama.primary_model] + config.ollama.fallback_models
        
        for model in all_models:
            status = "✅ Available" if any(model in m for m in available_models) else "❌ Not Found"
            table.add_row(model, status)
        
        console.print(table)
        
    except Exception as e:
        console.print(f"[red]❌ Check failed: {e}[/red]")


@cli.command()
@click.argument('filename', required=False)
@click.option('--force', is_flag=True, help='Force re-analysis of already analyzed files')
@click.pass_context
def analyze(ctx, filename, force):
    """Analyze a specific PDF file or all PDFs in the configured folder"""
    config_path = ctx.obj['config_path']
    
    try:
        analyzer = PDFAnalyzer(config_path)
        
        console.print(Panel("📖 PDF Analysis Starting", style="bold green"))
        
        if filename:
            pdf_path = Path(analyzer.config.pdf.folder_path) / filename
            if not pdf_path.exists():
                console.print(f"[red]❌ File not found: {pdf_path}[/red]")
                sys.exit(1)
            
            result = analyzer.analyze_pdf(pdf_path, force_reanalysis=force)
            _display_result(result)
        else:
            results = analyzer.analyze_all_pdfs(force_reanalysis=force)
            if not results:
                console.print("[yellow]⚠️  No PDF files found in the configured folder[/yellow]")
                return
            
            for result in results:
                _display_result(result)
                console.print()
        
        console.print("[green]🎉 Analysis complete![/green]")
        
    except Exception as e:
        console.print(f"[red]❌ Analysis failed: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.pass_context
def list_pdfs(ctx):
    """List all PDF files in the configured folder"""
    config_path = ctx.obj['config_path']
    
    try:
        analyzer = PDFAnalyzer(config_path)
        pdf_files = analyzer.scan_pdf_folder()
        
        if not pdf_files:
            console.print("[yellow]⚠️  No PDF files found[/yellow]")
            return
        
        table = Table(title=f"PDF Files in {analyzer.config.pdf.folder_path}")
        table.add_column("Filename", style="cyan")
        table.add_column("Size", style="green")
        
        for pdf_file in pdf_files:
            size = f"{pdf_file.stat().st_size / 1024:.1f} KB"
            table.add_row(pdf_file.name, size)
        
        console.print(table)
        
    except Exception as e:
        console.print(f"[red]❌ Failed to list PDFs: {e}[/red]")


@cli.command()
@click.argument('model_name')
@click.pass_context
def install_model(ctx, model_name):
    """Install a specific Ollama model"""
    config_path = ctx.obj['config_path']
    
    try:
        config_manager = ConfigManager(config_path)
        config = config_manager.get_config()
        ollama_client = OllamaClient(config.ollama)
        
        console.print(f"Installing model: {model_name}")
        success = ollama_client.install_model(model_name)
        
        if success:
            console.print(f"[green]✅ Successfully installed {model_name}[/green]")
        else:
            console.print(f"[red]❌ Failed to install {model_name}[/red]")
            sys.exit(1)
            
    except Exception as e:
        console.print(f"[red]❌ Installation failed: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.pass_context
def db_status(ctx):
    """Show database statistics"""
    config_path = ctx.obj['config_path']
    
    try:
        config_manager = ConfigManager(config_path)
        config = config_manager.get_config()
        
        if not config.database.enable_persistence:
            console.print("[yellow]⚠️  Database persistence is disabled[/yellow]")
            return
        
        from src.database_manager import DatabaseManager
        db_manager = DatabaseManager(config.database.path)
        stats = db_manager.get_database_stats()
        
        console.print(Panel("📊 Database Statistics", style="bold cyan"))
        
        table = Table()
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Files Analyzed", str(stats['file_count']))
        table.add_row("Total Analyses", str(stats['analysis_count']))
        table.add_row("Topics Extracted", str(stats['topic_count']))
        table.add_row("Models Used", ", ".join(stats['models_used']) if stats['models_used'] else "None")
        table.add_row("Database Path", stats['database_path'])
        table.add_row("Database Size", f"{stats['database_size_mb']:.2f} MB")
        
        console.print(table)
        
    except Exception as e:
        console.print(f"[red]❌ Failed to get database status: {e}[/red]")


@cli.command()
@click.option('--limit', default=20, help='Number of files to show')
@click.pass_context
def list_analyzed(ctx, limit):
    """List previously analyzed files"""
    config_path = ctx.obj['config_path']
    
    try:
        config_manager = ConfigManager(config_path)
        config = config_manager.get_config()
        
        if not config.database.enable_persistence:
            console.print("[yellow]⚠️  Database persistence is disabled[/yellow]")
            return
        
        from src.database_manager import DatabaseManager
        db_manager = DatabaseManager(config.database.path)
        files = db_manager.get_analyzed_files(limit)
        
        if not files:
            console.print("[yellow]⚠️  No analyzed files found[/yellow]")
            return
        
        table = Table(title=f"Recently Analyzed Files (Top {limit})")
        table.add_column("Filename", style="cyan")
        table.add_column("Author", style="green")
        table.add_column("Title", style="blue")
        table.add_column("Pages", style="yellow")
        table.add_column("Words", style="magenta")
        table.add_column("Analyses", style="red")
        table.add_column("Last Analyzed", style="dim")
        
        for file_info in files:
            table.add_row(
                file_info['filename'],
                file_info['author'] or 'N/A',
                file_info['title'] or 'N/A',
                str(file_info['page_count']) if file_info['page_count'] else 'N/A',
                str(file_info['word_count']) if file_info['word_count'] else 'N/A',
                str(file_info['analysis_count']),
                file_info['last_analyzed'][:19] if file_info['last_analyzed'] else 'N/A'
            )
        
        console.print(table)
        
    except Exception as e:
        console.print(f"[red]❌ Failed to list analyzed files: {e}[/red]")


@cli.command()
@click.confirmation_option(prompt='Are you sure you want to drop the database?')
@click.pass_context
def drop_db(ctx):
    """Drop (reset) the entire database - WARNING: This will delete all data!"""
    config_path = ctx.obj['config_path']
    
    try:
        config_manager = ConfigManager(config_path)
        config = config_manager.get_config()
        
        if not config.database.enable_persistence:
            console.print("[yellow]⚠️  Database persistence is disabled[/yellow]")
            return
        
        from src.database_manager import DatabaseManager
        db_manager = DatabaseManager(config.database.path)
        db_manager.reset_database()
        
        console.print("[green]✅ Database has been reset successfully[/green]")
        
    except Exception as e:
        console.print(f"[red]❌ Failed to reset database: {e}[/red]")
        sys.exit(1)


def _display_result(result):
    """Display analysis result with multiple topics and keywords"""
    
    # File info header
    header_content = f"""📄 **File:** {result.filename}
📊 **Pages:** {result.page_count} | **Words:** {result.word_count:,}
🤖 **Model:** {result.analysis_model}
🔒 **Hash:** {result.file_hash[:16]}...
🕒 **Analyzed:** {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}"""
    
    console.print(Panel(header_content, title="Analysis Result", style="bold blue"))
    
    # Display each topic with its keywords
    if result.topics:
        for i, topic_data in enumerate(result.topics, 1):
            topic_content = f"""📚 **Topic {i}:** {topic_data.topic}
⭐ **Confidence:** {topic_data.confidence_score:.2f}

🔑 **Keywords:**
{', '.join(topic_data.keywords) if topic_data.keywords else 'No keywords found'}"""
            
            console.print(Panel(topic_content, style="cyan", padding=(0, 1)))
    else:
        # Fallback to legacy display if no topics found
        fallback_content = f"""📚 **Topic:** {result.topic or 'Unknown'}

🔑 **Keywords:**
{', '.join(result.keywords) if result.keywords else 'No keywords found'}"""
        
        console.print(Panel(fallback_content, style="yellow", padding=(0, 1)))


if __name__ == '__main__':
    cli()