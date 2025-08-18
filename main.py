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


def _display_result(result):
    """Display analysis result with multiple topics and keywords"""
    
    # File info header
    header_content = f"""📄 **File:** {result.filename}
📊 **Pages:** {result.page_count} | **Words:** {result.word_count:,}
🔒 **Hash:** {result.file_hash[:12]}...
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