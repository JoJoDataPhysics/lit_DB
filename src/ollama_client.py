import requests
import json
import logging
import subprocess
import time
from typing import List, Optional, Dict, Any
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.console import Console

from src.models import ModelStatus, OllamaConfig


class OllamaClient:
    def __init__(self, config: OllamaConfig):
        self.config = config
        self.console = Console()
        self.logger = logging.getLogger(__name__)
        
    def is_ollama_running(self) -> bool:
        try:
            response = requests.get(f"{self.config.url}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False
    
    def list_models(self) -> List[str]:
        try:
            response = requests.get(f"{self.config.url}/api/tags")
            response.raise_for_status()
            data = response.json()
            return [model['name'] for model in data.get('models', [])]
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to list models: {e}")
            return []
    
    def check_model_availability(self, model_name: str) -> ModelStatus:
        available_models = self.list_models()
        is_available = any(model_name in model for model in available_models)
        
        return ModelStatus(
            name=model_name,
            available=is_available
        )
    
    def install_model(self, model_name: str) -> bool:
        if not self.is_ollama_running():
            self.console.print("[red]Ollama is not running. Please start Ollama first.[/red]")
            return False
            
        self.console.print(f"[yellow]Installing model: {model_name}[/yellow]")
        
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console
            ) as progress:
                task = progress.add_task(f"Downloading {model_name}...", total=None)
                
                response = requests.post(
                    f"{self.config.url}/api/pull",
                    json={"name": model_name},
                    stream=True,
                    timeout=1800
                )
                
                for line in response.iter_lines():
                    if line:
                        data = json.loads(line.decode('utf-8'))
                        if 'status' in data:
                            progress.update(task, description=f"{model_name}: {data['status']}")
                        if data.get('status') == 'success':
                            break
                            
            self.console.print(f"[green]Successfully installed {model_name}[/green]")
            return True
            
        except Exception as e:
            self.console.print(f"[red]Failed to install {model_name}: {e}[/red]")
            return False
    
    def ensure_model_available(self) -> str:
        models_to_try = [self.config.primary_model] + self.config.fallback_models
        
        for model in models_to_try:
            status = self.check_model_availability(model)
            if status.available:
                self.console.print(f"[green]Using model: {model}[/green]")
                return model
                
            if self.config.auto_install:
                self.console.print(f"[yellow]Model {model} not found. Attempting to install...[/yellow]")
                if self.install_model(model):
                    return model
        
        raise RuntimeError("No suitable model available and installation failed")
    
    def generate_response(self, prompt: str, model: str) -> str:
        try:
            response = requests.post(
                f"{self.config.url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "top_p": 0.9
                    }
                },
                timeout=120
            )
            response.raise_for_status()
            return response.json()['response']
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to generate response: {e}")
            raise