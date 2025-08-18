import yaml
import logging
from pathlib import Path
from typing import Dict, Any
from src.models import AppConfig


class ConfigManager:
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = Path(config_path)
        self.config: AppConfig = self._load_config()
        
    def _load_config(self) -> AppConfig:
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
            
        with open(self.config_path, 'r') as file:
            config_data = yaml.safe_load(file)
            
        return AppConfig(**config_data)
    
    def get_config(self) -> AppConfig:
        return self.config
    
    def update_config(self, updates: Dict[str, Any]) -> None:
        config_dict = self.config.model_dump()
        self._deep_update(config_dict, updates)
        self.config = AppConfig(**config_dict)
    
    def save_config(self) -> None:
        with open(self.config_path, 'w') as file:
            yaml.dump(self.config.model_dump(), file, default_flow_style=False)
    
    def _deep_update(self, base_dict: Dict, update_dict: Dict) -> None:
        for key, value in update_dict.items():
            if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value