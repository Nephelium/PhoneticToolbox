
from pathlib import Path
from typing import Dict, Any
import json
from phonetic_toolbox.models.config import AcousticConfig

from dataclasses import asdict

class SettingsService:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SettingsService, cls).__new__(cls)
            # 移除文件依赖，直接使用 AcousticConfig 的默认值
            # settings.json 不再被使用
            cls._instance.config = asdict(AcousticConfig())
            
            # 确保路径一致性
            cls._instance.config["reaper_bin_path"] = AcousticConfig.reaper_bin_path
            
        return cls._instance

    def load(self):
        # 废弃：不再从文件加载
        pass

    def save(self):
        # 废弃：不再保存到文件
        # 如果需要，可以在这里打印一条日志，提示用户修改 config.py
        # print("Settings are now managed in phonetic_toolbox/models/config.py. Runtime changes are not persisted.")
        pass

    def get(self, key: str, default: Any = None) -> Any:
        return self.config.get(key, default)

    def set(self, key: str, value: Any):
        self.config[key] = value
        # 运行时修改仍然生效，但不会保存到磁盘
        # self.save() 

    def get_all(self) -> Dict[str, Any]:
        return self.config.copy()

    def get_config_object(self) -> AcousticConfig:
        """Get current settings as AcousticConfig object"""
        # Load directly from config dict, using default AcousticConfig structure as template
        # This is more robust than manual mapping if keys match exactly
        
        # Start with default config
        cfg = AcousticConfig()
        
        # Iterate over all fields in AcousticConfig
        for key, value in asdict(cfg).items():
            if key in self.config:
                # Convert type to match the dataclass field type (basic conversion)
                target_type = type(value)
                try:
                    # Special handling for bool because bool("False") is True
                    if target_type == bool:
                         setattr(cfg, key, bool(self.config[key]))
                    else:
                         setattr(cfg, key, target_type(self.config[key]))
                except (ValueError, TypeError):
                    # Fallback to default if conversion fails
                    pass
                    
        return cfg
