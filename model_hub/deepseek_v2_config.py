import json
from pathlib import Path
from typing import Union

class DeepSeekV2Config:
    """
    Loads and holds the configuration for the DeepSeek-V2 model from a config.json file.
    This class replaces the need for transformers.AutoConfig.
    """
    def __init__(self, **kwargs):
        # Set attributes from kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)

    @classmethod
    def from_pretrained(cls, model_path: Union[str, Path]):
        """
        Loads the configuration from a directory containing a config.json file.
        
        Args:
            model_path (Union[str, Path]): Path to the model directory.
        
        Returns:
            DeepSeekV2Config: The configuration object.
        """
        model_path = Path(model_path)
        config_file = model_path / "config.json"
        
        if not config_file.is_file():
            raise FileNotFoundError(f"Configuration file 'config.json' not found in {model_path}")

        with open(config_file, "r") as f:
            config_dict = json.load(f)
        
        return cls(**config_dict)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.__dict__})"

