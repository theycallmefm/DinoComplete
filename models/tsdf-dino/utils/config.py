# config.py

import yaml

class Config:
    """Simple namespace-like config object."""

    def __init__(self, yaml_path: str = "config.yaml"):
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)

        # Load all YAML attributes into this class
        for key, value in data.items():
            setattr(self, key, value)

    def __repr__(self):
        return f"<Config: {self.__dict__}>"

# Global config object
cfg = Config("config_distillation.yaml")
