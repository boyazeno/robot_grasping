import yaml
from yaml.loader import SafeLoader
from typing import Any, Dict
from pathlib import Path
import logging

def load_config_from_yaml(file_path=""):
    try:
        with open(file_path, 'r') as f: 
                config = yaml.load(f, Loader=SafeLoader)
        return config
    except Exception as ex:
        logging.error(f"Cannot load config from yaml, use default: {str(ex)}")
    return {}

class ParameterIO:
    def __init__(self, base_dir:Path, file_name:str) -> None:
        self._base_dir = base_dir
        self._file_name = file_name
        self._full_name = self._get_full_name()
        self._data = None


    def save(self, params:Dict):
        with open(self._full_name.as_posix(), 'w') as f:
            yaml.dump(params, f, sort_keys=False, default_flow_style=False)
        logging.info(f"Parameter saved to path {self._full_name}")

    def load(self):
        if not self._is_file_exist(self._full_name):
            raise FileExistsError(f"File not exist: {self._full_name}")

        with open(self._full_name, 'r') as f: 
            self._data = yaml.load(f, Loader=SafeLoader)
        return self._data

    def get(self, param_name:str)->Any:
        if self._data is None:
            logging.warn(f"Parameter not loaded! Loading parameters")
            self.load()
            logging.warn(f"Done.")
        return self._data[param_name]


    def _get_full_name(self, base_dir:Path, file_name:str)->Path:
        return base_dir/(file_name + ".yaml")

    
    def _is_file_exist(self, file_name:str)->bool:
        return Path(file_name).exists()