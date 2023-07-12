import yaml
import pprint
import os

import pathlib as pl

from logger import log

if __name__ == '__main__':
    config_file = "knowledge_base_config.yaml"
    log.info(f"parsing file '{config_file}'.")
    config_dict = {}
    try:
        with open(f'resources/{config_file}') as f:
            config_dict = yaml.safe_load(f)['knowledgebase']
    except FileNotFoundError:
        log.error(f"could not find file {config_file}. "
                 f"make sure your working directory corresponds to the root of the project."
                 f"make sure you created the '{config_file}' "
                 f"based on the structure of 'knowledge_base_config.yaml.example'.")

    log.info("file successfully parsed")
    log.info(f"config: \n{pprint.pformat(config_dict)}")

