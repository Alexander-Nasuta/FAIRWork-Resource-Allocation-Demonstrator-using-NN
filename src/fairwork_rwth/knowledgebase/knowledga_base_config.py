from typing import Dict

import yaml
import pprint

from fairwork_rwth.utils.logger import log

log.info("setting up knowledge base...")

credentials_file = "knowledge_base_config.yaml"
log.info(f"parsing file '{credentials_file}'.")
config_dict = {}
try:
    with open(f'resources/{credentials_file}') as f:
        config_dict = yaml.safe_load(f)['knowledgebase']
except FileNotFoundError:
    log.error(f"could not find file {credentials_file}. "
              f"make sure your working directory corresponds to the root of the project."
              f"make sure you created the '{credentials_file}' "
              f"based on the structure of 'knowledge_base_config.yaml.example'.")

log.info("file successfully parsed")
log.info(f"config: \n{pprint.pformat({k:v if k != '_pass' else '*************' for k,v in config_dict.items()})}")


if __name__ == '__main__':
    pass
