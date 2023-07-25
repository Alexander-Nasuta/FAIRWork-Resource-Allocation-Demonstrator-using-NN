from typing import Dict

import yaml
import pprint
import asyncio

from fairwork_rwth.utils.logger import log
from swagger_client import AuthorizationControllerApi, LoginRez

log.info("setting up knowledge base...")

credentials_file = "knowledge_base_credentials.yaml"
log.info(f"parsing file '{credentials_file}'.")
credentials_dict = {}
try:
    with open(f'resources/{credentials_file}') as f:
        credentials_dict = yaml.safe_load(f)['knowledgebase']
except FileNotFoundError:
    log.error(f"could not find file {credentials_file}. "
              f"make sure your working directory corresponds to the root of the project."
              f"make sure you created the '{credentials_file}' "
              f"based on the structure of 'knowledge_base_credentials.yaml.example'.")

log.info("file successfully parsed")
log.info(f"config: \n{pprint.pformat({k:v if k != '_pass' else '*************'  for k,v in credentials_dict.items()})}")

auth_controller = AuthorizationControllerApi()


class AuthContext:
    __token: str | None
    __credentials: Dict

    def __init__(self, credentials: Dict):
        self.__token = None
        self.__credentials = credentials
        self.get_token()

    def get_token(self) -> str:
        if self.__token is None:
            login_res = auth_controller.login_submit_using_post(**credentials_dict)
            if login_res.error is not None:
                raise ValueError(f"login failed. response body: {login_res}")
            else:
                self.__token = login_res.token
                log.info(f"received auth token: '{self.__token}'")
        return self.__token


auth_context = AuthContext(credentials=credentials_dict)

if __name__ == '__main__':
    pass
