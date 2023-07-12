from fairwork_rwth.utils.logger import log

from fairwork_rwth.knowledgebase.auth_controller import get_token
from swagger_client import BreakdownControllerApi


if __name__ == '__main__':
    import pandas as pd

    data = {
        "calories": [420, 380, 390],
        "duration": [50, 40, 45]
    }

    df = pd.DataFrame(data)
    df.to_csv("dummy.csv", encoding='utf-8', index=False)
    """
:param async_req bool
        :param str file: (required)
        :param str model: Model name (required)
        :param int node: Breakdown element instance ID (required)
        :param str prop: Aggregated property name (URN) (required)
        :param str repository: Repository name (required)
        :param str token: Server connection token (required)
        
        breakdown_controller.append_aggr_prop_csv_using_post(
        token=get_token(),
        file='dummy.csv',
        node='261993359604',  # dummy data node
        model="dummy_model_name",
        prop="dummy_urn",
        repository="EDMtruePLM"
    )
    """

