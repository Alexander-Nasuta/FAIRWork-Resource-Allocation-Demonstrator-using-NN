import uuid

from typing import List

from fairwork_rwth.utils.logger import log

from fairwork_rwth.knowledgebase.auth_controller import auth_context
from swagger_client import AdminControllerApi, ProjectInfo, SystemProjectInfo, BreakdownControllerApi, BreakdownInfo, \
    BreakdownElementInfo, BreakdownElementSearchResultInfo
from swagger_client.rest import ApiException

if __name__ == '__main__':
    log.info("follow along example")

    adm_controller = AdminControllerApi()
    bkd_controller = BreakdownControllerApi()

    ##################### Create new project ######################
    # create new project
    log.info("creating a new project...")

    file = {
        'file': "./resources/Knurled_screws.stp"}  # declaring the required step file path (.stp, .txt, .dex)

    data_param = {"descr": "Crane assembly",  # description of the creating project
                  "folder": "",  # Project folder for the new project placement
                  "is_bkd_tmpl": "false",  # Create new project as a breakdown template(true) or not(false)
                  'is_tmpl': "false",  # Create new project as a project template(true) or not(false)
                  'name': "Palfinger_Crane_Assembly",  # User required Name of the Project
                  'node_type': "",  # node type
                  'src': "pdm",  # Type of the data file (pdm for STEP files)
                  'tmpl': ""}

    try:
        proj_info: ProjectInfo = adm_controller.create_project_using_post(
            **file, **data_param,
            token=auth_context.get_token()
        )
        log.info(f"creation request response: {proj_info}")
    except ApiException as e:
        log.warning(e)

    ##################### project information ######################

    project_list: List[SystemProjectInfo] = adm_controller.get_all_proj_using_get(token=auth_context.get_token())
    log.info(f"projects: {[p.name for p in project_list]}")

    #####################  root_bd element information ######################

    model = "Palfinger_Crane_Assembly"  # the project name
    repository = "TruePLMprojectsRep"

    log.info(f"fetching root node for project '{model}'")
    root_node: BreakdownInfo = bkd_controller.get_root_node_using_get(
        model=model,
        repository=repository,
        token=auth_context.get_token()
    )
    log.info(f"response: {root_node}")

    #####################  Creating new BD element ######################

    # Required input parameters

    node = root_node.root_bkdn_elem.instance_id  # '214748369406'  # provide required instance id or node path
    model = 'Palfinger_Crane_Assembly'  # Project name

    repository = "TruePLMprojectsRep"  # Project repo

    new_bkd_elem_name = f'SENSORS_{uuid.uuid4()}'

    data_param = {'act_timestamp': '',  # Current time stamp
                  'descr': 'stores sensors data',  # Description of element
                  'name': new_bkd_elem_name,  # Required Breakdown element name
                  'node_type': 'urn:rdl:epm-std:Unit',  # Node type
                  'tmpl': ''}  # Name of the breakdown template

    log.info(f"creating a new breakdown element (root node 'instance_id':'{node}')")
    elem_info: BreakdownElementInfo = bkd_controller.create_node_using_post(
        token=auth_context.get_token(),
        node=node,
        model=model,
        repository=repository,
        **data_param
    )
    log.info(f"elemnt_info: {elem_info}")

    ######################## q_search methods ######################

    model = 'Palfinger_Crane_Assembly'  # Project name

    repository = "TruePLMprojectsRep"  # Project repo

    data_search_params = {"case_sens": "false",  # Use case sensitive search or not
                          "domains": "ID",
                          # CSV list of subjects for search, can include ID, DESCRIPTION, CLASS, PROPERY
                          "folder_only": "true",  # Return only direct children of parent folder
                          "node": "",  # Breakdown element instance id - root of interesting branch
                          "page": "",  # Start page of output
                          "page_size": "",  # Page size of output
                          "pattern": "SENSORS_*",  # Search string pattern (for LIKE operations in EXPRESS)
                          "props": ""}  # CSV list of property names where to apply search pattern (when PROPERTY is listed)

    log.info(f"performing 'q_search'")
    q_search_res_list: List[BreakdownElementSearchResultInfo] = bkd_controller.quick_search_using_get(
        token=auth_context.get_token(),
        model=model,
        repository=repository,
        **data_search_params
    )
    log.info(f"q_search results: {[res.bkdn_elem_info.name for res in q_search_res_list]}")

    #####################  BD search element ######################

    log.info(f"performing advanced search")
    adv_search_res: List[BreakdownElementSearchResultInfo] = bkd_controller.advanced_search_node_using_get(
        token=auth_context.get_token(),
        model=model,
        repository=repository
    )
    log.info(f"performing advanced search results: {[res for res in adv_search_res]}")

    #####################  Updating breakdown element Property information  ######################


