import json
import pprint
import random
import uuid
import requests
import urllib

from typing import List

from fairwork_rwth.utils.logger import log

from fairwork_rwth.knowledgebase.auth_controller import auth_context, credentials_dict
from fairwork_rwth.knowledgebase.knowledga_base_config import config_dict
from swagger_client import AdminControllerApi, ProjectInfo, SystemProjectInfo, BreakdownControllerApi, BreakdownInfo, \
    BreakdownElementInfo, BreakdownElementSearchResultInfo, BreakdownElementInfoWrapper, IFDControllerApi, StringResult
from swagger_client.rest import ApiException

if __name__ == '__main__':
    log.info("follow along example")

    adm_controller = AdminControllerApi()
    bkd_controller = BreakdownControllerApi()
    idf_controller = IFDControllerApi()

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

    #####################  Deleting Break down element ######################

    for search_entry in q_search_res_list:
        node_id = search_entry.bkdn_elem_info.instance_id
        if node_id == elem_info.instance_id:
            continue
        log.info(f"deleting elem with id: {node_id}")
        del_res = DeleteNodeResponse = bkd_controller.delete_node_using_delete(
            model=model,
            node=node_id,
            repository=repository,
            token=auth_context.get_token()
        )

    #####################  Updating breakdown element Property information  ######################

    """
    
    print(elem_info.date_created)
    upload_prop_data = {
        'act_timestamp': '',
        'props': [
            'urn:rdl:Palfinger_Crane_Assembly:Manufacturer',
        ],
        'ptypes': [
            'string',
            #'N',
            #'string'
        ],
        'units': [
            '',
        ],
        'vals': [
            'ABC',
        ]
    }

    bkd_elem_info_wrapper: BreakdownElementInfoWrapper = bkd_controller.update_prop_using_post(
        node=elem_info.instance_id,
        model=model,
        repository=repository,
        token=auth_context.get_token(),
        **upload_prop_data
    )
    
    """

    #####################  Uploading aggr data  ######################

    log.info("uploading json data")

    # Data to be written
    # IT NEEDS TO BE AN ARRAY. "normal" object-style JSON does not work. Big uff.
    j_data = [{
        "foo": "some string",
        "bar": 1337
    }]

    # Serializing json
    json_object = json.dumps(j_data, indent=4)

    # Writing to sample.json
    with open("./resources/sample.json", "w") as outfile:
        outfile.write(json_object)

    with open('./resources/sample.json', 'r') as f:
        r_json = json.load(f)
        log.info(f"dummy json object: {pprint.pformat(r_json)}")

    json_prop_name = f"some_json"

    prn_urn = "urn:plcs:rdl:TruePLM:ComplexDataType"
    example_url = f"http://" \
                  f"172.31.10.11:8080/" \
                  f"EDMtruePLM/" \
                  f"api/ifd_concept/class/" \
                  f"TruePLMprojectsRep/Palfinger_Crane_Assembly_RDL/" \
                  f"urn:rdl:Palfinger_Crane_Assembly:postman1/" \
                  f"urn:plcs:rdl:TruePLM:ComplexDataType/" \
                  f"F5GTTNR6F5BKAOTYDE"

    urn_without_prefix = f"code_{uuid.uuid4()}"
    urn = f"urn:rdl:Palfinger_Crane_Assembly:{urn_without_prefix}"
    url = f"http://" \
          f"{credentials_dict['server']}:{credentials_dict['port']}/" \
          f"{config_dict['project_prefix']}/" \
          f"api/ifd_concept/class/" \
          f"{repository}/{model}_RDL/" \
          f"{urn}/" \
          f"{prn_urn}/" \
          f"{auth_context.get_token()}"

    # log.info(f"creating aggregate struct. urn: {urn}")
    # log.info(f"req-url: {url}")
    # res = requests.post(url=url)
    # res = idf_controller.add_class_using_post(**args)
    # log.info(f"result: {res}, text: {res.text}")

    # urn = f"urn:rdl:Palfinger_Crane_Assembly:prop_for_agg_scruct_{urn_without_prefix}"
    # log.info(f"creating bkd elem prop. urn: ")

    prop_url_1 = f"" \
                 f"http://172.31.10.11:8080/" \
                 f"EDMtruePLM/" \
                 f"api/adm_user/prop_info/" \
                 f"TruePLMprojectsRep/" \
                 f"Palfinger_Crane_Assembly/" \
                 f"propUi3/" \
                 f"{auth_context.get_token()}?_=1689845726919"

    # log.info(f"creating breakdown property. urn: {urn}")
    # log.info(f"req-url-1: {prop_url_1}")
    # res = requests.get(url=prop_url_1)
    # log.info(f"result: {res}, text: {res.text}")

    prop_url_2 = f"http://" \
                 f"172.31.10.11:8080/" \
                 f"EDMtruePLM/" \
                 f"api/ifd_concept/prop/" \
                 f"TruePLMprojectsRep/" \
                 f"Palfinger_Crane_Assembly_RDL/" \
                 f"urn:rdl:Palfinger_Crane_Assembly:propUi1/" \
                 f"F5GTTNR6F5BKAOTYDE"

    # log.info(f"creating breakdown property. urn: {urn}")
    # log.info(f"req-url-2: {prop_url_2}")
    # res = requests.post(url=prop_url_2)
    # log.info(f"result: {res}, text: {res.text}")

    prop_url_3 = f"http://" \
                 f"172.31.10.11:8080/" \
                 f"EDMtruePLM/" \
                 f"api/ifd_concept/prop/" \
                 f"TruePLMprojectsRep/" \
                 f"Palfinger_Crane_Assembly_RDL/" \
                 f"urn:rdl:Palfinger_Crane_Assembly:propUi1/" \
                 f"F5GTTNR6F5BKAOTYDE"

    log.info(f"creating breakdown property. urn: {urn}")
    log.info(f"req-url-3: {prop_url_3}")
    # res = requests.post(url=prop_url_3)
    # log.info(f"result: {res}, text: {res.text}")

    # log.info("uploading json file...")
    args = {
        "file": './resources/sample.json',
        "model": model,
        "node": elem_info.instance_id,
        "prop": f"FooBarJSONCode",
        "repository": repository,
        "token": auth_context.get_token()
    }
    # log.info(pprint.pformat(args))

    # res = bkd_controller.append_aggr_prop_json_using_post(**args)
    # log.info(f"res: {res}")
