import os

import pandas as pd
import numpy as np
import pathlib as pl

from rwth.demonstrator_ml.demonstrator_nn_v01 import processed_prediction, model_prediction
from rwth.utils.logger import log

from flask import Flask
from flask_restx import Api, Resource, fields

from werkzeug.middleware.proxy_fix import ProxyFix

app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app)
api = Api(app, version='0.3', title='AI models API',
          description='AI Enrichment FAIRWork Demonstrator.',
          )

ns = api.namespace('demonstrator-nn', description='Endpoint calls')

demonstrator_output_model = api.model('DemonstratorOutput', {
    'allocation': fields.List(required=True, description='worker allocation.', cls_or_instance=fields.Float)
})
demonstrator_multi_output_model = api.model('DemonstratorMultiOutput', {
    'allocation': fields.List(required=True, description='allocated worker IDs', cls_or_instance=fields.Integer)
})

input_fields = ["ID", "Woker available", "Medical condtion", "Efficiency on the line", "Difficulty of the geometry",
                "production priority", "due date in days", "UTE allocation", "worker preference"]

defaults_dict = {
    "ID": [1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010, 1011, 1012, 1013, 1014, 1015, 1016, 1017,
           1018, 1019, 1020, 1021, 1022, 1023, 1024, 1025, 1026, 1027, 1028, 1029, 1030, 1031, 1032, 1033, 1034, 1035,
           1036, 1037, 1038, 1039, 1040, 1041, 1042, 1043, 1044, 1045, 1046, 1047, 1048, 1049, 1050, 1051, 1052, 1053,
           1054, 1055, 1056, 1057, 1058, 1059, 1060, 1061, 1062, 1063, 1064, 1065, 1066, 1067, 1068, 1069, 1070, 1071,
           1072, 1073, 1074, 1075, 1076, 1077, 1078, 1079, 1080, 1081, 1082, 1083, 1084, 1085, 1086, 1087, 1088, 1089,
           1090, 1091, 1092, 1093, 1094, 1095, 1096, 1097, 1098, 1099, 1100, 1101, 1102, 1103, 1104, 1105, 1106, 1107,
           1108, 1109, 1110, 1111, 1112, 1113, 1114, 1115, 1116, 1117, 1118, 1119, 1120, 1121, 1122, 1123, 1124, 1125,
           1126, 1127, 1128, 1129, 1130, 1131, 1132, 1133, 1134, 1135, 1136, 1137, 1138, 1139, 1140, 1141, 1142, 1143,
           1144, 1145, 1146, 1147, 1148, 1149, 1150, 1151, 1152, 1153, 1154, 1155, 1156, 1157, 1158],
    "Woker available": [1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                        1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0,
                        0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1,
                        1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0,
                        0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1],
    "Medical condtion": [0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1,
                         0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1,
                         0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1,
                         0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1,
                         0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0],
    "Efficiency on the line": [0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0,
                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0,
                               0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1,
                               0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0,
                               1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1,
                               0, 1, 1, 0, 0, 1, 1, 0, 1],
    "Difficulty of the geometry": [1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0,
                                   0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1,
                                   0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1,
                                   0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1,
                                   0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1,
                                   1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1],
    "production priority": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                            1, 1, 1, 1],
    "due date in days": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    "UTE allocation": [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    "worker preference": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 1, 0],
    "num_workers": 4
}

demonstrator_input_model = api.model('DemonstratorInput', {
    **{
        field: fields.List(
            required=True,
            description=f'{field} colum',
            cls_or_instance=fields.Float,
            default=defaults_dict[field]
        )
        for field in input_fields
    },
    'num_workers': fields.Integer(required=True, description='number of workers to allocate.', default=4)
})

demonstrator_input_model_multi = api.model('DemonstratorMultiInput', {
    **{
        field: fields.List(
            required=True,
            description=f'{field} colum. This list has to be of lenght {len(defaults_dict[field])}.',
            cls_or_instance=fields.Float,
            default=defaults_dict[field]
        )
        for field in input_fields
    },
    'order-id': fields.Integer(required=True, description='number of workers to allocate.', default=1),
    'line': fields.Integer(default=20),
    'backup-line': fields.Integer(default=24),
    'required-number-of-operators': fields.Integer(required=True, description='number of workers to allocate.'),
    'number-stevedors': fields.Integer(default=2),
    'weight': fields.Float(default=1.47),
    'type-of-part': fields.String(default="small parts"),
    'pieces-per-container': fields.Integer(deafult=10),
    'mandatory': fields.Integer(default=0),
    'due-date': fields.Integer(default=1),
    'quantity-to-produce': fields.Integer(deafult=1_000)
})


@ns.route('/')
class DemonstratorNN(Resource):

    @ns.doc('allocate_workers')
    @ns.expect(demonstrator_input_model)
    @ns.marshal_list_with(demonstrator_output_model)
    def post(self):
        """perform a prediction on the given input data. `num_workers` is the number of workers to allocate."""
        log.info("received request to perform a prediction on the given input data.")
        log.info(f"payload: {api.payload}")

        data = dict(api.payload)
        n_workers = data.pop("num_workers")

        df = pd.DataFrame(data)

        log.info(f"input_data (shape: {df.shape}): \n{df.head()}")

        x_data = np.ravel(df.to_numpy())

        if x_data.shape != (1431,):
            raise ValueError(f"expected shape (1431,), but got {x_data.shape}."
                             f"make sure to provide all list input fields have length '159'.")

        prediction = processed_prediction(x_data, n_workers=n_workers)

        return {"allocation": prediction}


@ns.route('/multi/')
class DemonstratorMultipleAllocationsNN(Resource):

    @ns.doc('allocate_multi_workers')
    @ns.expect([demonstrator_input_model_multi])
    @ns.marshal_list_with(demonstrator_multi_output_model)
    def post(self):
        """perform a prediction on the given input data. `num_workers` is the number of workers to allocate."""
        log.info("received request to perform a prediction on the given input data.")
        log.info(f"payload: {api.payload}")

        data = list(api.payload)

        # check if enough workers are available in total
        n_required = sum([d["required-number-of-operators"] for d in data])
        n_workers = sum(data[0]["Woker available"])

        log.info(f"the request requires {n_required} workers in total, while {n_workers} are available.")
        if n_required > n_workers:
            raise ValueError(f"the request requires {n_required} workers in total, while {n_workers} are available.")

        # sort by field mandatory to mandetory first
        # data = sorted(data, key=lambda d: d["mandatory"], reverse=True)

        res = []
        allocated_workers = []
        for i, d in enumerate(data):
            n_workers = d.pop("number-stevedors")
            df = pd.DataFrame({k: v for k, v in d.items() if k in input_fields})
            # set availability of already allocated workers to 0
            # for idx in allocated_workers:
            #  df.iloc[idx, 1] = 0

            # print(f"available workers {df.iloc[:, 1].sum()}")
            x_data = np.ravel(df.to_numpy())

            if x_data.shape != (1431,):
                raise ValueError(f"expected shape (1431,), but got {x_data.shape}."
                                 f"make sure to provide all list input fields have length '159'.")

            prediction = model_prediction(x_data)
            # mask out already allocated workers
            prediction = np.array([p if i not in allocated_workers else 0.0 for i, p in enumerate(prediction)])
            # set n_workers largest values to 1, rest to 0
            prediction = np.array(
                [1 if i in np.argpartition(prediction, -n_workers)[-n_workers:] else 0 for i in range(len(prediction))])
            allocated_workers.extend([i for i, p in enumerate(prediction) if p == 1])
            allocated_worker_ids = [d["ID"][i] for i, p in enumerate(prediction) if p == 1]
            res.append({"allocation": allocated_worker_ids})

        return res


def main() -> None:
    from waitress import serve
    log.info("starting flask app...")
    serve(app, host="0.0.0.0", port=8080)


if __name__ == '__main__':
    # change working directory to the root of the project
    os.chdir(pl.Path(__file__).parent.parent.parent.parent)
    log.info(f"working directory: {os.getcwd()}")
    # print([random.random() for _ in range(159)])
    main()
