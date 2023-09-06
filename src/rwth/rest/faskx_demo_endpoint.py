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
api = Api(app, version='0.2', title='AI models API',
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
demonstrator_input_model = api.model('DemonstratorInput', {
    **{
        field: fields.List(required=True, description=f'{field} colum', cls_or_instance=fields.Float)
        for field in input_fields
    },
    'num_workers': fields.Integer(required=True, description='number of workers to allocate.')
})

demonstrator_input_model_multi = api.model('DemonstratorMultiInput', {
    **{
        field: fields.List(required=True, description=f'{field} colum', cls_or_instance=fields.Float)
        for field in input_fields
    },
    'order-id': fields.Integer(required=True, description='number of workers to allocate.'),
    'line': fields.Integer(),
    'backup-line': fields.Integer(),
    'required-number-of-operators': fields.Integer(required=True, description='number of workers to allocate.'),
    'number-stevedors': fields.Integer(),
    'weight': fields.Float(),
    'type-of-part': fields.String(),
    'pieces-per-container': fields.Integer(),
    'mandatory': fields.Integer(),
    'due-date': fields.Integer(),
    'quantity-to-produce': fields.Integer()
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
            #for idx in allocated_workers:
            #  df.iloc[idx, 1] = 0

            #print(f"available workers {df.iloc[:, 1].sum()}")
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
    log.info("starting flask app...")
    app.run()


if __name__ == '__main__':
    # change working directory to the root of the project
    os.chdir(pl.Path(__file__).parent.parent.parent.parent)
    log.info(f"working directory: {os.getcwd()}")
    # print([random.random() for _ in range(159)])
    main()
