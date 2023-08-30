import os

import pandas as pd
import numpy as np
import pathlib as pl

from rwth.demonstrator_ml.demonstrator_nn_v01 import processed_prediction
from rwth.utils.logger import log

from flask import Flask
from flask_restx import Api, Resource, fields
from werkzeug.middleware.proxy_fix import ProxyFix

app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app)
api = Api(app, version='0.1', title='AI models API',
          description='AI Enrichment FAIRWork Demonstrator.',
          )

ns = api.namespace('demonstrator-nn', description='Endpoint calls')

demonstrator_output_model = api.model('DemonstratorInput', {
    'allocation': fields.List(required=True, description='worker allocation.', cls_or_instance=fields.Float)
})


input_fields = ["ID", "Woker avaibale", "Medical condtion", "Efficiency on the line", "Difficulty of the geometry",
                "production priority", "due date in days", "UTE allocation", "worker preference"]
demonstrator_input_model = api.model('DemonstratorInput', {
    **{
        field: fields.List(required=True, description=f'{field} colum', cls_or_instance=fields.Float)
        for field in input_fields
    },
    'num_workers': fields.Integer(required=True, description='number of workers to allocate.')
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


def main() -> None:
    log.info("starting flask app...")
    app.run()


if __name__ == '__main__':
    # change working directory to the root of the project
    os.chdir(pl.Path(__file__).parent.parent.parent.parent)
    log.info(f"working directory: {os.getcwd()}")
    # print([random.random() for _ in range(159)])
    main()
