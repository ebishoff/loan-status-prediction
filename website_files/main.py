from flask import Flask, url_for, request, render_template
from flask_restful import Resource, Api

import pandas as pd
import numpy as np
import pickle
import os
# import cloudstorage
from google.cloud import storage

# from google.appengine.api import app_identity


app = Flask(__name__)

api = Api(app)


@app.route('/')

def home():
    return render_template('home.html')


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080, debug=True)

# def get(self):
#     bucket_name = os.environ.get(
#         'BUCKET_NAME', app_identity.get_default_gcs_bucket_name())

#     self.response.headers['Content-Type'] = 'text/plain'
#     self.response.write(
#         'Demo GCS Application running from Version: {}\n'.format(
#             os.environ['CURRENT_VERSION_ID']))
#     self.response.write('Using bucket name: {}\n\n'.format(bucket_name))

MODEL_BUCKET = os.environ['MODEL_BUCKET']
MODEL_FILENAME = os.environ['MODEL_FILENAME']
MODEL = None
COLS_BUCKET = os.environ['COLS_BUCKET']
COLS_FILENAME = os.environ['COLS_FILENAME']
COLS = None
FIT_BUCKET = os.environ['FIT_BUCKET']
FIT_FILENAME = os.environ['FIT_FILENAME']
FIT = None
SCALER_BUCKET = os.environ['SCALER_BUCKET']
SCALER_FILENAME = os.environ['SCALER_FILENAME']
SCALER = None

# def read_file(self, filename):
#     self.response.write(
#         'Abbreviated file content (first line and last 1K):\n')

#     with cloudstorage.open(filename) as cloudstorage_file:
#         self.response.write(cloudstorage_file.readline())
#         cloudstorage_file.seek(-1024, os.SEEK_END)
#         self.response.write(cloudstorage_file.read())

@app.before_first_request
def _load_model():
    global MODEL
    global COLS
    global FIT
    global SCALER

    client = storage.Client()
    bucket = client.get_bucket(MODEL_BUCKET)
    bucket_cols = client.get_bucket(COLS_BUCKET)
    bucket_fit = client.get_bucket(FIT_BUCKET)
    bucket_scaler = client.get_bucket(SCALER_BUCKET)

    blob = bucket.get_blob(MODEL_FILENAME)
    blob_cols = bucket_cols.get_blob(COLS_FILENAME)
    blob_fit = bucket_fit.get_blob(FIT_FILENAME)
    blob_scaler = bucket_scaler.get_blob(SCALER_FILENAME)

    s = blob.download_as_string()
    t = blob_cols.download_as_string()
    u = blob_fit.download_as_string()
    v = blob_scaler.download_as_string()

    MODEL = pickle.loads(s)
    COLS = pickle.loads(t)
    FIT = pickle.loads(u)
    SCALER = pickle.loads(v)

class Predict(Resource):

    def post(self):

        data = request.get_json(force=True)


        arr = np.array([[float(data['loan_amnt']), float(data['int_rate']),
        float(data['annual_inc']), float(data['dti']), float(data['open_acc']),
        float(data['revol_bal']), float(data['revol_util']), float(data['total_acc']), (data['term']),
         (data['grade']), (data['emp_length']), (data['home_ownership']),
         (data['verification_status']), (data['purpose']),
         (data['issue_d_mnth']), (data['earliest_cr_line_mnth']), (data['last_credit_pull_d_mnth']),
         (data['delinq_2yrs_2cat']), (data['inq_last_6mths_2cats']),
         (data['pub_rec_2cats']), (data['pub_rec_bankruptcies_2cats'])]])

        columns_mod = ['loan_amnt', 'int_rate', 'annual_inc', 'dti', 'open_acc', 'revol_bal', 'revol_util', 'total_acc',
        'term', 'grade', 'emp_length', 'home_ownership', 'verification_status', 'purpose', 'issue_d_mnth', 'earliest_cr_line_mnth',
        'last_credit_pull_d_mnth', 'delinq_2yrs_2cat', 'inq_last_6mths_2cats', 'pub_rec_2cats', 'pub_rec_bankruptcies_2cats']

        df = pd.DataFrame(arr, columns=columns_mod)

        # create dummy variables for categorical data
        all_cat_vars = ['term', 'emp_length', 'home_ownership', 'purpose',
              'pub_rec_bankruptcies_2cats', 'grade', 'verification_status', 'issue_d_mnth',
              'earliest_cr_line_mnth', 'last_credit_pull_d_mnth',
              'delinq_2yrs_2cat', 'inq_last_6mths_2cats', 'pub_rec_2cats']

        for var in all_cat_vars:
            df = pd.concat([df, pd.get_dummies(df[var], prefix=var)], axis=1)

        df = df.drop(columns=all_cat_vars)

        dff = pd.DataFrame(data=df, columns=COLS)
        dff.fillna(0, inplace=True)
        dff_n = pd.DataFrame(SCALER.transform(dff), columns=COLS)
        dff_m = FIT.transform(dff_n)
        answer = MODEL.predict(dff_m)[0]
        answer_prob_fp = MODEL.predict_proba(dff_m)[0][1]

        return {'loan_status': str(answer), 'probability': answer_prob_fp}


api.add_resource(Predict, '/predict')
