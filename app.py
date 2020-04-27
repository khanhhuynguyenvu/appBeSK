import os

from flask import Flask, request, flash, jsonify, Response
from flask_cors import CORS
from werkzeug.utils import secure_filename

from functions.Quick import get_records_cache
from models.Record import Record

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'csv'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DATA_FILE'] = 'Data.csv'


@app.route('/')
def hello_world():
    return 'App BE service!'


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/upload/data', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            return jsonify(uploaded=False, error='No file part')
        file = request.files['file']
        if file.filename == '':
            return jsonify(uploaded=False, error='No selected file')
        if file and allowed_file(file.filename):
            filename = secure_filename(app.config['DATA_FILE'])
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return jsonify(uploaded=True, filename=filename)
        return jsonify(uploaded=False, error='Not allowed extensions')


@app.route('/get/records/<int:num>', methods=['GET'])
def get_records(num):
    if request.method == 'GET':
        df = get_records_cache(num)
        limit = max(0, min(num, df.shape[0]))
        records = [Record(df.columns, df.iloc[i].values).toDictionary() for i in range(limit)]
        return jsonify({'Records': records})


if __name__ == '__main__':
    app.run()
