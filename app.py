import os

from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

from functions.Drive import get_Drive_csv
from functions.Processing import get_final_data
from models.RecordList import RecordList

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'csv'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DATA_FILE'] = 'Data.csv'
app.config['URL_DRIVE'] = 'https://drive.google.com/open?id=1LGYpja_MiEJa2i_iNlICMjGkzSXqR-O8'


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


@app.route('/get/records', methods=['GET'])
def get_records():
    if request.method == 'GET':
        if 'num' not in request.args.keys():
            return jsonify(message='num is required')
        num = int(request.args.get('num'))
        df = get_Drive_csv(google_drive_url=app.config['URL_DRIVE'], numberOfRow=num)
        records = RecordList(df, num).toList()
        return jsonify({'records': records, 'size': num})


@app.route('/upload/drive', methods=['POST'])
def get_drive_csv():
    if request.method == 'POST':
        if 'url' not in request.args.keys():
            return jsonify(message='url is required')
        url = request.args.get('url')
        df = get_Drive_csv(google_drive_url=url)
        app.config['URL_DRIVE'] = url  # set global
        return jsonify(message='upload success')
    return jsonify(message='upload failed')


@app.route('/get/records/cleaned', methods=['GET'])
def get_records_clean():
    if request.method == 'GET':
        num = None
        if 'num' in request.args.keys():
            num = int(request.args.get('num'))
        df = get_Drive_csv(google_drive_url=app.config['URL_DRIVE'], numberOfRow=num)
        return jsonify({'records': get_final_data(df), 'size': df.shape[0]})
    return jsonify(message='request failed')


if __name__ == '__main__':
    app.run()
