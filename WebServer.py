#-*-coding:utf-8-*-
from flask import Flask
from flask import request
from flask import make_response, jsonify
import os
from werkzeug.utils import secure_filename
from model.rice_health_model import RiceHealthModel
from model.rice_price_regression_model import RiceQualityValidator

app = Flask(__name__)
basedir=os.path.abspath(os.path.dirname(__file__))

@app.route('/')
def test():
    return 'flask restful API HeartBeat OK'

#upload image, return predict score and disease
@app.route('/upload',methods=['POST'])
def upload():
    #get post params
    parameters = request.form["parameters"]
    #get uploaded file content
    f = request.files['file']
    # 当前文件所在路径
    basepath = os.path.dirname(__file__)
    upload_path = os.path.join(basepath, 'upload', secure_filename(f.filename))
    # 保存文件
    f.save(upload_path)
    # return file path
    # response = make_response(jsonify({'file_url': upload_path}, 200))
    result = RiceHealthModel().predict(upload_path)
    result['file_url'] = upload_path
    return jsonify(result)


@app.route('/valuateprice', methods=['POST'])
def valuate_price():
    return jsonify(RiceQualityValidator.predict())

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
    # app.run(host='192.168.3.13')
