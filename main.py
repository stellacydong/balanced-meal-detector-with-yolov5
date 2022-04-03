# run by typing python3 main.py in a terminal 
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename
from flask import Flask, flash, request, redirect, url_for, render_template
from flask import send_from_directory
from utils import get_base_url, allowed_file, and_syntax

import argparse
import requests
import tldextract
import pytube
import hashlib
import time
import base64

from PIL import Image
from flask import Flask, request, render_template, redirect, make_response, jsonify
from pathlib import Path
from werkzeug.utils import secure_filename
from modules import get_prediction, get_video_prediction
from flask_cors import CORS, cross_origin
from werkzeug.utils import secure_filename


parser = argparse.ArgumentParser('YOLOv5 Online Food Recognition')



# setup the webserver
'''
    coding center code
    port may need to be changed if there are multiple flask servers running on same server
    comment out below three lines of code when ready for production deployment
'''
port = 12222
base_url = get_base_url(port)
app = Flask(__name__, static_url_path=base_url+'static')
app.secret_key = "super secret key"

'''
    cv scaffold code
    uncomment below line when ready for production deployment
'''
# app = Flask(__name__)

UPLOAD_FOLDER = 'images'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])




def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def allowed_image_filesize(filesize):

    if int(filesize) <= app.config["MAX_IMAGE_FILESIZE"]:
        return True
    else:
        return False
    
    
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1


UPLOAD_FOLDER = './static/assets/uploads/'
CSV_FOLDER = './static/csv/'
# VIDEO_FOLDER = './static/assets/videos/'
DETECTION_FOLDER = './static/assets/detections/'
METADATA_FOLDER = './static/metadata/'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['CSV_FOLDER'] = CSV_FOLDER
app.config['DETECTION_FOLDER'] = DETECTION_FOLDER


IMAGE_ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def allowed_file_image(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in IMAGE_ALLOWED_EXTENSIONS



def file_type(path):
    filename = path.split('/')[-1]
    if allowed_file_image(filename):
        filetype = 'image'
    else:
        filetype = 'invalid'
    return filetype



# @app.route('/')
@app.route(base_url + 'webapp')
def webapp():
    resp = make_response(render_template("upload-file.html"))
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp


@app.route(base_url + '/analyze', methods=['POST', 'GET'])
@cross_origin(supports_credentials=True)
def analyze():
    if request.method == 'POST':
        out_name = None
        filepath = None
        filename = None
        filetype = None
        csv_name1 = None
        csv_name2 = None

        print("File: ", request.files)

        if 'upload-button' in request.form:
            # Get uploaded file

            f = request.files['file']
            ori_file_name = secure_filename(f.filename)
            _, ext = os.path.splitext(ori_file_name)

            filetype = file_type(ori_file_name)

            if filetype == 'image':
                # Get cache name by hashing image
                data = f.read()
                filename = hashlib.sha256(data).hexdigest() + f'{ext}'

                # Save file to /static/uploads
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

                np_img = np.fromstring(data, np.uint8)
                img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
                cv2.imwrite(filepath, img)


        # Get all inputs in form
        iou = request.form.get('threshold-range')
        confidence = request.form.get('confidence-range')
        model_types = request.form.get('model-types')
        enhanced = request.form.get('enhanced')
        ensemble = request.form.get('ensemble')
        ensemble = True if ensemble == 'on' else False
        enhanced = True if enhanced == 'on' else False
        model_types = str.lower(model_types)
        min_conf = float(confidence)/100
        min_iou = float(iou)/100

        if filetype == 'image':
            # Get filename of detected image

            out_name = "Image Result"
            output_path = os.path.join(
                app.config['DETECTION_FOLDER'], filename)

            filename = get_prediction(
                filepath,
                output_path,
                model_name=model_types,
                ensemble=ensemble,
                min_conf=min_conf,
                min_iou=min_iou,
                enhance_labels=enhanced)

        elif filetype == 'video':
            # Get filename of detected video

            out_name = "Video Result"
            output_path = os.path.join(
                app.config['DETECTION_FOLDER'], filename)

            filename = get_video_prediction(
                filepath,
                output_path,
                model_name=model_types,
                min_conf=min_conf,
                min_iou=min_iou,
                enhance_labels=enhanced)
        else:
            error_msg = "Invalid input url!!!"
            return render_template('detect-input-url.html', error_msg=error_msg)

        filename = os.path.basename(filename)
        csv_name, _ = os.path.splitext(filename)

        csv_name1 = os.path.join(
            app.config['CSV_FOLDER'], csv_name + '_info.csv')
        csv_name2 = os.path.join(
            app.config['CSV_FOLDER'], csv_name + '_info2.csv')

        if 'url-button' in request.form:
            return render_template('detect-input-url.html', out_name=out_name, fname=filename, filetype=filetype, csv_name=csv_name1, csv_name2=csv_name2)

        elif 'webcam-button' in request.form:
            return render_template('detect-webcam-capture.html', out_name=out_name, fname=filename, filetype=filetype, csv_name=csv_name1, csv_name2=csv_name2)

        return render_template('detect-upload-file.html', out_name=out_name, fname=filename, filetype=filetype, csv_name=csv_name1, csv_name2=csv_name2)

    return redirect('/')


@app.route(base_url + '/api', methods=['POST'])
def api_call():
    if request.method == 'POST':
        response = {}
        if not request.json or 'url' not in request.json:
            response['code'] = 404
            return jsonify(response)
        else:
            # get the base64 encoded string
            url = request.json['url']
            filename, filepath = download(url)

            model_types = request.json['model_types']
            ensemble = request.json['ensemble']
            min_conf = request.json['min_conf']
            min_iou = request.json['min_iou']
            enhanced = request.json['enhanced']

            output_path = os.path.join(
                app.config['DETECTION_FOLDER'], filename)

            get_prediction(
                filepath,
                output_path,
                model_name=model_types,
                ensemble=ensemble,
                min_conf=min_conf,
                min_iou=min_iou,
                enhance_labels=enhanced)

            with open(output_path, "rb") as f:
                res_im_bytes = f.read()
            res_im_b64 = base64.b64encode(res_im_bytes).decode("utf8")
            response['res_image'] = res_im_b64
            response['filename'] = filename
            response['code'] = 200
            return jsonify(response)

    return jsonify({"code": 400})


@app.after_request
def add_header(response):
    # Include cookie for every request
    response.headers.add('Access-Control-Allow-Credentials', True)

    # Prevent the client from caching the response
    if 'Cache-Control' not in response.headers:
        response.headers['Cache-Control'] = 'public, no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '-1'
    return response


#@app.route('/')
@app.route(base_url)
def Homepage():
    return render_template('Homepage.html')


#@app.route('/')
@app.route(base_url + "/Our-Team")
def OurTeam():
    return render_template('Our-Team.html')


#@app.route('/', methods=['POST'])
@app.route(base_url, methods=['POST'])
def home_post():
    # check if the post request has the file part
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)

    file = request.files['file']
    # if user does not select file, browser also
    # submit an empty part without filename
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return redirect(url_for('results', filename=filename))
    
    if "filesize" in request.cookies:
        if not allowed_image_filesize(request.cookies["filesize"]):
            print("Filesize exceeded maximum limit")
            return redirect(request.url)
    
    
    else:
        flash('Allowed image types are -> png, jpg, jpeg, gif')
        return redirect(request.url)
#         


#@app.route('/uploads/<filename>')
@app.route(base_url + '/uploads/<filename>')
def results(filename): 
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    img = load_img(image_path, target_size=(299, 299))
    img = img_to_array(img)
    img = img / 255
    img = np.expand_dims(img,axis=0)
    res = classify(img)
    return render_template('results.html', filename=filename, labels = res)
    
    

       

#@app.route('/files/<path:filename>')
@app.route(base_url + '/files/<path:filename>')
def files(filename):
    return send_from_directory(UPLOAD_FOLDER, filename, as_attachment=True)

if __name__ == "__main__":
    '''
    coding center code
    '''
    # IMPORTANT: change the cocalcx.ai-camp.dev to the site where you are editing this file.
    website_url = 'coding.ai-camp.dev'
    print(f"Try to open\n\n    https://{website_url}" + base_url + '\n\n')

    # remove debug=True when deploying it
    app.run(host = '0.0.0.0', port=port, debug=True)
    import sys; sys.exit(0)

    '''
    cv scaffold code
    '''
    # Only for debugging while developing
    # app.run(port=80, debug=True)

