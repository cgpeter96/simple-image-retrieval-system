#!/usr/bin/env python
# -*- coding: utf-8 -*-
from flask import Flask, request, render_template, url_for, send_from_directory
from flask_dropzone import Dropzone
from flask_uploads import UploadSet, IMAGES, configure_uploads, patch_request_class
from gevent.pywsgi import WSGIServer
from werkzeug.utils import secure_filename
from retrieval_images import *
from classify import *

app = Flask(__name__)
app.config['UPLOADED_PHOTOS_DEST'] = './search/'
app.config['UPLOAD_FOLDER'] = './clothes2/'

dropzone = Dropzone(app)
photos = UploadSet('photos', IMAGES)
configure_uploads(app, photos)
patch_request_class(app)  # 限制文件大小


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST' and 'photo' in request.files:
        filename = photos.save(request.files['photo'])
        filepath =photos.path(filename)
        print('=====',filepath)
        print(filename)
        label = classifier(filepath)
        img_path = query(filepath)

        img_urls = [url_for('download', filename=i) for i in img_path]
        # file_url = photos.url(filename)
        # return "%s,%s"%(file_url,filename)
        print(img_urls)
        return render_template('show.html', image_list=img_urls,label=label)
    else:
        return render_template('index.html')
@app.route('/predict',methods=['GET','POST'])
def classify_image():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)


        result =  classifier(file_path)  # Convert to string
        return result
    else:
        return render_template('classify.html')

@app.route('/uploads/<filename>')
def download(filename):
    # 通过浏览器输入指定文件的文件名来下载
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


def query(filename):
    print('deal ',filename)
    img_path = search_images(filename)  # 从search搜索
    return img_path


if __name__ == '__main__':
    # app.run()
    print('starting....')
    WSGIServer(('0.0.0.0', 5000), app).serve_forever()
