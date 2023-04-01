from flask import Flask, json, request, jsonify
import os
import urllib.request
from werkzeug.utils import secure_filename
import pickle
from sklearn.decomposition import PCA
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

path ="/Users/shuvamdas/PycharmProjects/BRAILLE_FINAL/braille_1 2/train/"
files = os.listdir(path)[:5]
print(files)

classes={'a':0, 'b':1, 'c':2}
pca = PCA(.98)

import cv2
x=[]
y=[]

for cl in classes:
  pth = path+cl
  for img_name in os.listdir(pth):
    img = cv2.imread(pth+"/"+img_name,0)
    img = cv2.resize(img, (64, 64))
    x.append(img)
    y.append(classes[cl])


x=np.array(x)
y=np.array(y)

x_new = x.reshape(len(x),-1)

xtrain, xtest, ytrain, ytest = train_test_split(x_new, y, test_size=0.20, random_state=10)
x_train = xtrain/255
x_test = xtest/255

#print(x_train.shape, x_test.shape)
pca = PCA(.98)
xtrain = pca.fit_transform(x_train)
xtest = pca.transform(x_test)

log = LogisticRegression()
log.fit(xtrain, ytrain)

app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))
app.secret_key = "caircocoders-ednalan"

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def main():
    return 'Homepage'


@app.route('/upload', methods=['POST'])
def upload_file():
    # check if the post request has the file part
    if 'files[]' not in request.files:
        resp = jsonify({'message': 'No file part in the request'})
        resp.status_code = 400
        return resp

    files = request.files.getlist('files[]')

    errors = {}
    success = False

    for file in files:
        if file and allowed_file(file.filename):
            import os
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            success = True

        else:
            errors[file.filename] = 'File type is not allowed'

    if success and errors:
        errors['message'] = 'File(s) successfully uploaded'

        resp = jsonify(errors)
        resp.status_code = 500
        return resp
    if success:
        import cv2
        import base64
        import numpy as np
        from io import BytesIO

        import os
        import pandas as pd

        img = cv2.imread("/Users/shuvamdas/PycharmProjects/BRAILLE_FINAL/static/uploads/rtg.jpeg", 0)
        img = cv2.resize(img, (64, 64))
        img = pca.transform(img.reshape(1, -1)/255)


        # Make a prediction
        pred = log.predict(img)[0]
        decode = {0: 'A', 1: 'B', 2: 'C'}
        pred_class = decode[pred]
        return jsonify({'class': pred_class})

    else:
        resp = jsonify(errors)
        resp.status_code = 500
        return resp


if __name__ == '__main__':
    app.run(debug=True)