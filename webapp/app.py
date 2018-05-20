import io, traceback

from flask import Flask, request, g
from flask import send_file
from flask_mako import MakoTemplates, render_template
from plim import preprocessor

from PIL import Image, ExifTags
from scipy.misc import imresize
import numpy as np
from keras.models import load_model
import tensorflow as tf

app = Flask(__name__, instance_relative_config=True)
mako = MakoTemplates(app)
app.config['MAKO_PREPROCESSOR'] = preprocessor
app.config.from_object('config.ProductionConfig')

model = load_model('./model/main_model.hdf5', compile=False)
graph = tf.get_default_graph()

def ml_predict(image):
    with graph.as_default():
        prediction = model.predict(image[None, :, :, :])
    prediction = prediction.reshape((224,224, -1))
    return prediction

def rotate_by_exif(image):
    try :
        for orientation in ExifTags.TAGS.keys() :
            if ExifTags.TAGS[orientation]=='Orientation' : break
        exif=dict(image._getexif().items())
        if not orientation in exif:
            return image

        if   exif[orientation] == 3 :
            image=image.rotate(180, expand=True)
        elif exif[orientation] == 6 :
            image=image.rotate(270, expand=True)
        elif exif[orientation] == 8 :
            image=image.rotate(90, expand=True)
        return image
    except:
        traceback.print_exc()
        return image

THRESHOLD = 0.5
@app.route('/predict', methods=['POST'])
def predict():
    image = request.files['file']
    image = Image.open(image)
    image = rotate_by_exif(image)
    resized_image = imresize(image, (224, 224)) / 255.0

    prediction = ml_predict(resized_image[:, :, 0:3])
    prediction = imresize(prediction[:, :, 1], (image.height, image.width))
    prediction[prediction>THRESHOLD*255] = 255
    prediction[prediction<THRESHOLD*255] = 0

    transparent_image = np.append(np.array(image)[:, :, 0:3], prediction[: , :, None], axis=-1)
    transparent_image = Image.fromarray(transparent_image)
   
    width = 132
    ratio = (width / float(transparent_image.size[0]))
    height = int(float(transparent_image.size[1]) * float(ratio))
    transparent_image = transparent_image.resize((width, height)) 

    byte_io = io.BytesIO()
    res_img = Image.new("RGB", (396, 340), (255, 255, 255))
    res_img.paste(transparent_image, (0, 0), transparent_image)
    res_img.paste(transparent_image, (132, 0), transparent_image)
    res_img.paste(transparent_image, (264, 0), transparent_image)
    res_img.paste(transparent_image, (0, 170), transparent_image)
    res_img.paste(transparent_image, (132, 170), transparent_image)
    res_img.paste(transparent_image, (264, 170), transparent_image)
    res_img.save(byte_io, 'PNG')
    byte_io.seek(0)
    return send_file(byte_io, mimetype='image/png')

@app.route('/')
def homepage():
    return render_template('index.html.slim', name='mako')


if __name__ == '__main__':
    app.run(host='0.0.0.0')
