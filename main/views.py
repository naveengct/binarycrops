from django.shortcuts import render
from . import urls
from django.http import HttpResponse
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import tensorflow_hub as hub
# Create your views here.
def home(request):
    return render(request,'base.html')
def hello(request):
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODEL_PATH = 'breast_cancer.h5'
    model = models.load_model(os.path.join(BASE_DIR,'main/model_vegetable.h5'),custom_objects={'KerasLayer': hub.KerasLayer})
    img = image.img_to_array(image.load_img(request.FILES.get('img'), target_size=(224, 224, 3))) / 255
    img=np.reshape(img,(1,224,224,3))
    temp=model.predict(img)
    y=np.argmax(temp)
    CAT=['ladies_finger', 'maize', 'carrot', 'sunflower', 'thulasi',
       'tomato', 'pumpkin', 'onion', 'brinjal']
    return render(request, 'base.html', {'res1':CAT[y]})
