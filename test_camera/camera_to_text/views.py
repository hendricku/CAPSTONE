 
from django.shortcuts import render
from django.http import HttpResponse
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from django.conf import settings
from django.core.files.storage import FileSystemStorage
import cv2
from django.http import StreamingHttpResponse
from gtts import gTTS

from django.http import JsonResponse
import io
import pyttsx3
import speech_recognition as sr

# Load the saved model
model = tf.keras.models.load_model('static/model/model_fruits.keras')

class_labels = ['Apple', 'Banana', 'Grape', 'Mango', 'Strawberry'] ################### class name connected sa training class palitan ang dapat palitan ###################

def index(request):
    return render(request, "index.html",{})

def gen_frames():
    camera = cv2.VideoCapture(0)
    while True:
        success,frame = camera.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def video_feed(request):
    return StreamingHttpResponse(gen_frames(), content_type='multipart/x-mixed-replace; boundary=frame')


def capture_image(request):
    camera = cv2.VideoCapture(0)
    success, frame = camera.read()
    if success:
        image_path = os.path.join(settings.STATICFILES_DIRS[0], 'image.jpg')
        cv2.imwrite(image_path, frame)
        return render(request, 'index.html', {'image_url': 'image.jpg'})
    else:
        return render(request, 'index.html', {})


def perform_fruit_detection(request):

    if request.method == 'POST':
        imageName = request.POST.get('image', '')
        image_path = os.path.join(settings.STATICFILES_DIRS[0], imageName)
        print(image_path)

        img = image.load_img(image_path, target_size=(150, 150))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.

        img_array = np.concatenate([img_array, np.ones((img_array.shape[0], img_array.shape[1], img_array.shape[2], 1))], axis=-1) ### pag may error try to comment this
                                                                               
        predictions = model.predict(img_array)

        predicted_class = np.argmax(predictions)
        predicted_probability = predictions[0][predicted_class]

        if predicted_class >= len(class_labels) or predicted_probability < 0.5:
            return render(request, 'index.html', {'extracted_text': 'Image not identified or recognized by the model.', 'image_url': imageName})
        else:
            predicted_label = class_labels[predicted_class]
            accuracy = "{:.2f}%".format(predicted_probability * 100)
           
            # tts_text = "This is a {}.".format(predicted_label)
            tts_text = "This is a {}  with {:.2f}% predicted accuracy .".format(predicted_label, predicted_probability * 100) 
            tts = gTTS(text=tts_text, lang='en')
            static_folder = os.path.join(settings.STATICFILES_DIRS[0])
            os.makedirs(static_folder, exist_ok=True)
            output_path = os.path.join(static_folder, 'output.mp3')
            tts.save(output_path)
            return render(request, 'index.html', {'predicted_class': predicted_label, 'accuracy': accuracy, 'image_url': imageName, 'show': True, 'extracted_text': tts_text})
    else:
        return render(request, 'index.html', {'extracted_text': 'No image uploaded!'})

def save_image(request):
    if request.method == 'POST' and 'image' in request.FILES:
        uploaded_image = request.FILES['image']
        fs = FileSystemStorage(location=settings.STATICFILES_DIRS[0])

        existing_image_path = os.path.join(settings.STATICFILES_DIRS[0], 'saved_image.png')
        if os.path.exists(existing_image_path):
            os.remove(existing_image_path)

        filename = fs.save('saved_image.png', uploaded_image)
        image_url = fs.url(filename)
        print(filename)
        return render(request, 'index.html', {'image_url': filename})
    else:
        return render(request, 'index.html', {})



def real_time(request):
    return render(request, "r_detection.html",{})


####################################################################### this function is used in r_detection.html for real time prediction
def classify_image(request):
    if request.method == 'POST' and request.FILES['image']:
        uploaded_image = request.FILES['image']
        image_name = uploaded_image.name  # get the name of the uploaded image
        # read the content of the uploaded file
        image_content = uploaded_image.read()
        # create an in-memory file-like object from the content
        image_stream = io.BytesIO(image_content)
        img = image.load_img(image_stream, target_size=(150, 150))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.
        img_array = np.concatenate([img_array, np.ones((img_array.shape[0], img_array.shape[1], img_array.shape[2], 1))], axis=-1)  ### pag may error try to comment this
        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions)
        predicted_class = class_labels[predicted_class_index]
        predicted_probability = float(predictions[0][predicted_class_index]) 
        accuracy = "{:.2f}%".format(predicted_probability * 100)
        text = "This is a {} with {:.2f}% predicted accuracy .".format(predicted_class, predicted_probability * 100) 

        return JsonResponse({'result': predicted_class, 'accuracy': accuracy, 'image_name': image_name, 'message': text})
    else:
        return JsonResponse({'error': 'No image uploaded.'})


# Initialize speech recognition
recognizer = sr.Recognizer()

# Function to convert speech to text
def speech_to_text(request):

    try:
        with sr.Microphone() as source:
            print("Listening...")
            audio = recognizer.listen(source)

        print("Recognizing...")
        query = recognizer.recognize_google(audio)
        print("You asked:", query)
        response_text = answer_query(query)
        return JsonResponse({'response': response_text})
    
    except sr.UnknownValueError:
        return JsonResponse({'error': 'Sorry, could not understand audio.'})
    except sr.RequestError as e:
        return JsonResponse({'error': f'Could not request results from Google Speech Recognition service: {e}'})
    


# Function to respond to the query
def answer_query(query):
    response_text = ""

    # Define some example queries and responses

   
    responses = {
        "how are you": "I'm just a computer program, but I'm functioning properly, thank you for asking!",
        "what are you": "I'm an AI assistant. You can call me Henius The AI",
        "what is the benefits of eating fruits": "Fruits are a great source of minerals and vitamins, they reduce the risk of heart-related diseases and cancer, they regulate blood cholesterol levels, reduce the risk of obesity and type 2 diabetes in individuals, improve bowel movements and prevent constipation, regulate blood pressure, promote weight loss, aid digestion, promote skin and hair health, hydrate the body, and boost immune system.",
        "scientific name of strawberry": "a widely grown hybrid species of the genus Fragaria, collectively known as the strawberries, which are cultivated worldwide for their fruit. ",
        "scientific name of grapes": "There are many species of grapes, and all of these belong to the genus Vitis. One of the most widely consumed types of grapes in the world is the common grapevine (Vitis vinifera).",
        "scientific name of apple": " is a round, edible fruit produced by an apple tree (Malus spp., among them the domestic or orchard apple; Malus domestica). Apple trees are cultivated worldwide and are the most widely grown species in the genus Malus.",
        "scientific name of banana":" is an elongated, edible fruit – botanically a berry[1] – produced by several kinds of large herbaceous flowering plants in the genus Musa. In some countries, cooking bananas are called plantains, distinguishing them from dessert bananas. "
        # Add more queries and responses as needed
    }


    # Check if the query matches any predefined responses
    for key in responses:
        if key in query.lower():
            response_text = responses[key]
            break

    # If no predefined response is found, provide a default response
    if not response_text:
        response_text = "I'm sorry, I don't understand your question."

    
    text_to_speech(response_text)

# Function to convert text to speech
def text_to_speech(text):

    engine = pyttsx3.init()

    engine.say(text)
    engine.runAndWait()

    engine.stop()



