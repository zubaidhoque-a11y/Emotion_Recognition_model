from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from fastapi import FastAPI ,HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse 
import numpy as np
import re
from pydantic import BaseModel , Field 
from keras.models import load_model
import pickle


model_path = "BiGRU_model.keras"
tokenizer_path = "tokenizer.pkl"

max_sequence_length = 50

emotion_labels = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']

Emotion_emojis = {
    "sadness": "😢",
    "joy": "😄",
    "love": "❤️",
    "anger": "😠",
    "fear": "😨",
    "surprise": "😲",
}
#Preprocess the upcoming text
def preprocess_text(text : str) ->str :
    text = text.lower()
    text = re.sub(r"'" , "" , text)
    text = re.sub(r"[^a-z0-9\s]" , " " , text )
    text = re.sub(r"\s+" , " " , text).strip()
    return text

#Request and Response Schemas
class TextInput(BaseModel):
    text : str = Field(
        ...,
        min_length = 1,
        max_length = 2000,
        description = "The sentence to analyze" ,
        json_schema_extra = {"example":"I am feeling so excited"}
        )

class PredictionResponse(BaseModel):
    text: str
    predicted_emotion : str 
    confidence : float
    all_probabilities : dict[str , float]

class HealthResponse(BaseModel):
    status : str
    model_loaded: bool

#Model Loading and Lifespan management 
#Load the model and tokenizer once the server starts up.
dl_model ={}

@asynccontextmanager
async def lifespan(app:FastAPI):
    print("Loading the model and tokenizer")
    dl_model['BiGRU'] = load_model(model_path)          #BiGRU model
    with open (tokenizer_path, 'rb') as file:
        dl_model['Tokenizer'] = pickle.load(file)      #Tokenizer model
    print("Model are loaded Successfully")
    yield #Pause, model is loaded and server is running at this point the model will request and respond 
    dl_model.clear()
app =FastAPI(
    lifespan = lifespan
 )

#Mount the static files (frontend end files )to the FastAPI app
app.add_middleware(
    CORSMiddleware,
    allow_origins = ['*'],
    allow_credentials = True,
    allow_methods = ['*'],
    allow_headers = ['*'],
)

app.mount('/static' , StaticFiles(directory = 'static') , name = 'static')


#API endpoints 
#1.Serve UI at homepage
@app.get ('/' , include_in_schema = False )
def serve_ui():
    return FileResponse('static/index.html')

#2.Health Check Endpoint('/health')
@app.get('/health' , response_model = HealthResponse)
def health_check():
    return HealthResponse (status = "Server is running" , model_loaded = bool(dl_model))

#3.Prediction Emotion Endpoint 
@app.post('/predict' , response_model = PredictionResponse)
def predict_emotion(text_input:TextInput):
    #clean the input sentences
    BiGRU_model = dl_model.get('BiGRU')
    tokenizer_model = dl_model.get('Tokenizer')
    if BiGRU_model is None or tokenizer_model is None:
        raise HTTPException(status_code = 503 , detail ="Model is not laoded yet.Please try again later." )
    cleaned_text  = preprocess_text(text_input.text)
    tokenized_text = tokenizer_model.texts_to_sequences([cleaned_text])
    padded_sequences = pad_sequences(
        tokenized_text,
        maxlen = max_sequence_length,
        padding = "post",
        truncating = "post"
    )

    probabilities = BiGRU_model.predict(padded_sequences)[0]
    top_emotion_index = int(np.argmax(probabilities))

    all_probabilities = {
        label :float(prob) for label, prob in zip(emotion_labels,probabilities)
    }
    return PredictionResponse(
        text = text_input.text,
        predicted_emotion = emotion_labels[top_emotion_index],
        confidence = float(probabilities[top_emotion_index]),
        all_probabilities = all_probabilities,
    )

    
