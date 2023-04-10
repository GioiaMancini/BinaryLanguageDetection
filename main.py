# This code defines a FastAPI app instance and creates two endpoints.

# The first endpoint is a GET request to the root URL (/). 
# When a user navigates to the root URL, the root function is called, 
# which returns an HTML template called index.html.

# The second endpoint is a POST request to the /predict URL.
# When a user submits a form on the homepage, the detect_italian function is called. 
# This function loads a pre-trained SVM model from a pickle file, 
# uses the model to make predictions on the user's input text, 
# and returns the prediction as a JSON object.


# Import necessary libraries and modules
from fastapi import FastAPI, Form, Request  # FastAPI framework
import pickle  # For loading trained model
import scipy  # Required by the trained model
from fastapi.templating import Jinja2Templates  # For rendering HTML templates
from pydantic import BaseModel  # For defining the input data model
import regex as re
import uvicorn

# Create a FastAPI app instance
app = FastAPI()

# Define the input data model using Pydantic
class isItalian(BaseModel):
    text: str

# Create a Jinja2Templates instance and set the templates directory
templates = Jinja2Templates(directory='')

# GET HTTP method; # Define a root endpoint that returns the index.html template
@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})


# Define a predict endpoint that receives POST requests and returns predictions
@app.post('/predict')
async def detect_italian(corpus: isItalian):
    # Extract the text data from the input model
    data = corpus.dict()
    data_in = data['text']
    #print(data_in)
    
    data_in = re.sub(r'[\.!@#$(),\n"%^*?\+\-\':;~`0-9\=\[\]]', ' ', data_in) # removing special characters, symbols and numbers
    data_in = re.sub(r'http[s]?\://\S+|www\.\S+', ' ', data_in) # removing URLs
    data_in = re.sub(r'<.*?>', ' ', data_in) # removing html tags
    data_in = re.sub(r'\s+'," ", data_in) # removing extra large spaces
    data_in = data_in.lower()

    # Load the pre-trained SVM model
    classifier = pickle.load(open('MultinomialNB_2023-04-09_20-58-01.pkl','rb'))
    # Use the trained model to make predictions on the input data
    prediction = classifier.predict([data_in])
    
    # Return the prediction as a JSON object
    return {'prediction': prediction.tolist()[0]}

# use Uvicorn to run the server on localhost:5000 port
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=5000)