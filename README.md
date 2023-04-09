# Binary Language Detection
Binary Language Detection Machine Learning project

This repository contains an implementation of a Binary Language Detection problem aimed at classifying sentences as either Italian or non-Italian. The proposed model is implemented in a Jupyter notebook that can be run locally or through Google Colaboratory. By downloading the dataset and running the cells, it is possible to train a new model and storing it in a pickle file, as well. The notebook includes data preprocessing, model selection and training, and performance evaluation. The trained model is selected with a GridSearchCV algorithm that enables the search for the best estimator for our task. The resulting model achieved high performance on the test set, with an accuracy score of 0.98, as well as strong precision, recall, and F1 scores.

You can directly try to detect the language of a sentence through a RESTful API, which allows to make a POST inference call. The final solution exposes the call to a service predict on port `localhost:5000/predict` via a web interface or curl command.

To further enhance the usability of this project, a dockerfile and a Docker image [link](https://hub.docker.com/r/gioiamancini/binarylanguagedetection) on Docker Hub have been added to the repository. The Docker image allows users to easily deploy the language detection model without having to worry about setting up the necessary environment.

# Deploy of the model

The final solution has been deployed as local endpoint to expose a REST API (POST) inference call to a service predict at http://localhost:5000/predict.

## Local usage

In order to test the solution on your local machine, you can follow these instructinos.

1. Clone this repository to your local machine using: `git clone https://github.com/your-username/your-repository.git`
2. Navigate to the root directory of the cloned repository using the command line interface.
3. Install the required dependencies by running the following command: `pip install -r requirements.txt`
4. Start the server:
  - you can run it through python command: `python main.py`
  - or by using Uvicorn: `uvicorn main:app --reload --port 5000`
  Note that you can run it in a different port by: `uvicorn main:app --reload --port <int>`, specifying the port number.
5. The server should start running and you should see the following message: `* Running on http://localhost:5000/ (Press CTRL+C to quit)`
6. Now, you can open a web browser and navigate to `http://localhost:5000` to use the API via a web page, you should see a window like this:
<img src="https://github.com/GioiaMancini/BinaryLanguageDetection/blob/main/webpage.jpg" width="420">

7. You can use a `curl` to make requests via the terminal: 
```
curl -X "POST" "http://127.0.0.1:5000/predict" -H "accept: application/json" -H "Content-Type: application/json" -d "{\"text\": \"Is this an italian sentence?\"}"
```
N.B.: If you want to test sentences with accented characters, please use a json file as the one provided in the main folder (`curl_request.json`).

If you want to use the docker image, you can either build it from scratch given the provided dockerfile or you can directly use the one provided at: .
To build the image:
1. `docker build -t binarylanguagedetection .`
3. `docker run -dp 5000:5000 -ti --name LanguageDetectionContainer binarylanguagedetection`

Now you can test the model going to Docker Desktop>Containers>Ports and clicking on port 5000.

In case you want to use pull the [image from Docker Hub](https://hub.docker.com/r/gioiamancini/binarylanguagedetection):
`docker pull gioiamancini/binarylanguagedetection` and run it with: `docker run -dp 5000:5000 -ti gioiamancini/binarylanguagedetection`

# Usage - Machine Learning Model

## Dataset - Kaggle

### Prerequisites

1. You must have Python 3 installed on your system.
2. You must have a Kaggle account.

### Download

1. Install the Kaggle API by running this command in your terminal:
`pip install kaggle`

2. Log in to your Kaggle account by following these steps:
  - Go to your Kaggle account and click on "Create New API Token". This will donwload a JSON file with your API credentials.
  - Note that if you are using the Kaggle CLI tool, place this file in the location ~/.kaggle/kaggle.json on Linux, OSX, and other UNIX-based operating systems, and at     C:\Users\<Windows-username>\.kaggle\kaggle.json on Windows. If you are using the Kaggle API directly, where you keep the token doesn’t matter, so long as you are able   to provide your credentials at runtime.
  
3. Download the dataset using this command:
`kaggle datasets download -d basilb2s/language-detection`

4. Unzip the .csv file.

### Additional Notes

1. You can download the dataset to any directory you want by changing the path in step 3.
2. If you don't have a Kaggle account, you can download the dataset from the Kaggle website directly by first creating an account, then skip 2. and 3 and unzip the downloaded file.

## Dataset - Kaggle on Google Colab

### Prerequisites

1. You must have a Kaggle account.
2. You must mount your drive in Colab.

### Download

1. Install Kaggle API `!pip install kaggle`
2. Create and download your API token as shown in step 2. of previous section. Note that in this case the kaggle.json file must be in the root of your Google Drive.
3. Mount your drive: 
```
from google.colab import drive
drive.mount('/content/drive')
```
4. Move your kaggle.json file in this directory:
```
!mkdir -p /root/.kaggle
!cp '/content/drive/My Drive/kaggle.json' /root/.kaggle/
```
5. Go in the directory in which you want to download the dataset, for example:
`%cd /content/drive/MyDrive/Datasets`

6. Finally, you can download the dataset and unzip it:
```
!kaggle datasets download -d basilb2s/language-detection
!unzip language-detection.zip
```

### ML model 

Once you have downloaded the dataset and you have the right set-up, you can start exploring the text preprocessing stage, the model selection and the final model evaluation.

#### Preprocessing and cleaning

As you can see from the notebook, at first you have to import the required dependencies, which involve preprocessing and machine learning models' libraries. Then, you have to load the csv file of the dataset into a pandas DataFrame. Some functions for data visualization are described, here are some examples of the DataFrame visualization:

|    | Text              |                   Language   |
|---:|:------------------|:-----------------------------|
|  0 | Nature, in the broadest sense, is the natural, physical, material world or universe.   | English    |
|  1 | "Nature" can refer to the phenomena of the physical world, and also to life in general.| English    |
|  2 | The study of nature is a large, if not the only, part of science.                      | English    |
|  3 | Although humans are part of nature, human activity is often understood as a separate category from other natural phenomena.                                                                                    | English    |
|  4 | [1] The word nature is borrowed from the Old French nature and is derived from the Latin word natura, or "essential qualities, innate disposition", and in ancient times, literally meant "birth".                | English    |

Here you can see the number of sentences per language in the dataset:

<img src="https://github.com/GioiaMancini/BinaryLanguageDetection/blob/main/docs/nSentencesPerLanguage.png" width="420">

Here is the number of words per language in the dataset:

<img src="https://github.com/GioiaMancini/BinaryLanguageDetection/blob/main/docs/nWordsPerLanguage.png" width="420">

After visualizing information about the dataset, we have to perform data cleaning on the text data. In particular, we perform regex-based text cleaning operations, including removing special characters, symbols, numbers, URLs, html tags and extra large spaces. Then, the text is transformed to lowercase and finally a list of cleaned text is returned. After this process, the total number of words in the dataset goes from 202335 to 201905.

#### Train and test split

To perform our ML algorithms over our text data, we have to split the dataset into train and test sets. We define a `random_state=42` to allow reproducibility.
The final data are split into 8269 sentences for train set and 2068 for test set.

#### Model Selection

When dealing with 



