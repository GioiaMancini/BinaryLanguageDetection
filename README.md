# Binary Language Detection
Binary Language Detection Machine Learning project


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
