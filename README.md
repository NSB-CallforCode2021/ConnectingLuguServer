# Connecting Lugu Server program

#### Introduction
Python Flask API service for ConnectingLugu

This project contains:<br>
    1. Flask restful api service<br>
    2. Health check model for rice: This model can check whether the rice is in disease and what is the disease.<br>
    Model Architecture:
    
    ![image](https://user-images.githubusercontent.com/18240201/127437702-c5dba3fe-9214-4829-9ede-e75299e749e7.png)
    
    3. Prediction of rice quality and price (in progress): We use the mean score during rice growth to evalute the quality. 
    We will build and traine CNN-LSTM model for rice quality evaluation in next step.
  

#### Requirements
- Python 3.8.5
- numpy 1.19.4
- Flask 2.0.1
- tensorflow 2.5.0
- Werkzeug 2.0.1
- scikit-learn 0.24.1
- pandas 1.2.4

#### Usage
First, install prerequisites with: `$pip install -r requirements.txt`<br>

To train a model for rice health check:
    `$python rice_health_model.py`
    
Restful APIs:<br>
Online demo url: http://47.100.80.164/<br>
1. POST /upload<br>
   Example: 
   Response body:
   `
    {
      "score": 80,
      "disease": "brown spot"
    }
   `
2. POST /evaluateprice<br>

 Response body:
   `
    {
      "score": 85.5,
      "level": "B"
    }
   `


