neural-machine-translation
==============================

An LSTM-based Recurrent Neural Network that translates Modern English into Old (Shakespearean) English

![alt text](https://github.com/markbotros1/neural-machine-translation/blob/main/example.png?raw=true)

Summary
-------
- Encoder-Decoder RNN that takes modern English as input and outputs Old English
- Fronted interface built with Flask to interact with model

Running the project
------------
1. Clone repo
2. Create and activate virtual environment
```
cd path/to/NeuralMachineTranslation

python -m venv project-env
source project-env/bin/activate
```
3. Install requirements
```
pip install -r requirements.txt
```
4. From command line, run app using following
```
python src/app.py
```
5. Paste the generated localhost server url into browser
```
Example: http://127.0.0.1:5000/
```

Improving the model
------------
1. Modify the RNN's architecture found in: ```src/nmt.py```
2. Finetune the model's hyperparameters in: ```src/train_model.py```

Project Organization
------------

    ├── README.md          <- The top-level README for developers using this project.
    ├── datasets           <- Data files used to train and test model
    |
    ├── models             <- Trained and serialized models
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   ├── app.py         <- Runs web application
    │   │
    │   ├── data           <- Helper functions to carry out data preprocessing
    │   │   ├── preprocessing.py
    │   │   └── vocab.py
    │   │
    │   ├── model          <- Modules and scripts to build/train models and make translations
    │   │   ├── encoder.py
    │   │   ├── decoder.py
    │   │   ├── nmt.py
    │   │   ├── train_model.py
    │   │   └── predict_model.py
    │   ├── static         <- Contains JavaScript and CSS files
    │   │   ├── site.css
    │   │   └── input.js   
    │   ├── templates      <- Contains HTML
    │   │   └── index.html             
