import os
import torch

##### Uncomment following lines to run this module as a script
# import sys
# from pathlib import Path
# sys.path.append(str(Path(__file__).resolve().parents[1]))
# from helpers import predict
# from nmt import NMT
# from data.vocab import Vocab

##### Uncomment following lines when importing this module
from .helpers import predict
from .nmt import NMT
from data.vocab import Vocab

def make_prediction(model_name, input_string):
    device = torch.device("cpu")
    mdl = NMT.load(os.path.join("models", model_name))
    mdl.to(device)
    mdl.device = device
    mdl.decoder.device = device
    return predict(mdl, input_string)

if __name__ == "__main__":
    # Select which model to use for predictions
    model_name = "mod_a.ckpt"
    # Choose desired input string
    input_string = "where are you from?"
    # Make prediction
    print(make_prediction(model_name, input_string))