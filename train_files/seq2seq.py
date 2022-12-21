import streamlit as st
import sys
sys.path.append("..")
from NoCodeMLdash import utils

def lstm_train(url):
	st.write("Train a LSTM model for seq2sqeq")
	utils.looper()
	
def transformers_train(url,model):
	st.write("Train a Transformer model for seq2sqeq")
	utils.looper()
	