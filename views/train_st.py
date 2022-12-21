import streamlit as st
import time
from train_files import classifier,seq2seq
import utils
	
def train():
	#Show UI for choosing options
	placeholder = st.empty()
	with placeholder.container():
		method = st.radio(
			"Choose task for training: ",
			("Sequence Classification","Seq2seq")
			)
		if method=="Sequence Classification":
			model = st.selectbox(
				"Model architecture",
				("Tiny-BERT","Bi-LSTM","Distil-BERT","BERT","RoBERTA"))
		else:
			model = st.selectbox(
				"Model architecture",
				("Bi-LSTM", "GPT", "T5","BART"))
				
		### second level options
		if method=="Seq2seq":
				st.write("CSV file headers should be 'input' and 'output' for the seq2seq task")
		else:
			st.write("CSV file headers should be 'input' and 'label' for the classification task\
				 (label can be a string(negative/positive) or integer(1/0)")
		url = st.text_input("Enter URL of dataset in CSV or JSON form",r"D:\TSApy\NoCodeMLdash\data\sequence\seq_sarcasm_train.json")
		split = st.number_input("Percentage of data for training (rest will be taken as test)",
		 min_value=60, max_value=90, value=70, step=10)
		submitted = st.button("Submit")
	
	#Obtain selected option and choose training model
	if submitted:
		placeholder.empty()
		if method == "Seq2seq":
			if model == "Bi-LSTM":
				seq2seq.lstm_train(url,split)
			else:
				STATS,dataset = utils.get_dataset_hf(url,split)
				seq2seq.transformers_train(STATS,dataset,model)
		else:
			if model == "Bi-LSTM":
				classifier.lstm_train(*utils.get_dataset_torchtext(url,split))
			else:
				classifier.transformers_train(*utils.get_dataset_hf(url,split),model)
	# return