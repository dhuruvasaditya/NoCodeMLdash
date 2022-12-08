import streamlit as st
import time
from train_files import classifier,seq2seq
def train():
	placeholder = st.empty()
	with placeholder.container():
		method = st.radio(
			"Choose training method aim: ",
			("Seq2seq", "Sequence Classification")
			)
		if method=="Sequence Classification":
			models = st.selectbox(
				"Model architecture",
				("Bi-LSTM", "BERT", "RoBERTA",))
		else:
			models = st.selectbox(
				"Model architecture",
				("Bi-LSTM", "GPT", "T5","BART"))
		submitted = st.button("Submit")
	if submitted:
		placeholder.empty()
		my_bar = st.progress(0)
		stats = st.empty()
		for percent_complete in range(20):
			time.sleep(0.1)
			my_bar.progress(percent_complete + 1)
			stats.text(f"⏳ {percent_complete} seconds have passed")
			if percent_complete%10==0:
				st.write(f"✔️ {percent_complete} seconds have passed")
		#dropdown ker

		st.success('This is a success message!')
		classifier.hida()
		seq2seq.hida()