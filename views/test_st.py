import streamlit as st
import time
from test_files import classifier,seq2seq
def obtain_file(task):
	#obtain files wrt task


	files = ("bert1","bertv2")
	if len(files)==0:
		st.write("No trained models under this category.")
	else:
		models = st.selectbox(
			"Trained models to load",
			files)
	submitted = st.button("Submit")
	return models

def test():
	placeholder = st.empty()
	with placeholder.container():
		method = st.radio(
			"Choose task for testing: ",
			("Seq2seq", "Sequence Classification")
			)
		if method == "Seq2seq":
			model = obtain_file("seq")
		else:
			model = obtain_file("clasf")
		
	if model:
		placeholder.empty()
		st.success('This is a success message!')
		classifier.hida()
		seq2seq.hida()