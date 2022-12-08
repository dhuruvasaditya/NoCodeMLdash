import streamlit as st
import time
# Using object notation
from views import train_st,test_st,ngram_st,ner_st
st.title('NLP AutoML \n', )
st.subheader("by Aditya TS")

# Sidebar options
option = st.sidebar.selectbox('Navigation', 
	["Home",
	"Training",
	"Testing", 
	"N-Gram Analysis", 
	"Parts of Speech Analysis", 
	"Named Entity Recognition",
	"Text Summarizer"])

if option == "Training":
	train_st.train()
elif option == "Testing":
	test_st.test()
elif option == "N-Gram Analysis":
	ngram_st.exec()
elif option == "Named Entity Recognition":
	ner_st.exec()
	



