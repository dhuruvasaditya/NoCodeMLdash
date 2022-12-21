import pandas as pd
import streamlit as st
import os,time
from datasets import Dataset
from torchtext.legacy.data import Field, TabularDataset
import torch

def sayhi():
	print("hello")

def load_data(url):
	usecol = ["input","output"]
	if not os.path.exists(url):
		st.error("File path does not exist")
	else:
		if "csv" in url:
			df = pd.read_csv(url,encoding='utf-8',usecols=usecol)
		else:
			df = pd.read_json(url,encoding='utf-8',lines=True)
			df = df[usecol][:100]
		return df	

def get_dataset_torchtext(url,split):
	df = load_data(url)
	STATS = {}
	STATS["num_labels"] = len(df["output"].unique())
	STATS["total_rows"] = len(df)
	del df

	label_field = Field(sequential=False, use_vocab=False, batch_first=True, dtype=torch.float)
	text_field = Field(tokenize='spacy',tokenizer_language = 'en_core_web_sm', lower=True, include_lengths=True, batch_first=True)
	# fields = [('output', label_field), ('input', text_field)]
	fields = {'input':("input",text_field),
			'output':("output",label_field)}
	# TabularDataset
	train_data = TabularDataset(path=url,format='JSON', fields=fields)#, skip_header=True)

	train_data, valid_data = train_data.split(split_ratio=0.7, random_state = random.seed(13))
	STATS["train_rows"] = len(train_data)
	STATS["test_rows"] = len(valid_data)
	return STATS,train_data,text_field,valid_data

def get_dataset_hf(url,split):
	dataset = load_data(url)
	STATS = {}
	STATS["total_rows"] = len(dataset)
	STATS["num_labels"] = len(dataset["output"].unique())
	dataset = Dataset.from_pandas(dataset)
	st.success("Data loaded succesfully.")
	dataset = dataset.train_test_split(train_size=split/100,seed=13)
	STATS["train_rows"] = dataset["train"].num_rows
	STATS["test_rows"] = dataset["test"].num_rows	
	return STATS,dataset

def looper():
	my_bar = st.progress(0)
	stats = st.empty()
	for percent_complete in range(20):
		time.sleep(0.1)
		my_bar.progress(percent_complete + 1)
		stats.text(f"⏳ {percent_complete} seconds have passed")
		if percent_complete%10==0:
			st.write(f"✔️ {percent_complete} seconds have passed 🔥")
	st.success('This is a success message!')
