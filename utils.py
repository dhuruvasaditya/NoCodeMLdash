import pandas as pd
import streamlit as st
import os,time
from datasets import Dataset

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

def get_dataset_hf(url,split):
	df = load_data(url)
	STATS = {}
	STATS["num_labels"] = len(df["output"].unique())
	dataset = Dataset.from_pandas(df)
	st.success("Data loaded succesfully.")
	dataset = dataset.train_test_split(train_size=split/100,seed=13)
	STATS["train_rows"] = dataset["train"].num_rows
	STATS["test_rows"] = dataset["test"].num_rows
	STATS["total_rows"] = len(df)
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
