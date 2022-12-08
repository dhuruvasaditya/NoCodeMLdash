import streamlit as st
import sys,time
from datetime import timedelta
sys.path.append("..")
from automldash import utils
from transformers import AutoTokenizer
from transformers import AutoTokenizer,AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch
from transformers import get_scheduler
from tqdm.auto import tqdm

def lstm_train(data):
	st.write("Train a LSTM model for classification")
	utils.looper()
	
def transformers_train(STATS,dataset,model_name):
	st.write("Train a Transformer model for classification")
	
	def tokenize_function(examples):
		return tokenizer(examples["input"], padding="longest", truncation=True)

	def download_n_tokenize():
		global model,tokenizer
		#cach_dir
		model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=STATS["num_labels"])#,cache_dir="")
		tokenizer = AutoTokenizer.from_pretrained(model_name)
		tokenized_datasets = dataset.map(tokenize_function, batched=True)
		tokenized_datasets = tokenized_datasets.remove_columns(["input"])
		tokenized_datasets = tokenized_datasets.rename_column("output", "labels")
		tokenized_datasets.set_format("torch")
		return tokenized_datasets


	model_card = {"Tiny-BERT":"prajjwal1/bert-tiny",
	"Distil-BERT":"distilbert-base-uncased",
	"BERT":"bert-base-uncased",
	"RoBERTA":"roberta-base"}
	model_name = model_card[model_name]

	num_epochs = 5
	with st.spinner("Downloading pre-trained model to fine-tune"):
		tokenized_datasets = download_n_tokenize()

	train_dataloader = DataLoader(tokenized_datasets["train"], shuffle=True, batch_size=8)
	eval_dataloader = DataLoader(tokenized_datasets["test"], batch_size=8)

	print("here")
	optimizer = AdamW(model.parameters(), lr=5e-5)
	num_training_steps = num_epochs * len(train_dataloader)
	lr_scheduler = get_scheduler(
		name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
	)

	print("here81")
	device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
	model.to(device)

	# progress_bar = tqdm(range(num_training_steps))
	t0 = time.time()
	model.train()
	my_bar = st.progress(0)
	stats = st.empty()
	iterr,run_loss = 0,0
	for epoch in range(num_epochs):
		for batch in train_dataloader:
			time.sleep(0.5)
			batch = {k: v.to(device) for k, v in batch.items()}
			outputs = model(**batch)
			loss = outputs.loss
			loss.backward()
			loss_value = loss.item()
			run_loss += loss_value
			optimizer.step()
			lr_scheduler.step()
			optimizer.zero_grad()

			iterr += 1
			percent = iterr/num_training_steps
			sec = int(time.time()-t0)
			est = int((1/percent)*sec)
			my_bar.progress(percent)
			stats.text(f"⏳ Current loss: {loss_value:.4f}. Avg loss till now: {run_loss/iterr:.4f}. Time elapsed: {timedelta(seconds = sec)}. Est time: {timedelta(seconds = est)} ---{percent*100:.2f}%")
			# progress_bar.update(1)
	st.success("Model Trained successfully.")



