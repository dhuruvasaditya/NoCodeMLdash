import streamlit as st
import sys,time
from datetime import timedelta
sys.path.append("..")
from NoCodeMLdash import utils
from transformers import AutoTokenizer
from transformers import AutoTokenizer,AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch
from transformers import get_scheduler
from tqdm.auto import tqdm

# Preliminaries

from torchtext.legacy.data import Field, TabularDataset, BucketIterator
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# import seaborn as sns

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
	
def lstm_train(STATS,train,test):
	st.write("Train a LSTM model for classification")
	class LSTM(nn.Module):
		def __init__(self, vocab, dimension=128):
			super(LSTM, self).__init__()
			self.embedding = nn.Embedding(vocab, 300)
			self.dimension = dimension
			self.lstm = nn.LSTM(input_size=300,
								hidden_size=dimension,
								num_layers=1,
								batch_first=True,
								bidirectional=True)
			self.drop = nn.Dropout(p=0.5)

			self.fc = nn.Linear(2*dimension, 1)

		def forward(self, text, text_len):

			text_emb = self.embedding(text)

			packed_input = pack_padded_sequence(text_emb, text_len, batch_first=True, enforce_sorted=False)
			packed_output, _ = self.lstm(packed_input)
			output, _ = pad_packed_sequence(packed_output, batch_first=True)

			out_forward = output[range(len(output)), text_len - 1, :self.dimension]
			out_reverse = output[:, 0, self.dimension:]
			out_reduced = torch.cat((out_forward, out_reverse), 1)
			text_fea = self.drop(out_reduced)

			text_fea = self.fc(text_fea)
			text_fea = torch.squeeze(text_fea, 1)
			text_out = torch.sigmoid(text_fea)
			return text_out
	
	def lstm_train(STATS,train_data,text_field,valid_data):
		train_iter = BucketIterator(train_data, batch_size=32, sort_key=lambda x: len(x.input),
									device=device, sort=True, sort_within_batch=True)
		test_iter = BucketIterator(valid_data, batch_size=32, sort_key=lambda x: len(x.input),
									device=device, sort=True, sort_within_batch=True)
		text_field.build_vocab(train_data, min_freq=3)
		print("Hi da",STATS)
		return len(text_field.vocab),train_iter,test_iter

	def train(model,
			optimizer,
			train_loader,
			valid_loader,
			eval_every,
			num_epochs = 5,
			criterion = nn.BCELoss(),
			best_valid_loss = float("Inf")):
		
		# initialize running values
		running_loss = 0.0
		valid_running_loss = 0.0
		global_step = 0
		train_loss_list = []
		valid_loss_list = []
		global_steps_list = []
		num_training_steps = num_epochs * len(train_loader)
		
		# training loop
		t0 = time.time()
		my_bar = st.progress(0)
		stats = st.empty()
		model.train()
		for epoch in range(num_epochs):
			print("no of steps",len(train_loader))
			for x in train_loader:         
				labels = x.output.to(device)
				text = x.input[0].to(device)
				text_len = x.input[1].to(device)
				output = model(text, text_len)

				loss = criterion(output, labels)
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

				# update running values
				running_loss += loss.item()
				global_step += 1

				# evaluation step
				if global_step % eval_every == 0:
					model.eval()
					with torch.no_grad():                    
						# validation loop
						for y in valid_loader:
							labels = y.output.to(device)
							text = y.input[0].to(device)
							text_len = y.input[1].to(device)
							output = model(text, text_len)

							loss = criterion(output, labels)
							valid_running_loss += loss.item()

					# evaluation
					average_train_loss = running_loss / eval_every
					average_valid_loss = valid_running_loss / len(valid_loader)
					train_loss_list.append(average_train_loss)
					valid_loss_list.append(average_valid_loss)
					global_steps_list.append(global_step)

					# resetting running values
					running_loss = 0.0                
					valid_running_loss = 0.0
					model.train()

					# print progress
					print('Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}, Valid Loss: {:.4f}'
						.format(epoch+1, num_epochs, global_step, num_epochs*len(train_loader),
								average_train_loss, average_valid_loss))
					
					# checkpoint
					if best_valid_loss > average_valid_loss:
						best_valid_loss = average_valid_loss
						# save_checkpoint(file_path + '/model.pt', model, optimizer, best_valid_loss)
						# save_metrics(file_path + '/metrics.pt', train_loss_list, valid_loss_list, global_steps_list)
				iterr += 1
				percent = iterr/num_training_steps
				sec = int(time.time()-t0)
				est = int((1/percent)*sec)
				my_bar.progress(percent)
				stats.text(f"⏳ Current loss: {running_loss:.4f}. Avg loss till now: {running_loss/iterr:.4f}. Time elapsed: {timedelta(seconds = sec)}. Est time: {timedelta(seconds = est)} ---{percent*100:.2f}%")
				# progress_bar.update(1)
		st.success("Model Trained successfully.")

		# save_metrics(file_path + '/metrics.pt', train_loss_list, valid_loss_list, global_steps_list)
		print('Finished Training!')

		url = r"D:\TSApy\NoCodeMLdash\data\sequence\seq_sarcasm_train.json"
		STATS,train_data,text_field,valid_data = get_dataset_torchtext(url,80)
		vocab,train_loader,valid_loader = lstm_train(STATS,train_data,text_field,valid_data) 
		model = LSTM(vocab).to(device)
		optimizer = optim.Adam(model.parameters(), lr=0.001)

		train(model,optimizer,train_loader,valid_loader,len(train_loader)//10,num_epochs=1)



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
			stats.text(f"⏳ Current loss: {loss_value:.4f}. Avg loss till now: {run_loss/iterr:.4f}.\n Time elapsed: {timedelta(seconds = sec)}. Est time: {timedelta(seconds = est)} ---{percent*100:.2f}%")
			# progress_bar.update(1)
	st.success("Model Trained successfully.")



