import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib
import time
import re
from tqdm.auto import tqdm
# Use corpus from nltk
import nltk
# nltk.download('brown')
from nltk.corpus import brown

####1.Load data
corpus_tokenized = nltk.corpus.brown.sents(categories='news')
print('corpus_tokenized sample :',len(corpus_tokenized))

#1. tokenization
corpus = [[word.lower() for word in sent] for sent in corpus_tokenized]
corpus = corpus[:100]

#2. numeralization
#find unique words
flatten = lambda l: [item for sublist in l for item in sublist]
#assign unique integer
vocabs = list(set(flatten(corpus))) #all the words we have in the system - <UNK>

#create handy mapping between integer and word
word2index = {v:idx for idx, v in enumerate(vocabs)}

#append UNK
vocabs.append('<UNK>')
word2index['<UNK>'] = len(vocabs) - 1

#just in case we need to use
index2word = {v:k for k, v in word2index.items()} 

####2.Prepare train data
def random_batch(batch_size, corpus, window_size = 1):
    # Make skip gram of one size window
    skip_grams = []
    # loop each word sequence
    # we starts from 1 because 0 has no context
    # we stop at second last for the same reason
    for sent in corpus:
        for i in range(window_size,len(sent)-window_size): #start from 2 to second last
            context_word = []
            target = word2index[sent[i]]
            for j in range(window_size):
                context = [word2index[sent[i -j -1]], word2index[sent[i +j +1]]] #window_size adjustable
                #here we want to create (banana, apple), (banana, fruit) append to some list
                for w in context:
                    context_word.append(w)
                    skip_grams.append([target, w])
    
    random_inputs = []
    random_labels = []
    random_index = np.random.choice(range(len(skip_grams)), batch_size, replace=False) #randomly pick without replacement
        
    for i in random_index:
        random_inputs.append([skip_grams[i][0]])  # target, e.g., 2
        random_labels.append([skip_grams[i][1]])  # context word, e.g., 3
            
    return np.array(random_inputs), np.array(random_labels)

####3. Model
class Skipgram(nn.Module):
    
    def __init__(self, voc_size, emb_size):
        super(Skipgram, self).__init__()
        self.embedding_center  = nn.Embedding(voc_size, emb_size)
        self.embedding_outside = nn.Embedding(voc_size, emb_size)
    
    def forward(self, center, outside, all_vocabs):
        center_embedding     = self.embedding_center(center)  #(batch_size, 1, emb_size)
        outside_embedding    = self.embedding_center(outside) #(batch_size, 1, emb_size)
        all_vocabs_embedding = self.embedding_center(all_vocabs) #(batch_size, voc_size, emb_size)
        
        top_term = torch.exp(outside_embedding.bmm(center_embedding.transpose(1, 2)).squeeze(2))
        #batch_size, 1, emb_size) @ (batch_size, emb_size, 1) = (batch_size, 1, 1) = (batch_size, 1) 

        lower_term = all_vocabs_embedding.bmm(center_embedding.transpose(1, 2)).squeeze(2)
        #batch_size, voc_size, emb_size) @ (batch_size, emb_size, 1) = (batch_size, voc_size, 1) = (batch_size, voc_size) 
        
        lower_term_sum = torch.sum(torch.exp(lower_term), 1)  #(batch_size, 1)
        
        loss = -torch.mean(torch.log(top_term / lower_term_sum))  #scalar
        
        return loss
    
#prepare all vocabs
voc_size   = len(vocabs)
print('voc_size :', voc_size)

def prepare_sequence(seq, word2index):
    idxs = list(map(lambda w: word2index[w] if word2index.get(w) is not None else word2index["<UNK>"], seq))
    return torch.LongTensor(idxs)
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

batch_size = 2
emb_size   = 2
all_vocabs = prepare_sequence(list(vocabs), word2index).expand(batch_size, voc_size)
all_vocabs

model      = Skipgram(voc_size, emb_size)
optimizer  = optim.Adam(model.parameters(), lr=0.001)

#### 4. Training
num_epochs = 5000
save_path = f'models/{model.__class__.__name__}.pt'
start = time.time()
for epoch in tqdm(range(num_epochs)):
    
    #get batch
    input_batch, label_batch = random_batch(batch_size, corpus)
    input_tensor = torch.LongTensor(input_batch)
    label_tensor = torch.LongTensor(label_batch)
    
    #predict
    loss = model(input_tensor, label_tensor, all_vocabs)
    
    #backprogate
    optimizer.zero_grad()
    loss.backward()
    
    #update alpha
    optimizer.step()

    end = time.time()

    epoch_mins, epoch_secs = epoch_time(start, end)
    
    torch.save(model.state_dict(), save_path)
    #print the loss
    if (epoch + 1) % 1000 == 0:
        print(f"Epoch {epoch+1:6.0f} | Loss: {loss:2.6f}")