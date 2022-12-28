from allennlp.models import Model
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from typing import Dict, Optional

import torch
import torch.nn as nn

from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder
from allennlp.training.metrics import SpanBasedF1Measure

@Model.register('ner_lstm')
class NerLSTM(Model):

    def __init__(self,
                 vocab: Vocabulary,
                 embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder) -> None:
        super().__init__(vocab)

        self._embedder = embedder
        self._encoder = encoder
        self._classifier = nn.Linear(in_features=encoder.get_output_dim(),
                                     out_features=vocab.get_vocab_size('labels'))

        self._f1 = SpanBasedF1Measure(vocab, 'labels')
    
    
    def forward(self,
                tokens: Dict[str, torch.Tensor],
                label: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        
        mask = get_text_field_mask(tokens)
        
        #the tokens input isn’t a tensor of token indexes, it’s a dict. 
        # That dict contains all the namespaces defined by the  token_indexers.
        
        #`get_text_field_mask` - this function takes the tokens dict and returns a binary mask over the tokens. 
        # The mask is passed into the encoder, the metrics, and the sequence loss function so we can ignore missing text.

        embedded = self._embedder(tokens) #embed the input tokens using our pretrained word embeddings
        encoded = self._encoder(embedded, mask) #encode them using our LSTM or GRU encoder
        classified = self._classifier(encoded) #classify each timestep to the target label space
        
        self._f1(classified, label, mask) #compute some classification loss over the sequence of tokens

        output: Dict[str, torch.Tensor] = {}
        output['logits'] = classified  #for reporting

        if label is not None:
            output["loss"] = sequence_cross_entropy_with_logits(classified, label, mask) #for backpropagating
            
        #`sequence_cross_entropy_with_logits` - this is the cross-entropy loss applied to sequence classification/tagging tasks. 

        return output
    
    #note that this function is automatically called when we train
    def get_metrics(self, reset: bool = True) -> Dict[str, float]:
        return self._f1.get_metric(reset)