import itertools
from typing import Dict, List, Iterator
from allennlp.data.tokenizers import Token
from allennlp.data import DatasetReader, Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.fields import Field, LabelField, TextField, SequenceLabelField

@DatasetReader.register("conll_03_reader")
class CoNLL03DatasetReader(DatasetReader):    
    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 **kwargs,
                 ) -> None:
        super().__init__(**kwargs)
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
    
    def _read(self, file_path: str) -> Iterator[Instance]:
        is_divider = lambda line: line.strip() == ''

        with open(file_path, 'r') as conll_file:
            for divider, lines in itertools.groupby(conll_file, is_divider):  #read each sentence groupby empty line
                if not divider:
                    fields = [l.strip().split() for l in lines] #e.g., [['EU', 'NNP', 'I-NP', 'I-ORG'], ['rejects', 'VBZ', 'I-VP', 'O'],...
                    # switch it so that each field is a list of tokens/labels
                    fields = [l for l in zip(*fields)]  #[('EU', 'rejects', 'German', 'call', 'to', 'boycott', 'British', 'lamb', '.'), ('NNP
                    # only keep the tokens and NER labels
                    tokens, _, _, ner_tags = fields  #('EU', 'rejects', 'German', 'call', 'to', 'boycott', 'British', 'lamb', '.') ('I-ORG            

                    yield self.text_to_instance(tokens, ner_tags)
                    
    def text_to_instance(self,
                         words: List[str],
                         ner_tags: List[str]) -> Instance:
        fields: Dict[str, Field] = {}
        # wrap each token in the file with a token object
        tokens = TextField([Token(w) for w in words], self._token_indexers)

        # Instances in AllenNLP are created using Python dictionaries,
        # which map the token key to the Field type
        fields["tokens"] = tokens
        fields["label"] = SequenceLabelField(ner_tags, tokens)

        return Instance(fields)