{
  dataset_reader: {
    type: 'conll_03_reader',
    token_indexers: {
      tokens: {
        type: 'single_id',
        namespace: 'tokens',
        lowercase_tokens: true
      }
    },
  },
  data_loader: {
    batch_sampler: {
      type: 'bucket',
      batch_size: 10
    }
  },
  train_data_path: '../data/conll2003/train.txt',
  validation_data_path: '../data/conll2003/validation.txt',
  model: {
    type: 'ner_lstm',
    embedder: {
      token_embedders: {
        tokens: {
        type: 'embedding',
        pretrained_file: "https://allennlp.s3.amazonaws.com/datasets/glove/glove.6B.50d.txt.gz",
        embedding_dim: 50,
        trainable: false
        }
      }
    },
    encoder: {
      type: 'lstm',
      input_size: 50,
      hidden_size: 25,
      bidirectional: true
    }
  },
  trainer: {
    num_epochs: 40,
    patience: 10,
    cuda_device: -1,
    grad_clipping: 5.0,
    validation_metric: '-loss',
    optimizer: {
      type: 'adam',
      lr: 0.003
    }
  }
}