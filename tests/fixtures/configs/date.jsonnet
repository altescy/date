{
  dataset_reader: {
    type: 'text_classification_json',
    tokenizer: 'whitespace',
  },
  train_data_path: 'tests/fixtures/data/imdb_corpus.jsonl',
  validation_data_path: 'tests/fixtures/data/imdb_corpus.jsonl',
  model: {
    type: 'date',
    anomaly_label: 'pos',
    text_field_embedder: {
      token_embedders: {
        tokens: {
          type: 'embedding',
          embedding_dim: 10,
          trainable: true,
        },
      },
    },
    rmd_seq2vec_encoder: {
      type: 'bag_of_embeddings',
      embedding_dim: 10,
    },
    rtd_seq2seq_encoder: {
      type: 'pass_through',
      input_dim: 10,
    },
    masks: [
      '256.gAUEWmgfHYsEIgiMJSkCBYkUFAwg0gEFBBRxIRIMAC0=',
      '256.EIACACQQQVBjgAoQAgTGBgACBDEAgI1BFDNICABAQYA=',
      '256.LIjiYQFKpBIJoRBW8QoSMCgC8iQjAkADMBIiGAMCASQ=',
    ],
    namespace: 'tokens',
    primary_token_indexer: 'tokens',
  },
  data_loader: {
    type: 'simple',
    batch_size: 2,
    shuffle: false,
  },
  trainer: {
    optimizer: {
      type: 'adam',
      lr: 0.01,
    },
    validation_metric: '+auc',
    num_epochs: 3,
    patience: 5,
    cuda_device: -1,
  },
}
