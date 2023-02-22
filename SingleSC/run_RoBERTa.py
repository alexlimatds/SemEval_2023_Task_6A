import application

ENCODER_ID = 'roberta-base' # id from HuggingFace
MODEL_REFERENCE = 'RoBERTa'
MAX_SEQUENCE_LENGTH = 512
EMBEDDING_DIM = 768
N_EPOCHS = 4
LEARNING_RATE = 1e-5
BATCH_SIZE = 8 # for GPUNodes
#BATCH_SIZE = 16 # for RTX6000Node
DROPOUT_RATE = 0.1
USE_MLP = False
SAVE_BEST = False

train_params = {}
train_params['max_seq_len'] = MAX_SEQUENCE_LENGTH
train_params['learning_rate'] = LEARNING_RATE
train_params['n_epochs'] = N_EPOCHS
train_params['batch_size'] = BATCH_SIZE
train_params['encoder_id'] = ENCODER_ID
train_params['model_reference'] = MODEL_REFERENCE
train_params['dropout_rate'] = DROPOUT_RATE
train_params['embedding_dim'] = EMBEDDING_DIM
train_params['use_mlp'] = USE_MLP
train_params['save_best'] = SAVE_BEST
train_params['weight_decay'] = 1e-3
train_params['eps'] = 1e-8
train_params['freeze_layers'] = False
train_params['warmup'] = False
train_params['use_mock'] = False
train_params['n_iterations'] = 4

application.evaluate_BERT(train_params)
