import application

ENCODER_ID = 'allenai/longformer-base-4096' # id from HuggingFace
MODEL_REFERENCE = 'Longformer'
#MAX_SEQUENCE_LENGTH = 1280
MAX_SEQUENCE_LENGTH = 1024
MIN_CONTEXT_LENGTH = 350
EMBEDDING_DIM = 768
N_EPOCHS = 4
LEARNING_RATE = 2e-5
BATCH_SIZE = 2 # for GPUNodes
#BATCH_SIZE = 2 # for RTX6000Node
DROPOUT_RATE = 0.1
USE_MLP = False
SAVE_BEST = False

train_params = {}
train_params['max_seq_len'] = MAX_SEQUENCE_LENGTH
train_params['min_context_len'] = MIN_CONTEXT_LENGTH
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
#train_params['n_documents'] = 5
train_params['use_mock'] = False
train_params['n_iterations'] = 4

application.evaluate_BERT(train_params)
