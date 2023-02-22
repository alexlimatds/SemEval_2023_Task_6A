# Models and related classes and functions
import torch
from torch.utils.data import Dataset
import transformers
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import numpy as np
import time
import models

class LongformerEncoder(torch.nn.Module):
    
    def __init__(self, encoder_id, sep_token_id):
        super(LongformerEncoder, self).__init__()
        self.sep_token_id = sep_token_id
        self.encoder = transformers.AutoModel.from_pretrained(encoder_id)
        
    def forward(self, input_ids, attention_mask):
        # The global_attention_mask tells which tokens must have global attention. 
        # Here, <s> and </s> have global attention
        global_attention_mask = torch.zeros(
            input_ids.shape, 
            dtype=torch.long, 
            device=input_ids.device
        )
        global_attention_mask[:, 0] = 1 # global attention for <s> since it must be the first token in a chunk
        idx_sep = torch.nonzero(input_ids == self.sep_token_id, as_tuple=True)
        for i in range(idx_sep[0].shape[0]):
            global_attention_mask[idx_sep[0][i], idx_sep[1][i]] = 1 # global attention for </s>
        
        # Encoding
        output_1 = self.encoder(
            input_ids=input_ids,             # input_ids.shape: (batch_size, seq_len)
            attention_mask=attention_mask    # attention_mask.shape: (batch_size, seq_len)
        )
        hidden_state = output_1.last_hidden_state  # hidden states of last encoder's layer => shape: (batch_size, seq_len, embedd_dim)
        
        ### In this approach we concatenate <s> embedding with </s> embedding
        # getting embeddings of core </s> and <s> tokens
        batch_size = input_ids.shape[0]
        core_embeddings = []
        for i in range(batch_size): # iterates sequences/blocks
            cls_embedding = hidden_state[i, 0, :] # cls_embedding.shape: (embedd_dim)
            idx_seps = torch.nonzero(input_ids[i] == self.sep_token_id, as_tuple=True)[0] # indexes of all </s> tokens in currrent sentence
            # we dont want the embeddings of the 1st and last </s> tokens
            idx_core_sep = idx_seps[1:-1] # index of core </s> in current sequence
            core_sep_emb = hidden_state[i, idx_core_sep, :] # embeddings of core </s>. shape: (n core sentences in chunk, embedd_dim)
            for j in range(core_sep_emb.shape[0]):
                core_embeddings.append(
                    torch.hstack((cls_embedding, core_sep_emb[j]))
                )
        core_embeddings = torch.vstack(core_embeddings) # core_embeddings.shape: (n core sentences in batch, embedding_dim * 2)
        return core_embeddings

class DFCSC_Longformer(torch.nn.Module):
    """
    Sentence Classifier based on a Longformer kind encoder. This model expects to 
    get as inputs sentence represenations from a ContextualizedSC_Dataset object.
    The sentence encoder must be a pre-trained model based on Longformer's architecture.
    """
    def __init__(self, encoder_id, sep_token_id, n_classes, dropout_rate, embedding_dim, use_mlp, n_hidden_mlp=100):
        '''
        This model comprises a pre-trained sentence encoder and a classification head. 
        The sentence encoder must be a model following Longformer architecture.  
        The classification head is a linear classifier (a single feedforward layer).
        Arguments:
            encoder_id: ID (string) of the encoder model in Hugging Faces repository.
            sep_token_id: ID (integer) of the [SEP] token.
            n_classes: number of classes.
            dropout_rate: dropout rate of classification layer.
            embedding_dim: dimension of hidden units in the sentence encoder (e.g., 768 for Longformer).
            use_mlp: indicates the use of a MLP classifier (True) or a single layer one (False) in the classification head.
            n_hidden_mlp: the number of hidden units of the MLP classifier.
        '''
        super(DFCSC_Longformer, self).__init__()
        self.encoder = LongformerEncoder(encoder_id, sep_token_id)
        dropout = torch.nn.Dropout(dropout_rate)
        n_classes = n_classes
        if use_mlp:
            dense_hidden = torch.nn.Linear(embedding_dim * 2, n_hidden_mlp)
            torch.nn.init.kaiming_uniform_(dense_hidden.weight)
            relu = torch.nn.ReLU()
            dense_out = torch.nn.Linear(n_hidden_mlp, n_classes)
            torch.nn.init.xavier_uniform_(dense_out.weight)
            self.classifier = torch.nn.Sequential(
                dropout, dense_hidden, relu, dropout, dense_out
            )
        else:
            dense_out = torch.nn.Linear(embedding_dim * 2, n_classes)
            torch.nn.init.xavier_uniform_(dense_out.weight)
            self.classifier = torch.nn.Sequential(
                dropout, dense_out
            )

    def forward(self, input_ids, attention_mask):
        '''
        Each call to this method process a batch of contextualized sentences (i.e. 
        core sentences together with its neighbor tokens). The model exploits the 
        context and the core sentences to produce embeddings for the core sentences, 
        but it doesn't produce embeddings for context tokens.
        This method returns one logit tensor for each core sentence in the batch.
        Arguments:
            input_ids : tensor of shape (batch_size, seq_len)
            attention_mask : tensor of shape (batch_size, seq_len)
        Returns:
            logits : tensor of shape (n of core sentences in batch, n of classes)
        '''
        core_embeddings = self.encoder(input_ids, attention_mask)
        
        logits = self.classifier(core_embeddings)   # logits.shape: (n core sentences in batch, num of classes)

        return logits

def evaluate(model, test_dataloader, loss_function, device):
    """
    Evaluates a provided SingleSC model.
    Arguments:
        model: the model to be evaluated.
        test_dataloader: torch.utils.data.DataLoader instance containing the test data.
        loss_function: instance of the loss function used to train the model.
        device: device where the model is located.
    Returns:
        eval_loss (float): the computed test loss score.
        precision (float): the computed test Precision score.
        recall (float): the computed test Recall score.
        f1 (float): the computed test F1 score.
        confusion_matrix: the computed test confusion matrix.
    """
    predictions = torch.tensor([]).to(device)
    y_true = torch.tensor([]).to(device)
    eval_loss = 0
    model.eval()
    with torch.no_grad():
        for data in test_dataloader:
            ids = data['ids'].to(device)
            mask = data['mask'].to(device)
            y_true_batch = data['targets'].to(device)
            y_hat = model(ids, mask)
            loss = loss_function(y_hat, y_true_batch)
            eval_loss += loss.item()
            predictions_batch = y_hat.argmax(dim=1)
            predictions = torch.cat((predictions, predictions_batch))
            y_true = torch.cat((y_true, y_true_batch))
        predictions = predictions.detach().to('cpu').numpy()
        y_true = y_true.detach().to('cpu').numpy()
    eval_loss = eval_loss / len(test_dataloader)
    t_metrics_micro = precision_recall_fscore_support(
        y_true, 
        predictions, 
        average='micro', 
        zero_division=0
    )
    t_metrics_weighted = precision_recall_fscore_support(
        y_true, 
        predictions, 
        average='weighted', 
        zero_division=0
    )
    cm = confusion_matrix(
        y_true, 
        predictions
    )
    
    return eval_loss, t_metrics_micro[0], t_metrics_micro[1], t_metrics_micro[2], t_metrics_weighted[0], t_metrics_weighted[1], t_metrics_weighted[2], cm

def fit(train_params, ds_train, ds_test, device):
    """
    Creates and train an instance of DFCSC_Longformer.
    Arguments:
        train_params: dictionary storing the training params.
        ds_train: instance of ContextualizedSC_Dataset storing the train data.
        tokenizer: the tokenizer of the chosen pre-trained sentence encoder.
        device: device where the model is located.
    """
    learning_rate = train_params['learning_rate']
    weight_decay = train_params['weight_decay']
    n_epochs = train_params['n_epochs']
    batch_size = train_params['batch_size']
    encoder_id = train_params['encoder_id']
    n_classes = train_params['n_classes']
    dropout_rate = train_params['dropout_rate']
    embedding_dim = train_params['embedding_dim']
    use_mlp = train_params['use_mlp']
    use_mock = train_params['use_mock']
    
    dl_train = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, shuffle=True, collate_fn=models.collate_batch)
    dl_test = torch.utils.data.DataLoader(ds_test, batch_size=batch_size, shuffle=True, collate_fn=models.collate_batch)
    
    if use_mock:
        sentence_classifier = models.MockCtxSC_BERT(n_classes, train_params['sep_token_id']).to(device)
    else:
        sentence_classifier = DFCSC_Longformer(
            encoder_id, 
            train_params['sep_token_id'], 
            n_classes, 
            dropout_rate, 
            embedding_dim, 
            use_mlp
        ).to(device)
    
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(
        sentence_classifier.parameters(), 
        lr=learning_rate, 
        betas=(0.9, 0.999), 
        eps=train_params['eps'], 
        weight_decay=weight_decay
    )
    lr_scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps = 0, 
        num_training_steps = len(dl_train) * n_epochs
    )
    
    metrics = {} # key: epoch number, value: numpy tensor storing train loss, test loss, Precision (macro), Recall (macro), F1 (macro), Precision (weighted), Recall (weighted), F1 (weighted)
    confusion_matrices = {} # key: epoch number, value: scikit-learn's confusion matrix
    best_score = None
    start_train = time.perf_counter()
    for epoch in range(1, n_epochs + 1):
        print(f'Starting epoch {epoch}... ', end='')
        start_epoch = time.perf_counter()
        epoch_loss = 0
        sentence_classifier.train()
        for train_data in dl_train:
            optimizer.zero_grad()
            ids = train_data['ids'].to(device)
            mask = train_data['mask'].to(device)
            y_hat = sentence_classifier(ids, mask)
            y_true = train_data['targets'].to(device)
            loss = criterion(y_hat, y_true)
            epoch_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(sentence_classifier.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
        epoch_loss = epoch_loss / len(dl_train)
        # evaluation
        optimizer.zero_grad()
        eval_loss, p_micro, r_micro, f1_micro, p_weighted, r_weighted, f1_weighted, cm = evaluate(
            sentence_classifier, 
            dl_test, 
            criterion, 
            device
        )
        #storing metrics
        metrics[epoch] = np.array([epoch_loss, eval_loss, p_micro, r_micro, f1_micro, p_weighted, r_weighted, f1_weighted])
        confusion_matrices[epoch] = cm
        end_epoch = time.perf_counter()
        print('finished! Time: ', time.strftime("%Hh%Mm%Ss", time.gmtime(end_epoch - start_epoch)))
        # saving model
        if train_params['save_best'] and (best_score is None or f1_micro > best_score):
            best_score = f1_micro
            torch.save(
                {
                    'epoch': epoch, 
                    'model_state_dict': sentence_classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': epoch_loss, 
                    'test_loss': eval_loss, 
                    'precision_micro': p_micro, 
                    'recall_micro': r_micro, 
                    'f1_micro': f1_micro, 
                    'precision_weighted': p_weighted, 
                    'recall_weighted': r_weighted, 
                    'f1_weighted': f1_weighted, 
                    'train_params': train_params
                }, 
                f'{train_params["time_tag"]}.pt'
            )
            
    end_train = time.perf_counter()
    
    return metrics, confusion_matrices, time.strftime("%Hh%Mm%Ss", time.gmtime(end_train - start_train))
