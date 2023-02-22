# Models and related classes and functions
import torch
from torch.utils.data import Dataset
import transformers
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import numpy as np
import time

def n_tokens_in_core_sentences(idx_first_sent, idx_last_sent, tokenized_sentences):
    '''
    TODO
    '''
    n = 0
    for i in range(idx_first_sent, idx_last_sent + 1):
        n += (len(tokenized_sentences[i]) + 1) # +1 because future SEP token
    return n

def core_sentences_idx(tokenized_sentences, max_seq_len, min_context_len):
    '''
    Sets the number of sequences and their respective core sentences.
    Arguments:
        max_seq_len: maximum number of tokens in a sequence.
        min_context_len: desired minimum number of neighbor tokens.
        tokenized_sentences: list of list of integers (token IDs)
    Returns:
        List of tuples. Each tuple represents a sequence and it has two 
        integers: index of first and last sentences in the sequence.
    '''
    threshold_ctx = max_seq_len - min_context_len - 2 # -2 because CLS and first SEP
    
    n_sentences = len(tokenized_sentences)
    idx_core_sentences = [] # list of tuple of 2 integers (indexes of first and last core sentences in a sequence)
    idx_current_sent = 0
    while idx_current_sent < n_sentences:
        # new sequence
        idx_start = idx_current_sent # begining of current sequence
        idx_end = idx_current_sent   # end of current sequence
        idx_current_sent += 1
        len_core_sent_in_seq = n_tokens_in_core_sentences(idx_start, idx_end, tokenized_sentences)
        while len_core_sent_in_seq < threshold_ctx and idx_current_sent < n_sentences:
            # checking if current sentence fits into the sequence while maintain minimum context length
            if len_core_sent_in_seq + len(tokenized_sentences[idx_current_sent]) < threshold_ctx:
                idx_end = idx_current_sent
                idx_current_sent += 1
                len_core_sent_in_seq = n_tokens_in_core_sentences(idx_start, idx_end, tokenized_sentences)
            else:
                break
        idx_core_sentences.append((idx_start, idx_end))
    return idx_core_sentences

def tokenize_sentences(sentences, tokenizer, max_seq_len, min_context_len):
    '''
    Tokenizes some core sentences together their neighbor sentences. The total sequence length 
    is limited by max_seq_len argument. This function aims to entirely encode the core 
    sentences while it try to maintain a minimum number of tokens in the left and right 
    neighbor sentences. The min_context_len parameter sets the minimum number of neighbor 
    tokens. The number of core sentences is dynamically set and it depends on the length 
    of the core sentences and on the min_context_len param.
    Arguments:
        sentences: list of sentences (strings).
        tokenizer: tokenizer of respective sentence encoder.
        max_seq_len: maximum number of tokens in a sequence.
        min_context_len: desired minimum number of neighbor tokens.
    Returns:
        List of list of integers. Each sublist represents a set of core sentences. 
        The integers are token ids. A sublist will have the following 
        layout: [CLS] left context [SEP] core sentence 1 [SEP] core sentence 2 [SEP] ... 
        core sentence M [SEP] right context [SEP]
    '''
    # tokenizing sentences to get their number of tokens
    tokenized_sentences = tokenizer(
        sentences, 
        add_special_tokens=False,
        padding=False, 
        return_token_type_ids=False, 
        return_attention_mask=False, 
        truncation=True, 
        max_length=max_seq_len - 4 # -4 because special tokens
    )['input_ids']
    
    # sequences with indexes of respective core sentences
    idx_sequences = core_sentences_idx(tokenized_sentences, max_seq_len, min_context_len)
    
    # assembling sequences
    sequences = []
    for core_start, core_end in idx_sequences:
        current_seq = []
        # inserting core sentences
        for i in range(core_start, core_end + 1):
            sent = tokenized_sentences[i].copy()
            sent.append(tokenizer.sep_token_id)
            current_seq.extend(sent)
        # inserting context
        current_seq.insert(0, tokenizer.sep_token_id)
        len_ctx = max_seq_len - len(current_seq) - 2 # -2 because [CLS] and final [SEP]
        
        if len_ctx > 0:
            # flattening left neighbor tokens in a unique list
            left_tokens = []
            for j in range(core_start):
                left_tokens.extend(tokenized_sentences[j])
            # flattening right neighbor tokens in a unique list
            right_tokens = []
            for j in range(core_end + 1, len(tokenized_sentences)):
                right_tokens.extend(tokenized_sentences[j])
            # inserting neighbor tokens
            count = 0
            idx_left = len(left_tokens) - 1
            idx_right = 0
            while count < len_ctx and (idx_left >= 0 or idx_right < len(right_tokens)):
                if idx_left >= 0:
                    current_seq.insert(0, left_tokens[idx_left])
                    idx_left -= 1
                    count += 1
                if idx_right < len(right_tokens) and count < len_ctx:
                    current_seq.append(right_tokens[idx_right])
                    idx_right += 1
                    count += 1
        # insert special tokens
        current_seq.insert(0, tokenizer.cls_token_id)
        current_seq.append(tokenizer.sep_token_id)
        
        sequences.append(current_seq)
        
    return sequences

def pad_sentences(sentences_as_ids, pad_token_id, max_seq_len):
    """
    Inserts [PAD] tokens in tokenized sentences. The tokens are inserted at right 
    untill each tokenized sentence reachs max_seq_len. sentences_as_ids is 
    modified in place.
    Arguments:
        sentences_as_ids : list of token ids as returned by the tokenize_sentences function.
        pad_token_id : id of [PAD] token.
        max_seq_len : the desired final length for all sentences.
    """
    for s in sentences_as_ids:
        diff = max_seq_len - len(s)
        if diff > 0:
            for i in range(diff):
                s.append(pad_token_id)

class ContextualizedSC_Dataset(torch.utils.data.Dataset):
    """
    A dataset object to be used together a ContextualizedSC_BERT model. 
    Each item of the dataset represents a set of core sentences and context tokens (check 
    the tokenize_sentences function).
    """
    def __init__(self, dic_docs, labels_to_idx, tokenizer, max_seq_len, min_context_len):
        """
        Arguments:
            dic_docs: a dictionary whose each item is a document (key: docId, value: pandas DataFrame).
            labels_to_idx : dictionary that maps each label (string) to a index (integer).
            tokenizer : tokenizer of the encoder model.
            max_seq_len: maximum sequence length.
            min_context_len: 
        """
        self.labels = []    # list of list of strings (n sequences, n core sentences in sequence)
        self.targets = []   # list of unidimensional tensors (n sequences, n core sentences in sequence)
        self.input_ids = [] # tensor of shape (n sequences, max_seq_len)
        self.len = 0
        
        # tokenizing and spliting sentences in blocks/sequences
        for df in dic_docs.values():
            # sequences
            sequences = tokenize_sentences(df['sentence'].tolist(), tokenizer, max_seq_len, min_context_len)
            pad_sentences(sequences, tokenizer.pad_token_id, max_seq_len)
            sequences = torch.tensor(sequences, dtype=torch.long)
            self.input_ids.append(sequences)
            
            # labels and targets
            labels_in_doc = df['label'].tolist()
            idx_start = 0
            for i in range(sequences.shape[0]):
                n_sep_tokens = torch.count_nonzero(sequences[i] == tokenizer.sep_token_id)
                n_labels_in_sequence = n_sep_tokens - 2 # -2 because SEP token of each context side
                idx_end = idx_start + n_labels_in_sequence
                sequence_labels = labels_in_doc[idx_start:idx_end]
                assert n_labels_in_sequence == len(sequence_labels)
                self.labels.append(sequence_labels)
                self.targets.append(torch.tensor(
                    [labels_to_idx[l] for l in sequence_labels], 
                    dtype=torch.long
                ))
                idx_start = idx_end
        self.input_ids = torch.vstack(self.input_ids)
        self.len = self.input_ids.shape[0]
        
        # creating attention masks
        self.masks = torch.ones(self.input_ids.shape)
        self.masks[self.input_ids == tokenizer.pad_token_id] = 0

    def __getitem__(self, index):
        return {
            'ids': self.input_ids[index],       # shape: (max_seq_len)
            'mask': self.masks[index],          # shape: (max_seq_len)
            'targets': self.targets[index],      # shape: (num of core sentences in current sequence)
            'labels': self.labels[index]         # shape: (num of core sentences in current sequence)
        }
    
    def __len__(self):
        return self.len

def collate_batch(batch):
    '''
    Prepares a batch of dataset items.
    Arguments:
        batch: list of dataset items (dictionaries).
    Returns:
        A dictionary with following items:
            'ids': tensor of input ids. Shape: (n sequences in batch, max_seq_len)
            'masks': tensor of attention masks. Shape: (n sequences in batch, max_tokens)
            'targets': tensor of golden class ids. Shape: (n core sentences in batch)
            'labels': list of golden labels. Shape: (n core sentences in batch)
    '''
    labels = []
    targets = []
    input_ids = None
    masks = None
    for entry in batch:
        labels.extend(entry['labels'])
        targets.extend(entry['targets'])
        if input_ids is None:
            input_ids = entry['ids'].reshape(1,-1) # reshape to assure a 2-D tensor when the batch contains just one element
            masks = entry['mask'].reshape(1,-1)
        else:
            input_ids = torch.vstack((input_ids, entry['ids']))
            masks = torch.vstack((masks, entry['mask']))
    return {
        'ids': input_ids, 
        'mask': masks, 
        'targets': torch.tensor(targets, dtype=torch.long), 
        'labels': labels
    }

def count_labels(ds, labels):
    """
    Returns the number of sentences by label for a provided ContextualizedSC_Dataset.
    Arguments:
        ds : a ContextualizedSC_Dataset.
        labels: list of strings. The set of possible labels.
    Returns:
       A dictionary mapping each label (string) to its number of sentences (integer).
    """
    count_by_label = {l: 0 for l in labels}
    for sublist in ds.labels:
        for l in sublist:
            count_by_label[l] = count_by_label[l] + 1
    return count_by_label

class MockCtxSC_BERT(torch.nn.Module):
    '''
    A mock of ContextualizedSC_BERT. It's usefull to accelerate the validation
    of the training loop.
    '''
    def __init__(self, n_classes, sep_token_id):
        super(MockCtxSC_BERT, self).__init__()
        self.classifier = torch.nn.Linear(10, n_classes) # 10 => random choice
        self.sep_token_id = sep_token_id

    def forward(self, input_ids, attention_mask):
        # getting number of labels/targets
        n_chunks = input_ids.shape[0]
        n_sep_tokens = torch.count_nonzero(input_ids == self.sep_token_id).item()
        n_targets = n_sep_tokens - 2 * n_chunks
        
        mock_data = torch.rand((n_targets, 10), device=input_ids.device)
        logits = self.classifier(mock_data)    # shape: (n_targets, n_classes)

        return logits

class ContextualizedSC_BERT(torch.nn.Module):
    """
    Sentence Classifier based on a BERT kind encoder. This model expects to 
    get as inputs sentence represenations from a ContextualizedSC_Dataset object.
    The sentence encoder must be a pre-trained model based on BERT's architecture 
    like BERT, RoBERTa and ALBERT.
    """
    def __init__(self, encoder_id, sep_token_id, n_classes, dropout_rate, embedding_dim, use_mlp, n_hidden_mlp=100):
        '''
        This model comprises a pre-trained sentence encoder and a classification head. 
        The sentence encoder must be a model following BERT architecture.  
        The classification head is a linear classifier (a single feedforward layer).
        Arguments:
            encoder_id: ID (string) of the encoder model in Hugging Faces repository.
            sep_token_id: ID (integer) of the [SEP] token.
            n_classes: number of classes.
            dropout_rate: dropout rate of classification layer.
            embedding_dim: dimension of hidden units in the sentence encoder (e.g., 768 for BERT).
            use_mlp: indicates the use of a MLP classifier (True) or a single layer one (False) in the classification head.
            n_hidden_mlp: the number of hidden units of the MLP classifier.
        '''
        super(ContextualizedSC_BERT, self).__init__()
        self.sep_token_id = sep_token_id
        self.encoder = transformers.AutoModel.from_pretrained(encoder_id)
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
        output_1 = self.encoder(
            input_ids=input_ids,             # input_ids.shape: (batch_size, seq_len)
            attention_mask=attention_mask    # attention_mask.shape: (batch_size, seq_len)
        )
        hidden_state = output_1.last_hidden_state  # hidden states of last encoder's layer => shape: (batch_size, seq_len, embedd_dim)
        
        ### In this approach we concatenate [CLS] embedding with [SEP] embedding
        # getting embeddings of core [SEP] and [CLS]tokens
        batch_size = input_ids.shape[0]
        core_embeddings = [] # core_embeddings.shape: (n core sentences in batch, embedding_dim)
        for i in range(batch_size): # iterates sequences/blocks
            cls_embedding = hidden_state[i, 0, :] # cls_embedding.shape: (embedd_dim)
            idx_seps = torch.nonzero(input_ids[i] == self.sep_token_id, as_tuple=True)[0] # indexes of all [SEP] tokens in currrent sentence
            # we dont want the embeddings of the 1st and last [SEP] tokens
            idx_core_sep = idx_seps[1:-1] # index of core [SEP] in current sequence
            core_sep_emb = hidden_state[i, idx_core_sep, :] # embeddings of core [SEP]. shape: (n core sentences in chunk, embedd_dim)
            for j in range(core_sep_emb.shape[0]):
                core_embeddings.append(
                    torch.hstack((cls_embedding, core_sep_emb[j]))
                )
        core_embeddings = torch.vstack(core_embeddings)
        
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
    Creates and train an instance of ContextualizedSC_BERT.
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
    
    dl_train = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    dl_test = torch.utils.data.DataLoader(ds_test, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    
    if use_mock:
        sentence_classifier = MockCtxSC_BERT(n_classes, train_params['sep_token_id']).to(device)
    else:
        sentence_classifier = ContextualizedSC_BERT(
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
