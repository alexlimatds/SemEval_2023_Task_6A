from os import listdir
import pandas as pd
import numpy as np
import csv, random, torch, models, transformers, time, json, models
from datetime import datetime

def evaluate_model(train_params):
    # time tag
    model_reference = train_params['model_reference']
    chunk_layout = train_params['chunk_layout']
    time_tag = f'{chunk_layout}_{model_reference}_{datetime.now().strftime("%Y-%m-%d-%Hh%Mm%Ss")}'
    train_params['time_tag'] = time_tag
    
    # setting labels
    labels_to_idx = {
        'ANALYSIS': 0,
        'ARG_PETITIONER': 1, 
        'ARG_RESPONDENT': 2, 
        'FAC': 3, 
        'ISSUE': 4, 
        'NONE': 5, 
        'PREAMBLE': 6, 
        'PRE_NOT_RELIED': 7, 
        'PRE_RELIED': 8, 
        'RATIO': 9, 
        'RLC': 10, 
        'RPC': 11, 
        'STA': 12
    }
    labels = [None] * len(labels_to_idx)
    for l, idx in labels_to_idx.items():
        labels[idx] = l
    train_params['n_classes'] = len(labels)
    
    # tokenizer
    encoder_id = train_params['encoder_id']
    tokenizer = transformers.AutoTokenizer.from_pretrained(encoder_id)
    train_params['sep_token_id'] = tokenizer.sep_token_id
    
    # loading documents
    train_file_path = '../../BUILD/train.json'
    test_file_path = '../../BUILD/dev.json'
    dic_docs_train = raw_to_dic_docs(train_file_path)
    dic_docs_test = raw_to_dic_docs(test_file_path)
    
    # dataset objects
    max_seq_len = train_params['max_seq_len']
    max_sent_len = train_params['max_sent_len']
    max_sent_per_block = train_params['max_sent_per_block']
    if train_params.get('n_documents') is not None: # used in tests to speed up the train procedure
        n_documents = train_params.get('n_documents')
        dic_docs_train = {k: dic_docs_train[k] for k in sorted(dic_docs_train.keys())[:n_documents]}
        dic_docs_test = {k: dic_docs_test[k] for k in sorted(dic_docs_test.keys())[:n_documents]}
    if chunk_layout == 'Cohan':
        ds_train = models.CohanDataset(dic_docs_train, labels_to_idx, tokenizer, max_sent_len, max_sent_per_block, max_seq_len)
        ds_test = models.CohanDataset(dic_docs_test, labels_to_idx, tokenizer, max_sent_len, max_sent_per_block, max_seq_len)
    elif chunk_layout == 'VanBerg':
        window_len = train_params['window_len']
        ds_train = models.OverlapedChunksDataset(dic_docs_train, tokenizer, max_sent_len, max_sent_per_block, max_seq_len, window_len, labels_to_idx)
        ds_test = models.OverlapedChunksDataset(dic_docs_test, tokenizer, max_sent_len, max_sent_per_block, max_seq_len, window_len, labels_to_idx)
    else:
        raise ValueError('Not supported chunk layout:', chunk_layout)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    raw_metrics = {} # key: epoch, value: numpy tensor of shape (n_iterations, 8)
    confusion_matrices = {} # key: iteration_id, value: dictionary (key: epoch, value: confusion matrix)
    cv_start = time.perf_counter()
    seeds = [(42 + i * 10) for i in range(train_params['n_iterations'])]
    for i, seed_val in enumerate(seeds):
        print(f'Started iteration {i + 1}')
        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        torch.cuda.manual_seed_all(seed_val)
        # train
        iteration_metrics, cm, train_time = models.fit(train_params, ds_train, ds_test, device)
        confusion_matrices[i] = cm
        for epoch, scores in iteration_metrics.items():
            epoch_metrics = raw_metrics.get(epoch, None)
            if epoch_metrics is None:
                raw_metrics[epoch] = scores.reshape(1,-1)
            else:
                raw_metrics[epoch] = np.vstack((epoch_metrics, scores))
        print('  Iteration time: ', train_time)

    metrics = pd.DataFrame(columns=[
        'Epoch', 'Train loss', 'std', 'Test loss', 'std', 
        'P (micro)', 'P std', 'R (micro)', 'R std', 'F1 (micro)', 'F1 std', 
        'P (weighted)', 'P std', 'R (weighted)', 'R std', 'F1 (weighted)', 'F1 std', 
    ])
    for i, (epoch, scores) in enumerate(raw_metrics.items()):
        mean = np.mean(scores, axis=0)
        std = np.std(scores, axis=0)
        metrics.loc[i] = [
            f'{epoch}', 
            f'{mean[0]:.6f}', f'{std[0]:.6f}',    # train loss
            f'{mean[1]:.6f}', f'{std[1]:.6f}',    # test loss
            f'{mean[2]:.4f}', f'{std[2]:.4f}',    # precision (micro)
            f'{mean[3]:.4f}', f'{std[3]:.4f}',    # recall (micro)
            f'{mean[4]:.4f}', f'{std[4]:.4f}',    # f1 (micro)
            f'{mean[5]:.4f}', f'{std[5]:.4f}',    # precision (weighted)
            f'{mean[6]:.4f}', f'{std[6]:.4f}',    # recall (weighted)
            f'{mean[7]:.4f}', f'{std[7]:.4f}',    # f1 (weighted)
        ]
    
    cv_end = time.perf_counter()
    cv_time = time.strftime("%Hh%Mm%Ss", time.gmtime(cv_end - cv_start))
    print('End of evaluation. Total time:', cv_time)
    save_report(
        metrics, raw_metrics, labels, 
        confusion_matrices, f'test set ({len(seeds)} random seeds)', train_params, cv_time, device, time_tag
    )

def save_report(
    avg_metrics, complete_metrics, labels, 
    confusion_matrices, evaluation, train_params, train_time, device, time_tag):
    """
    Arguments:
        avg_metrics : A pandas Dataframe with the averaged metrics.
        complete_metrics : A dictionary with the metrics by epoch. The key indicates the epoch. 
                            Each value must be a numpy tensor of shape (n_iterations, 5).
        labels : list of all labels.
        confusion_matrices : A dictionary => key: iteration_id, value: dictionary (key: epoch, value: confusion matrix)
        evaluation : the kind of evalutaion (string). Cross-validation or Holdout.
        train_params : A dictionary.
        train_time : total time spent on training/evaluation (string).
        device : ID of GPU device
        time_tag : time tag to be appended to report file name.
    """
    model_reference = train_params['model_reference']
    report = (
        'RESULTS REPORT\n'
        f'Model: {model_reference}\n'
        f'Encoder: {train_params["encoder_id"] if not train_params["use_mock"] else "MOCK MODEL"}\n'
        f'Chunk layout: {train_params["chunk_layout"]}\n'
        f'Evaluation: {evaluation}\n'
        'Train scheme: fine-tuning\n'
        f'Max sequence length: {train_params["max_seq_len"]}\n'
        f'Max sentence length: {train_params["max_sent_len"]}\n'
        f'Max sentences per chunk: {train_params["max_sent_per_block"]}\n'
    )
    if train_params["chunk_layout"] == "VanBerg":
        report += f'Window length: {train_params["window_len"]}\n'
    report += (
        f'Batch size: {train_params["batch_size"]}\n'
        f'Dropout rate: {train_params["dropout_rate"]}\n'
        f'Learning rate: {train_params["learning_rate"]}\n'
        f'Adam Epsilon: {train_params["eps"]}\n'
        f'Use MLP: {train_params["use_mlp"]}\n'
        f'Weight decay: {train_params["weight_decay"]}\n'
        f'Train time: {train_time}\n'
        f'GPU name: {torch.cuda.get_device_name(device)}\n'
    )
        
    memory_in_bytes = torch.cuda.get_device_properties(device).total_memory
    memory_in_gb = round((memory_in_bytes/1024)/1024/1024,2)
    report += f'GPU memory: {memory_in_gb}\n\n'
    
    report += 'Averages:\n'
    report += avg_metrics.to_string(index=False, justify='center')
    
    report += '\n\n*** Detailed report ***\n'
    
    report += f'\nConfusion matrices\n{"-"*18}\n'
    for i, label in enumerate(labels):
        report += f'{label}: {i} \n'
    for iteration_id, cm_dic in confusion_matrices.items():
        report += f'=> Iteration {iteration_id}:\n'
        for e, cm in cm_dic.items():
            report += f'Epoch {e}:\n{cm}\n'

    report += f'\nScores\n{"-"*6}\n'
    for epoch, scores in complete_metrics.items():
        df = pd.DataFrame(
            scores, 
            columns=['Train loss', 'Test loss', 'P (micro)', 'R (micro)', 'F1 (micro)', 'P (weighted)', 'R (weighted)', 'F1 (weighted)'], 
            index=[f'Iteration {i}' for i in range(scores.shape[0])])
        report += f'Epoch: {epoch}\n' + df.to_string() + '\n\n'
    
    with open('./' + f'report-{time_tag}.txt', 'w') as f:
        f.write(report)

def raw_to_dic_docs(file_path):
    """
    Reads a json file containing data and returns a dictionary that maps each 
    document to a dataframe containing two columns: sentence and label.
    """
    with open(file_path) as data_file:
        raw_data = json.load(data_file)
    dic_ = {}
    for e in raw_data: # iterates documents
        doc_id = e['id']
        sentences = []
        labels = []
        for a in e['annotations']:
            for i, r in enumerate(a['result']): # iterates sentences
                s = r['value']['text'].replace('\n', '')
                sentences.append(s)
                labels.append(r['value']['labels'][0])
                if len(r['value']['labels']) > 1:
                    raise ValueError(f'Sentence with multiple labels. Doc {doc_id}, {i}-th sentence.')
        if len(sentences) != 0: # ignores documents wihtout sentences
            df = pd.DataFrame(list(zip(sentences, labels)), columns=['sentence', 'label'])
            dic_[doc_id] = df
    return dic_
