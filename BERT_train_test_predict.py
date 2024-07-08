# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score
from torch import nn
from tqdm import tqdm
from prettytable import PrettyTable

# Set seed for torch and numpy
seed = 1

# Torch RNG
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def train(PATH, epochs, model, bert_type, optimizer, lr_scheduler, train_on_gpu, train_dataloader, validation_dataloader):
    """
     Train a BERT model with the specified parameters.
    
     Args:
         PATH (str): Path to save the best model.
         epochs (int): Number of epochs to train the model.
         model (torch.nn.Module): The model to be trained.
         bert_type (str): Type of BERT model ('LSTM' or 'Sequence').
         optimizer (torch.optim.Optimizer): Optimizer for the training process.
         lr_scheduler (torch.optim.lr_scheduler): Learning rate scheduler.
         train_on_gpu (bool): If True, train the model on GPU.
         train_dataloader (torch.utils.data.DataLoader): Dataloader for training data.
         validation_dataloader (torch.utils.data.DataLoader): Dataloader for validation data.
    
     Returns:
         None
     """    
    # Sent model to GPU
    if(train_on_gpu):
        model.cuda()
    
    # Initialize Table for showing training results      
    table = PrettyTable(['Epoch', 'Train Loss', 'Val Loss', 'Train F1', 'Val F1',
                         'Train Acc.', 'Val Acc.', 'Train Bal. Acc.', 
                         'Val. Bal Acc.', 'Best Model'])
    
    # Initialize vectors for F1 scores and losses    
    avg_f1_score_list = []
    avg_f1_score_val_list = []
    #train_loss_set = []
    val_loss_set = []
    
    # Initialize start value for best F1 score
    best_val_f1 = 0
    
    # Initialize number of epochs (The authors recommend 2 - 4 epochs)
    n_epochs = range(1,epochs+1)
    
    # Initialized loss function
    loss_function = nn.CrossEntropyLoss()
    
    for epoch in n_epochs:  
      
        # Training
        
        # Set our model to training mode (as opposed to evaluation mode)
        model.train()
        
        # Tracking variables
        avg_val_loss = 0
        avg_train_loss = 0
        avg_f1_score = 0
        avg_f1_score_val = 0
        avg_train_ac = 0
        avg_val_ac = 0
        avg_bal_train_ac = 0
        avg_bal_val_ac = 0
      
        # Train the data for one epoch
        for step, batch in enumerate(tqdm(train_dataloader)):
            # Add batch to GPU
            # Unpack the inputs from our dataloader
            if train_on_gpu:
                batch = tuple(t.cuda() for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            
            # Clear out the gradients
            optimizer.zero_grad()
            # Forward pass
            if bert_type == 'LSTM':
                output = model(input_ids = b_input_ids, attention_mask=b_input_mask)
            if bert_type == 'Sequence':
                output = model(input_ids = b_input_ids, attention_mask=b_input_mask)
                output = output.logits
            
            train_loss = loss_function(output, b_labels)
            #train_loss.backward()
            
            avg_train_loss += train_loss.item()
            
            logits = F.softmax(output, dim = 1)
            
            preds = torch.max(logits, dim=1)[1]
            
            #train_loss_set.append(train_loss.item())
            # Backward pass
            train_loss.backward()
              
            # calculate f1 mackro score, accuracy and balaned accuracy
            avg_f1_score += f1_score(b_labels.cpu(), preds.cpu(), average = 'macro')
            avg_train_ac += accuracy_score(b_labels.cpu(), preds.cpu())
            avg_bal_train_ac += balanced_accuracy_score(b_labels.cpu(), preds.cpu())
          
            # Update parameters and take a step using the computed gradient
            optimizer.step()
            
            if lr_scheduler is not None:
                lr_scheduler.step()
          
            # Update tracking variables
            #avg_train_loss += output.loss.item()
        
        avg_train_loss /= len(train_dataloader)
        avg_train_ac /= len(train_dataloader)
        avg_bal_train_ac /= len(train_dataloader)
        avg_f1_score /= len(train_dataloader)
        
        avg_f1_score_list.append(avg_f1_score)
          
        # Validation
        
        # Put model in evaluation mode to evaluate loss on the validation set
        model.eval()
    
        # Evaluate data for one epoch
        for batch in tqdm(validation_dataloader):
            # Add batch to GPU
            if train_on_gpu:
                batch = tuple(t.cuda() for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch
            # Telling the model not to compute or store gradients, saving memory and speeding up validation
            with torch.no_grad():
              # Forward pass, calculate logit predictions
              if bert_type == 'LSTM':
                output = model(input_ids = b_input_ids, attention_mask=b_input_mask)
              if bert_type == 'Sequence':
                output = model(input_ids = b_input_ids, attention_mask=b_input_mask)
                output = output.logits
            
            logits = F.softmax(output, dim = 1)
            val_loss = loss_function(output, b_labels)
            
            preds = torch.max(logits, dim=1)[1]
            val_loss_set.append(val_loss.item())
        
             # calculate f1 mackro score, accuracy and balaned accuracy
            avg_f1_score_val += f1_score(b_labels.cpu(), preds.cpu(),  average = 'macro')
            avg_val_ac += accuracy_score(b_labels.cpu(), preds.cpu())
            avg_bal_val_ac += balanced_accuracy_score(b_labels.cpu(), preds.cpu())
            
            avg_val_loss += val_loss.item()
        
        avg_val_loss /= len(validation_dataloader)
        avg_val_ac /= len(validation_dataloader)
        avg_bal_val_ac /= len(validation_dataloader)
        avg_f1_score_val /= len(validation_dataloader)
              
        avg_f1_score_val_list.append(avg_f1_score_val)
        
        if avg_f1_score_val > best_val_f1:
                  best_val_f1 = avg_f1_score_val
                  torch.save(model.state_dict(), PATH)
                  evaluated_epoch = epoch
              
        table.add_row(['{}/{}'.format(epoch, list(n_epochs)[-1]), 
          round(avg_train_loss,4), round(avg_val_loss,4),round(avg_f1_score,4),
          round(avg_f1_score_val,4), round(avg_train_ac,4), round(avg_val_ac,4),
          round(avg_bal_train_ac,4),round(avg_bal_val_ac,4),evaluated_epoch])
        
    print(table)
        
    plt.plot(avg_f1_score_list)
    plt.plot(avg_f1_score_val_list)
    plt.xlabel('n epochs')
    plt.ylabel('f1 macro')
    plt.legend(['train', 'val'])
    plt.show()
    
def test(model, bert_type, train_on_gpu, test_dataloader, num_classes = 3):
    """
    Test the trained model on the test dataset.
    
    Args:
        model (torch.nn.Module): The trained model.
        bert_type (str): Type of BERT model ('LSTM' or 'Sequence').
        train_on_gpu (bool): If True, test the model on GPU.
        test_dataloader (torch.utils.data.DataLoader): Dataloader for test data.
        num_classes (int, optional): Number of classes in the dataset. Defaults to 3.
    
    Returns:
        float: The F1 score of the test dataset.
    """
    # Sent model to GPU
    if(train_on_gpu):
        model.cuda()
    
    model.eval()

    true_label = []
    pred_label = []
    test_loss_set = []
    avg_test_loss = 0
    
    # Initialized loss function
    loss_function = nn.CrossEntropyLoss()
    
    # iterate over test data
    for batch in tqdm(test_dataloader):
    
        if train_on_gpu:
            batch = tuple(t.cuda() for t in batch)
        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch
        # Telling the model not to compute or store gradients, saving memory and speeding up validation
        with torch.no_grad():
          # Forward pass, calculate logit predictions
          if bert_type == 'LSTM':
              output = model(input_ids = b_input_ids, attention_mask=b_input_mask)
          if bert_type == 'Sequence':
              output = model(input_ids = b_input_ids, attention_mask=b_input_mask)
              output = output.logits
        
        logits = F.softmax(output, dim = 1)
        test_loss = loss_function(output, b_labels)
        
        test_loss_set.append(test_loss.item())
        preds = torch.max(logits, dim=1)[1]
        
        pred_label.extend(preds.cpu().tolist())
        true_label.extend(b_labels.cpu().tolist())
        
        test_loss = loss_function(output, b_labels)
        avg_test_loss += test_loss.item()
     
    avg_test_loss /= len(test_dataloader)
    
    # f1 makro score over all test data    
    test_f1_score = f1_score(true_label, pred_label, average = 'macro')
    
    # balanced accuracy over all test data
    test_bal_acc = balanced_accuracy_score(true_label, pred_label)
    
    # accuracy over all test data
    test_acc = accuracy_score(true_label, pred_label)
     
    print('Test loss: {:.3f}'.format(avg_test_loss),
          'Test accuracy: {:.3f}'.format(test_acc),
          'Test balanced accuracy: {:.3f}'.format(test_bal_acc),
          'F1 Score: {:.3f}'.format(test_f1_score)) 
    
    # plot confusion matrix
    confn_matrix = confusion_matrix(pred_label, true_label)
    norm_c =  confn_matrix/confn_matrix.astype(float).sum(axis=0)
    df_cm = pd.DataFrame(norm_c, range(num_classes), range(num_classes))
    group_counts = ["{0:0.0f}".format(value) for value in confn_matrix.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in norm_c.flatten()]
    labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(num_classes,num_classes)
    if num_classes == 3:
        
        sn.heatmap(df_cm, annot=labels, annot_kws={"size": 12}, fmt="", cmap="Greys", cbar=False, xticklabels=["negative", "neutral", "positive"], yticklabels=["negative", "neutral", "postive"])
        
    elif num_classes == 2:
        
        sn.heatmap(df_cm, annot=labels, annot_kws={"size": 12}, fmt="", cmap="Greys", cbar=False, xticklabels=["negative", "positive"], yticklabels=["negative", "postive"])
        
    plt.xlabel("True Class")
    plt.ylabel("Predicted Class")
    plt.savefig(r'Figures\figureS1.svg', format='svg', dpi=500)
    plt.savefig(r'Figures\figureS1.eps', format='eps', dpi=500)
    plt.show()
    
    return(test_f1_score)
    
def predict(model, bert_type, train_on_gpu, dataloader):
    """
    Predict labels for the given dataset using the trained model.
    
    Args:
        model (torch.nn.Module): The trained model.
        bert_type (str): Type of BERT model ('LSTM' or 'Sequence').
        train_on_gpu (bool): If True, use GPU for prediction.
        dataloader (torch.utils.data.DataLoader): Dataloader for the dataset to predict.
    
    Returns:
        list: List of predicted labels.
    """
    #sent model to GPU
    if(train_on_gpu):
        model.cuda()  

    model.eval()
    
    pred_label = []
    
    # iterate over test data
    for step, batch in enumerate(tqdm(dataloader)):
    
        if train_on_gpu:
            batch = tuple(t.cuda() for t in batch)
        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask = batch
        # Telling the model not to compute or store gradients, saving memory and speeding up validation
        with torch.no_grad():
          # Forward pass, calculate logit predictions
          if bert_type == 'LSTM':
                output = model(input_ids = b_input_ids, attention_mask=b_input_mask)
          if bert_type == 'Sequence':
                output = model(input_ids = b_input_ids, attention_mask=b_input_mask)
                output = output.logits
        
        logits = F.softmax(output, dim = 1)
        
        #test_loss_set.append(output.loss.item())
        preds = torch.max(logits, dim=1)[1]
        
        pred_label.extend(preds.cpu().tolist())
        
    return(pred_label)