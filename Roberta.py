import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset

data = pd.read_csv('merged_mgh_bidmc.csv')

data['combined_text'] = data['icd'].astype(str) + data['med'].astype(str) + data['report_text'].astype(str)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

# Evaluation metrics
def evaluate_model(model, dataloader, device):
    model.eval()
    y_true = []
    y_pred_probs = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask, labels_batch = batch
            input_ids = input_ids.to(device)  # Move input tensors to the desired device (GPU)
            attention_mask = attention_mask.to(device)
            labels_batch = labels_batch.to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predictions = torch.sigmoid(logits)  # Apply sigmoid activation for binary classification
            y_true.extend(labels_batch.tolist())
            y_pred_probs.extend(predictions[:, 1].tolist())  # Use probabilities for the positive class (index 1)

    y_pred = [1 if prob >= 0.5 else 0 for prob in y_pred_probs]  # Set threshold to convert to binary predictions

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred_probs)  # Use probabilities for the positive class
    cm = confusion_matrix(y_true, y_pred)
    return accuracy, precision, recall, f1, roc_auc, cm

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
# Tokenize the text data and convert it into input tensors
encodings = tokenizer(list(data['combined_text']), truncation=True, padding=True, max_length=512, return_tensors='pt')
# Convert labels to tensors and move to the desired device (GPU)
labels_tensor = torch.tensor(list(data['annot'])).long().to(device)
# Create TensorDataset and DataLoader for the entire dataset
dataset = TensorDataset(encodings['input_ids'], encodings['attention_mask'], labels_tensor)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
# Define the model
model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)
model = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
loss_fn = torch.nn.CrossEntropyLoss()
# Perform 10-fold cross-validation
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
metrics = []

for fold, (train_index, val_index) in enumerate(skf.split(data['combined_text'], data['annot'])):
    print(f"Fold {fold + 1}")

    train_data_fold = data.iloc[train_index]
    val_data_fold = data.iloc[val_index]

    train_encodings_fold = tokenizer(list(train_data_fold['combined_text']), truncation=True, padding=True, max_length=512, return_tensors='pt')
    val_encodings_fold = tokenizer(list(val_data_fold['combined_text']), truncation=True, padding=True, max_length=512, return_tensors='pt')

    # Convert labels to tensors and move to the desired device (GPU or CPU)
    train_labels_tensor_fold = torch.tensor(list(train_data_fold['annot'])).long().to(device)
    val_labels_tensor_fold = torch.tensor(list(val_data_fold['annot'])).long().to(device)

    train_dataset_fold = TensorDataset(train_encodings_fold['input_ids'], train_encodings_fold['attention_mask'], train_labels_tensor_fold)
    val_dataset_fold = TensorDataset(val_encodings_fold['input_ids'], val_encodings_fold['attention_mask'], val_labels_tensor_fold)

    train_dataloader_fold = DataLoader(train_dataset_fold, batch_size=32, shuffle=True)
    val_dataloader_fold = DataLoader(val_dataset_fold, batch_size=32)

    # Inside the loop for cross-validation
    model.train()
    for epoch in range(3):
        for batch in train_dataloader_fold:
            input_ids, attention_mask, labels_batch = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels_batch = labels_batch.to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels_batch)
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    accuracy, precision, recall, f1, roc_auc, cm = evaluate_model(model, val_dataloader_fold, device)
    metrics.append((accuracy, precision, recall, f1, roc_auc, cm))

# Calculate average metrics over all folds
import numpy as np
# Convert the metrics list to a NumPy array
metrics_array = np.array(metrics)
# Calculate the mean along the rows (dimension 0) for accuracy, precision, recall, F1 score, and ROC AUC
avg_metrics = metrics_array[:, :5].mean(axis=0).tolist()
# Calculate the mean confusion matrix separately
confusion_matrix = metrics_array[:, 5].sum(axis=0)
confusion_matrix1 = metrics_array[:, 5]

# Print evaluation results
print("Accuracy:", avg_metrics[0])
print("Precision:", avg_metrics[1])
print("Recall:", avg_metrics[2])
print("F1 Score:", avg_metrics[3])
print("AUC:", avg_metrics[4])
print("Confusion Matrix:")
print(confusion_matrix)
print(confusion_matrix1)