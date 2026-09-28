
# 1. IMPORTS
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizer,                    # for BERT base
    BertForSequenceClassification,    # for BERT base model
    AutoTokenizer,                    # for FinBERT
    AutoModelForSequenceClassification) # for FinBERT model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import warnings
warnings.filterwarnings('ignore')
print("All imports done.")

# 2. LOAD DATA

df = pd.read_csv(
    r"C:\Users\alexa\Downloads\financial_sentiment_analysis.csv",
    encoding='latin-1',
    header=None,                        # no header row in this file
    names=['Sentiment', 'News Headline'] # assign column names directly
)

print(f"Shape:   {df.shape}")
print(f"Columns: {list(df.columns)}")
print(df.head(3))

# 3. CLASS DISTRIBUTION PLOT
counts  = df['Sentiment'].value_counts()
palette = {'positive': '#2ecc71', 'negative': '#e74c3c', 'neutral': '#7B9FCF'}

plt.figure(figsize=(5, 3))
sns.barplot(x=counts.index, y=counts.values, palette=palette)
plt.title('Class Distribution — Financial PhraseBank')
plt.xlabel('Sentiment')
plt.ylabel('Count')
for i, v in enumerate(counts.values):
    plt.text(i, v + 30, f'{v:,}', ha='center', fontweight='bold')
plt.tight_layout()
plt.show()
print(counts)

# 4. HEADLINE LENGTH DISTRIBUTION
df['word_count'] = df['News Headline'].str.split().str.len()

plt.figure(figsize=(7, 3))
sns.histplot(
    data=df,
    x='word_count',
    hue='Sentiment',
    bins=50,
    alpha=0.6,
    palette={'positive': '#2ecc71', 'negative': '#e74c3c', 'neutral': '#7B9FCF'}
)
plt.axvline(x=128, color='black', linestyle='--', linewidth=1.5, label='MAX_LEN=128')
plt.title('Headline Length Distribution by Sentiment')
plt.xlabel('Number of Words')
plt.legend()
plt.tight_layout()
plt.show()

print(df['word_count'].describe().round(1))

# 5. SAMPLE SENTENCES PER CLASS
for label in ['positive', 'negative', 'neutral']:
    sample = df[df['Sentiment'] == label]['News Headline'].iloc[0]
    print(f"\n[{label.upper()}]")
    print(sample[:300], '...')
    print('-' * 60)

# 6. CLEAN AND PREPARE
def clean_text(text):
    text = re.sub(r'<[^>]+>', ' ', str(text))  # remove HTML tags
    text = re.sub(r'\s+', ' ', text)            # collapse whitespace
    return text.strip()

df['text'] = df['News Headline'].apply(clean_text)

# Label encoding 
df['label'] = df['Sentiment'].map({
    'negative': 0,
    'neutral':  1,
    'positive': 2
})

# Label lookup dictionaries used throughout the notebook
name_to_label = {'negative': 0, 'neutral': 1, 'positive': 2}
label_to_name = {0: 'negative', 1: 'neutral',  2: 'positive'}

# Drop any rows with missing values after mapping
df.dropna(subset=['text', 'label'], inplace=True)
df['label'] = df['label'].astype(int)

print(f"Clean dataset: {len(df):,} rows")
print(f"Label counts:  {df['label'].value_counts().to_dict()}")
print()

print("Before cleaning:", df['News Headline'].iloc[0][:100])
print("After cleaning: ", df['text'].iloc[0][:100])

# TRAIN / TEST SPLIT WITH CLASS WEIGHTS

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter
import numpy as np
import torch

# Sample sizes
TRAIN_SIZE = 1268
TEST_SIZE  = 544

#SAMPLE A BALANCED SUBSET PER CLASS

SAMPLES_PER_CLASS_TRAIN = TRAIN_SIZE // 3   # ~666 per class for train
SAMPLES_PER_CLASS_TEST  = TEST_SIZE  // 3   # ~166 per class for test
SAMPLES_PER_CLASS_TOTAL = SAMPLES_PER_CLASS_TRAIN + SAMPLES_PER_CLASS_TEST

print("Samples requested per class:")
print(f"  Train portion: {SAMPLES_PER_CLASS_TRAIN}")
print(f"  Test portion:  {SAMPLES_PER_CLASS_TEST}")
print(f"  Total per class: {SAMPLES_PER_CLASS_TOTAL}")
print()

print("Available samples per class:")
print(f"  Negative (0): {(df['label'] == 0).sum()}")
print(f"  Neutral  (1): {(df['label'] == 1).sum()}")
print(f"  Positive (2): {(df['label'] == 2).sum()}")
print()

# Sample each class
neg  = df[df['label'] == 0].sample(SAMPLES_PER_CLASS_TOTAL, random_state=42)
neu  = df[df['label'] == 1].sample(SAMPLES_PER_CLASS_TOTAL, random_state=42)
pos  = df[df['label'] == 2].sample(SAMPLES_PER_CLASS_TOTAL, random_state=42)

subset = pd.concat([neg, neu, pos]) \
           .sample(frac=1, random_state=42) \
           .reset_index(drop=True)

texts  = subset['text'].values
labels = subset['label'].values

print(f"Total subset size: {len(subset):,}")
print(f"Subset distribution: {Counter(labels)}")
print()

# STEP 2 — STRATIFIED TRAIN / TEST SPLIT

train_texts, test_texts, train_labels, test_labels = train_test_split(
    texts,
    labels,
    test_size=TEST_SIZE / len(subset),
    random_state=42,
    stratify=labels       
)

test_df = pd.DataFrame({
    'text':  test_texts,
    'label': test_labels
})
test_df['sentiment_str'] = test_df['label'].map(label_to_name)

print(f"Train size: {len(train_texts):,}")
print(f"Test size:  {len(test_texts):,}")
print()

# Verify class distribution in each split
train_counts = Counter(train_labels)
test_counts  = Counter(test_labels)

print("--- Class Distribution After Split ---")
print(f"{'Class':<12} {'Train':>8} {'Test':>8}")
print("-" * 32)
for label in [0, 1, 2]:
    print(f"{label_to_name[label]:<12} {train_counts[label]:>8} {test_counts[label]:>8}")
print()

# COMPUTE CLASS WEIGHTS FOR LOSS FUNCTION

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.array([0, 1, 2]),
    y=train_labels
)

weights_tensor = torch.tensor(class_weights, dtype=torch.float)

print("Class weights for loss function:")
print(f"  Negative (0): {class_weights[0]:.4f}")
print(f"  Neutral  (1): {class_weights[1]:.4f}")
print(f"  Positive (2): {class_weights[2]:.4f}")
print()

# VISUALISE THE SPLIT

fig, axes = plt.subplots(1, 3, figsize=(13, 4))

datasets = {
    'Full Dataset':    Counter(df['label']),
    'Train Set':       train_counts,
    'Test Set':        test_counts
}

colors = ['#e74c3c', '#7B9FCF', '#2ecc71']  # neg, neu, pos

for ax, (title, counts) in zip(axes, datasets.items()):
    values = [counts.get(i, 0) for i in [0, 1, 2]]
    bars   = ax.bar(['Negative', 'Neutral', 'Positive'], values, color=colors)
    ax.set_title(title)
    ax.set_ylabel('Count')
    ax.set_xlabel('Sentiment')
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 5,
            str(val),
            ha='center',
            fontweight='bold'
        )

plt.suptitle('Class Distribution Across Dataset Splits', fontweight='bold')
plt.tight_layout()
plt.show()

# ============================================================
# STEP 5 — SUMMARY TABLE
# ============================================================

summary = pd.DataFrame({
    'Class':      ['Negative', 'Neutral', 'Positive', 'TOTAL'],
    'Full Dataset': [
        (df['label'] == 0).sum(),
        (df['label'] == 1).sum(),
        (df['label'] == 2).sum(),
        len(df)
    ],
    'Train Set': [
        train_counts[0],
        train_counts[1],
        train_counts[2],
        len(train_labels)
    ],
    'Test Set': [
        test_counts[0],
        test_counts[1],
        test_counts[2],
        len(test_labels)
    ],
    'Class Weight': [
        round(class_weights[0], 4),
        round(class_weights[1], 4),
        round(class_weights[2], 4),
        '-'
    ]
})

print("=" * 60)
print("FINAL SPLIT SUMMARY")
print("=" * 60)
print(summary.to_string(index=False))
print("=" * 60)

#tokenise 

MAX_LEN   = 128   
BATCH_SIZE = 16

# Load tokenizer — downloads on first run, cached after
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

print("Tokenizer loaded.")
print(f"Vocabulary size: {tokenizer.vocab_size:,}")

# FinBERT tokeniser
tokenizer_finbert = AutoTokenizer.from_pretrained('ProsusAI/finbert')

print("FinBERT Tokeniser loaded.")
print(f"Vocabulary size: {tokenizer_finbert.vocab_size:,}")

#tokenize BERT base
print("Tokenising train set...")
train_encodings = tokenizer(
    list(train_texts),
    max_length=MAX_LEN,
    padding='max_length',  
    truncation=True,       
    return_tensors='pt'     
)

print("Tokenising test set...")
test_encodings = tokenizer(
    list(test_texts),
    max_length=MAX_LEN,
    padding='max_length',
    truncation=True,
    return_tensors='pt'
)

print()
print(f"train_encodings['input_ids'] shape: {train_encodings['input_ids'].shape}")
# (2000, 128) — 2000 reviews, each 128 tokens

print(f"test_encodings['input_ids'] shape:  {test_encodings['input_ids'].shape}")

#tokenize finBERT
print("Tokenising train set...")
train_encodings = tokenizer_finbert(
    list(train_texts),
    max_length=MAX_LEN,
    padding='max_length',  
    truncation=True,       
    return_tensors='pt'     
)
print("Tokenising test set...")
test_encodings = tokenizer_finbert(
    list(test_texts),
    max_length=MAX_LEN,
    padding='max_length',
    truncation=True,
    return_tensors='pt'
)
print()
print(f"train_encodings['input_ids'] shape: {train_encodings['input_ids'].shape}")
print(f"test_encodings['input_ids'] shape:  {test_encodings['input_ids'].shape}")

# Simple Dataset class — wraps the tokenized data
class ReviewDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels    = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Return one sample: the tokenized inputs + its label
        return {
            'input_ids':      self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'labels':         torch.tensor(self.labels[idx], dtype=torch.long)
        }

# Create datasets
train_dataset = ReviewDataset(train_encodings, train_labels)
test_dataset  = ReviewDataset(test_encodings,  test_labels)

# Create DataLoaders — batches the data automatically
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

print(f"train_loader: {len(train_loader)} batches of {BATCH_SIZE}")
print(f"test_loader:  {len(test_loader)} batches of {BATCH_SIZE}")


# LOAD MODELS — BERT BASE AND FINBERT

model_bert = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=3
)

# FinBERT — same architecture, different pre-training
model_finbert = AutoModelForSequenceClassification.from_pretrained(
    'ProsusAI/finbert',
    num_labels=3
)

# Optimizers — one per model, identical hyperparameters for fair comparison
optimizer_bert    = torch.optim.AdamW(model_bert.parameters(),    lr=2e-5)
optimizer_finbert = torch.optim.AdamW(model_finbert.parameters(), lr=2e-5)

# Parameter counts
n_params_bert    = sum(p.numel() for p in model_bert.parameters())
n_params_finbert = sum(p.numel() for p in model_finbert.parameters())

print(f"BERT base:  bert-base-uncased + Linear(768→3) head")
print(f"Parameters: {n_params_bert:,}")
print()
print(f"FinBERT:    ProsusAI/finbert + Linear(768→3) head")
print(f"Parameters: {n_params_finbert:,}")
print()
print(f"Optimizer:  AdamW, lr=2e-5 (both models)")

# ============================================================
# TRAINING LOOP — BERT BASE AND FINBERT
# ============================================================

EPOCHS = 4

def train_model(model, optimizer, train_loader, train_dataset, model_name):
    """
    Runs the full 4-epoch training loop for a given model.
    Returns epoch losses and accuracies for plotting.
    """
    epoch_train_losses = []
    epoch_train_accs   = []

    print(f"\n{'=' * 60}")
    print(f"TRAINING: {model_name}")
    print(f"{'=' * 60}")
    print(f"Training on {len(train_dataset):,} samples for {EPOCHS} epochs...")
    print(f"Total batches per epoch: {len(train_loader)}")

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1} of {EPOCHS}")
        print("-" * 60)

        model.train()

        total_loss    = 0
        total_correct = 0
        total_samples = 0
        batch_losses  = []

        for batch_num, batch in enumerate(train_loader):
            input_ids      = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels         = batch['labels']

            optimizer.zero_grad()

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss

            loss.backward()
            optimizer.step()

            total_loss    += loss.item()
            preds          = outputs.logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
            batch_losses.append(loss.item())

            if (batch_num + 1) % 25 == 0 or (batch_num + 1) == len(train_loader):
                avg_loss = total_loss / (batch_num + 1)
                acc      = total_correct / total_samples * 100
                print(f"  Batch {batch_num+1:>4}/{len(train_loader)}  "
                      f"loss={avg_loss:.4f}  acc={acc:.1f}%")

        final_train_loss = total_loss / len(train_loader)
        final_train_acc  = total_correct / total_samples * 100
        epoch_train_losses.append(final_train_loss)
        epoch_train_accs.append(final_train_acc)

        print(f"\n  Epoch {epoch + 1} complete — "
              f"Loss: {final_train_loss:.4f}  "
              f"Accuracy: {final_train_acc:.1f}%")

    print(f"\n{'=' * 60}")
    print(f"TRAINING COMPLETE — {model_name}")
    print(f"{'=' * 60}")
    for i, (loss, acc) in enumerate(zip(epoch_train_losses, epoch_train_accs)):
        print(f"  Epoch {i+1}:  loss={loss:.4f}  acc={acc:.1f}%")

    return epoch_train_losses, epoch_train_accs


# ============================================================
# RUN TRAINING FOR BOTH MODELS
# ============================================================

bert_losses, bert_accs = train_model(
    model=model_bert,
    optimizer=optimizer_bert,
    train_loader=train_loader,
    train_dataset=train_dataset,
    model_name="BERT Base (bert-base-uncased)"
)

finbert_losses, finbert_accs = train_model(
    model=model_finbert,
    optimizer=optimizer_finbert,
    train_loader=train_loader,
    train_dataset=train_dataset,
    model_name="FinBERT (ProsusAI/finbert)"
)

# ============================================================
# PLOT TRAINING CURVES — BOTH MODELS SIDE BY SIDE
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Loss curves
axes[0].plot(range(1, EPOCHS + 1), bert_losses,
             marker='o', color='#4C72B0', linewidth=2, label='BERT Base')
axes[0].plot(range(1, EPOCHS + 1), finbert_losses,
             marker='o', color='#DD8452', linewidth=2, label='FinBERT')
axes[0].set_title('Training Loss per Epoch')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_xticks(range(1, EPOCHS + 1))
axes[0].legend()
axes[0].grid(alpha=0.3)

# Accuracy curves
axes[1].plot(range(1, EPOCHS + 1), bert_accs,
             marker='o', color='#4C72B0', linewidth=2, label='BERT Base')
axes[1].plot(range(1, EPOCHS + 1), finbert_accs,
             marker='o', color='#DD8452', linewidth=2, label='FinBERT')
axes[1].set_title('Training Accuracy per Epoch')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy (%)')
axes[1].set_xticks(range(1, EPOCHS + 1))
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.suptitle('Training Curves — BERT Base vs FinBERT', fontweight='bold')
plt.tight_layout()
plt.show()

# EVALUATION — BERT BASE

model_bert.eval()

all_preds_bert  = []
all_labels_bert = []

print("Evaluating BERT Base on test set...")

with torch.no_grad():
    for batch in test_loader:
        outputs = model_bert(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask']
        )
        preds = outputs.logits.argmax(dim=1)
        all_preds_bert.extend(preds.numpy())
        all_labels_bert.extend(batch['labels'].numpy())

all_preds_bert  = np.array(all_preds_bert)
all_labels_bert = np.array(all_labels_bert)

accuracy_bert = accuracy_score(all_labels_bert, all_preds_bert)
f1_bert       = f1_score(all_labels_bert, all_preds_bert, average='macro')

print(f"\nBERT Base Test Accuracy: {accuracy_bert*100:.2f}%")
print(f"BERT Base F1 Score:      {f1_bert:.4f}")
print()
print(classification_report(all_labels_bert, all_preds_bert, target_names=['Negative', 'Neutral', 'Positive']))


# EVALUATION — FINBERT

model_finbert.eval()

all_preds_finbert  = []
all_labels_finbert = []

print("Evaluating FinBERT on test set...")

with torch.no_grad():
    for batch in test_loader:
        outputs = model_finbert(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask']
        )
        preds = outputs.logits.argmax(dim=1)
        all_preds_finbert.extend(preds.numpy())
        all_labels_finbert.extend(batch['labels'].numpy())

all_preds_finbert  = np.array(all_preds_finbert)
all_labels_finbert = np.array(all_labels_finbert)

accuracy_finbert = accuracy_score(all_labels_finbert, all_preds_finbert)
f1_finbert       = f1_score(all_labels_finbert, all_preds_finbert, average='macro')

print(f"\nFinBERT Test Accuracy: {accuracy_finbert*100:.2f}%")
print(f"FinBERT F1 Score:      {f1_finbert:.4f}")
print()
print(classification_report(all_labels_finbert, all_preds_finbert, target_names=['Negative', 'Neutral', 'Positive']))

# CONFUSION MATRIX — BERT BASE

cm_bert = confusion_matrix(all_labels_bert, all_preds_bert)

plt.figure(figsize=(5, 4))
sns.heatmap(
    cm_bert,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['Negative', 'Neutral', 'Positive'],
    yticklabels=['Negative', 'Neutral', 'Positive']
)
plt.title(f'Confusion Matrix — BERT Base | Accuracy: {accuracy_bert*100:.1f}%')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.show()


# CONFUSION MATRIX — FINBERT

cm_finbert = confusion_matrix(all_labels_finbert, all_preds_finbert)

plt.figure(figsize=(5, 4))
sns.heatmap(
    cm_finbert,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['Negative', 'Neutral', 'Positive'],
    yticklabels=['Negative', 'Neutral', 'Positive']
)
plt.title(f'Confusion Matrix — FinBERT | Accuracy: {accuracy_finbert*100:.1f}%')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.show()

# Predict on new text 
def predict(text):
    """
    Predict sentiment for any text string.
    Returns: label ('POSITIVE' or 'NEGATIVE') and confidence score.
    """
    # Clean text
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    # Tokenise — the tokenizer handles [CLS], [SEP], padding automatically
    enc = tokenizer(
        text,
        max_length=MAX_LEN,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
# PREDICT FUNCTION — BERT BASE

def predict_bert(text):
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    enc = tokenizer(
        text,
        max_length=MAX_LEN,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    model_bert.eval()
    with torch.no_grad():
        outputs = model_bert(**enc)

    probs      = torch.softmax(outputs.logits, dim=1)[0]
    pred_id    = probs.argmax().item()
    label      = label_to_name[pred_id].upper()
    confidence = probs[pred_id].item()
    return label, confidence

print("predict_bert() defined. Usage: label, conf = predict_bert('text')")


# PREDICT FUNCTION — FINBERT

def predict_finbert(text):
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    enc = tokenizer_finbert(
        text,
        max_length=MAX_LEN,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    model_finbert.eval()
    with torch.no_grad():
        outputs = model_finbert(**enc)

    probs      = torch.softmax(outputs.logits, dim=1)[0]
    pred_id    = probs.argmax().item()
    label      = label_to_name[pred_id].upper()
    confidence = probs[pred_id].item()
    return label, confidence

print("predict_finbert() defined. Usage: label, conf = predict_finbert('text')")


# TEST BOTH ON SAMPLE HEADLINES

test_headlines = [
    "Company reports record profits for third consecutive quarter",
    "Firm faces bankruptcy after massive fraud scandal",
    "Markets remain stable ahead of interest rate decision"
]

print("\n" + "=" * 70)
print("BERT BASE PREDICTIONS")
print(f"{'Prediction':<12} {'Confidence':>11}  Headline")
print("-" * 70)
for headline in test_headlines:
    label, conf = predict_bert(headline)
    mark    = '+' if label == 'POSITIVE' else '-'
    display = headline[:55] + '...' if len(headline) > 55 else headline
    print(f"[{mark}] {label:<10} {conf*100:>9.1f}%   {display}")

print("\n" + "=" * 70)
print("FINBERT PREDICTIONS")
print(f"{'Prediction':<12} {'Confidence':>11}  Headline")
print("-" * 70)
for headline in test_headlines:
    label, conf = predict_finbert(headline)
    mark    = '+' if label == 'POSITIVE' else '-'
    display = headline[:55] + '...' if len(headline) > 55 else headline
    print(f"[{mark}] {label:<10} {conf*100:>9.1f}%   {display}")
    
# GEMMA 3 IMPLEMENTATION
# GEMMA3 VIA OLLAMA — SENTIMENT ANALYSIS
import requests
import time

def get_sentiment_gemma(text):
    prompt = f"""
    Classify the sentiment of the following financial news headline.
    Reply with exactly one word: Positive, Negative, or Neutral.

    Text: "{text}"
    """

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "gemma3",
            "prompt": prompt,
            "stream": False
        }
    )

    # Print the response to debug
    print("API Response:", response.text)

    response_json = response.json()
    if "response" in response_json:
        return response_json["response"].strip()
    else:
        print("Error: 'response' key not found in API response")
        print("Available keys:", response_json.keys())
        return "Unknown"


# ============================================================
# TEST EXAMPLE
# ============================================================

texts = [
    "Company reports record profits for third consecutive quarter",
    "Firm faces bankruptcy after massive fraud scandal",
    "Markets remain stable ahead of interest rate decision"
]

for t in texts:
    print("Text:", t)
    print("Sentiment:", get_sentiment_gemma(t))
    print("-" * 50)


# ============================================================
# EVALUATE GEMMA3 ON TEST SET
# ============================================================

def parse_label(response_text):
    """
    Parse Gemma3 response into 0, 1, 2 matching label_to_name.
    Returns -1 if the response is unrecognisable.
    """
    text = response_text.lower().strip()
    if 'positive' in text:
        return 2
    elif 'negative' in text:
        return 0
    elif 'neutral' in text:
        return 1
    else:
        return -1


all_preds_gemma  = []
all_labels_gemma = []

print("\n" + "=" * 60)
print("EVALUATING GEMMA3 ON TEST SET")
print(f"Total samples: {len(test_texts)}")
print("=" * 60)

skipped = 0

for i, (text, true_label) in enumerate(zip(test_texts, test_labels)):
    raw_response = get_sentiment_gemma(text)
    pred_label   = parse_label(raw_response)

    if pred_label == -1:
        print(f"  [SKIP] Unrecognised response: '{raw_response}' for: {text[:50]}")
        skipped += 1
        continue

    all_preds_gemma.append(pred_label)
    all_labels_gemma.append(true_label)

    if (i + 1) % 50 == 0:
        print(f"  Processed {i+1}/{len(test_texts)} samples...")

    time.sleep(0.1)  # small delay to avoid overwhelming Ollama

print(f"\nEvaluation complete. Skipped {skipped} unrecognisable responses.")
print(f"Evaluated on {len(all_preds_gemma)} samples.")

all_preds_gemma  = np.array(all_preds_gemma)
all_labels_gemma = np.array(all_labels_gemma)


# ============================================================
# METRICS — GEMMA3
# ============================================================

accuracy_gemma = accuracy_score(all_labels_gemma, all_preds_gemma)
f1_gemma       = f1_score(all_labels_gemma, all_preds_gemma, average='macro')

print(f"\nGemma3 Test Accuracy: {accuracy_gemma*100:.2f}%")
print(f"Gemma3 F1 Score:      {f1_gemma:.4f}")
print()
print(classification_report(all_labels_gemma, all_preds_gemma, target_names=['Negative', 'Neutral', 'Positive']))


# ============================================================
# CONFUSION MATRIX — GEMMA3
# ============================================================

cm_gemma = confusion_matrix(all_labels_gemma, all_preds_gemma)

plt.figure(figsize=(5, 4))
sns.heatmap(
    cm_gemma,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['Negative', 'Neutral', 'Positive'],
    yticklabels=['Negative', 'Neutral', 'Positive']
)
plt.title(f'Confusion Matrix — Gemma3 | Accuracy: {accuracy_gemma*100:.1f}%')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.show()


# ============================================================
# FINAL COMPARISON — BERT BASE vs FINBERT vs GEMMA3
# ============================================================

comparison_df = pd.DataFrame([
    {
        "Model":        "BERT Base",
        "Accuracy (%)": round(accuracy_bert * 100, 2),
        "F1 Macro":     round(f1_bert, 4)
    },
    {
        "Model":        "FinBERT",
        "Accuracy (%)": round(accuracy_finbert * 100, 2),
        "F1 Macro":     round(f1_finbert, 4)
    },
    {
        "Model":        "Gemma3",
        "Accuracy (%)": round(accuracy_gemma * 100, 2),
        "F1 Macro":     round(f1_gemma, 4)
    }
])

print("\n")
print("=" * 50)
print("FINAL MODEL COMPARISON")
print("=" * 50)
print(comparison_df.to_string(index=False))
print("=" * 50)

# Bar chart comparison
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
metrics = ["Accuracy (%)", "F1 Macro"]
colors  = ['#4C72B0', '#DD8452', '#2ecc71']

for ax, metric in zip(axes, metrics):
    values = comparison_df[metric].values
    bars   = ax.bar(["BERT Base", "FinBERT", "Gemma3"], values, color=colors)
    ax.set_title(metric)
    ax.set_ylabel(metric)
    ax.set_ylim(0, max(values) * 1.15)
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{val:.2f}",
            ha='center',
            fontweight='bold'
        )

plt.suptitle('BERT Base vs FinBERT vs Gemma3 — Test Set Performance', fontweight='bold')
plt.tight_layout()
plt.show()
    