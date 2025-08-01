import random
import time
import numpy as np
import pandas as pd
from Bio import SeqIO
import torch
import torch.nn as nn
import torch.nn.functional as F  # 
import torch.optim as optim
from tqdm import tqdm
from itertools import product
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, \
    matthews_corrcoef
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import esm
import matplotlib.pyplot as plt
import os




def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_fasta(fasta_file):
    sequences = []
    labels = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        try:
            desc_parts = record.description.split('|')
            label = int(desc_parts[1])
            if label not in [0, 1]:
                raise ValueError("Label must be 0 or 1.")
        except (IndexError, ValueError) as e:
            print(f"Skipping sequence {record.id} due to parsing error: {e}")
            continue
        sequences.append(str(record.seq).upper())
        labels.append(label)
    return pd.DataFrame({'sequence': sequences, 'label': labels})


def calculate_aa_frequency(seq, amino_acids):
    seq_length = len(seq)
    if seq_length == 0:
        return [0] * len(amino_acids)
    freq = {aa: 0 for aa in amino_acids}
    for aa in seq:
        if aa in freq:
            freq[aa] += 1
    return [freq[aa] / seq_length for aa in amino_acids]


def calculate_dipeptide_frequency(seq, dipeptides):
    seq_length = len(seq)
    if seq_length <= 1:
        return [0] * len(dipeptides)
    dipep_count = {dp: 0 for dp in dipeptides}
    for i in range(seq_length - 1):
        dp = seq[i:i + 2]
        if dp in dipep_count:
            dipep_count[dp] += 1
    total = seq_length - 1
    return [dipep_count[dp] / total for dp in dipeptides]


def calculate_physicochemical_properties(seq):
    physicochemical_properties = {
        'hydrophobicity': {'A': -0.4, 'R': -0.59, 'N': -0.92, 'D': -1.31, 'C': -0.91, 'Q': -1.22, 'E': -0.67,
                           'G': -0.64, 'H': 1.25, 'I': 1.22, 'L': -0.67, 'K': 1.02, 'M': 1.92, 'F': -0.49, 'P': -0.55,
                           'S': -0.28, 'T': 0.5, 'W': 1.67, 'Y': 0.91, 'V': -0.92},
        'molecular_weight': {'A': 89, 'R': 174, 'N': 132, 'D': 133, 'C': 121, 'Q': 146, 'E': 147, 'G': 75, 'H': 155,
                             'I': 131, 'L': 131, 'K': 146, 'M': 149, 'F': 165, 'P': 115, 'S': 105, 'T': 119, 'W': 204,
                             'Y': 181, 'V': 117},
        'polarity': {'A': 8.1, 'R': 10.5, 'N': 11.6, 'D': 13, 'C': 5.5, 'Q': 10.5, 'E': 12.3, 'G': 9, 'H': 10.4,
                     'I': 5.2, 'L': 4.9, 'K': 11.3, 'M': 5.7, 'F': 5.2, 'P': 8, 'S': 9.2, 'T': 8.6, 'W': 5.4, 'Y': 6.2,
                     'V': 5.9},
        'pi': {'A': 6, 'R': 10.76, 'N': 5.41, 'D': 2.77, 'C': 5.07, 'Q': 5.65, 'E': 3.22, 'G': 5.97, 'H': 7.59,
               'I': 6.02, 'L': 5.98, 'K': 9.74, 'M': 5.74, 'F': 5.48, 'P': 6.3, 'S': 5.68, 'T': 5.6, 'W': 5.89,
               'Y': 5.66, 'V': 5.96},
        'average_flexibility': {'A': 0.36, 'R': 0.53, 'N': 0.46, 'D': 0.51, 'C': 0.35, 'Q': 0.49, 'E': 0.5, 'G': 0.54,
                                'H': 0.32, 'I': 0.46, 'L': 0.37, 'K': 0.47, 'M': 0.3, 'F': 0.31, 'P': 0.51, 'S': 0.51,
                                'T': 0.44, 'W': 0.31, 'Y': 0.42, 'V': 0.39},
        'Conformational parameter for alpha helix': {'A': 1.49, 'R': 1.224, 'N': 0.772, 'D': 0.966, 'C': 1.191, 'Q': 1.164, 'E': 1.504,
                                   'G': 0.54, 'H': 1.003, 'I': 1.003, 'L': 1.236, 'K': 1.172, 'M': 1.363, 'F': 1.195,
                                   'P': 0.492, 'S': 0.739, 'T': 0.785, 'W': 1.09, 'Y': 0.787, 'V': 0.99},
        'Conformational parameter for beta-sheet': {'A': 0.709, 'R': 0.92, 'N': 0.604, 'D': 0.541, 'C': 0.84, 'Q': 0.84, 'E': 0.657,
                                   'G': 0.567, 'H': 0.863, 'I': 1.799, 'L': 1.261, 'K': 0.721, 'M': 1.21, 'F': 1.393,
                                   'P': 0.354, 'S': 0.928, 'T': 1.221, 'W': 1.306, 'Y': 1.266, 'V': 1.965},
        'bulkiness': {'A': 11.5, 'R': 14.28, 'N': 12.82, 'D': 11.68, 'C': 13.46, 'Q': 14.45, 'E': 13.57, 'G': 3.4,
                      'H': 13.69, 'I': 21.4, 'L': 21.4, 'K': 15.71, 'M': 16.25, 'F': 19.8, 'P': 17.43, 'S': 9.47,
                      'T': 15.77, 'W': 21.67, 'Y': 18.03, 'V': 21.57},
        'charge': {'A': -0.02410542, 'R': 0.9758914, 'N': -0.02410542, 'D': -1.023312, 'C': -0.05475885,
                   'Q': -0.02410542, 'E': -1.022848, 'G': 0.2161477, 'H': -0.02410542, 'I': -0.02410542,
                   'L': 0.9757361, 'K': -0.02410542, 'M': -0.02410542, 'F': -0.02410542, 'P': -0.02410542,
                   'S': -0.02410542, 'T': -0.02410542, 'W': -0.02410542, 'Y': -0.02410542, 'V': -0.02410542}
    }
    if not seq:
        return [0] * len(physicochemical_properties)
    avg_properties = [np.mean([prop_values.get(aa, 0) for aa in seq]) for _, prop_values in
                      physicochemical_properties.items()]
    return avg_properties


def extract_traditional_features(sequences):
    amino_acids = list('ACDEFGHIKLMNPQRSTVWY')
    dipeptides = [''.join(dp) for dp in product(amino_acids, repeat=2)]
    traditional_features = []
    for seq in tqdm(sequences, desc="Extracting Traditional Features"):
        aa_freq = calculate_aa_frequency(seq, amino_acids)
        dipep_freq = calculate_dipeptide_frequency(seq, dipeptides)
        phys_props = calculate_physicochemical_properties(seq)
        features = aa_freq + dipep_freq + phys_props
        traditional_features.append(features)
    return np.array(traditional_features)


def extract_esm2_features(sequences, model, alphabet, batch_size, device):
    batch_converter = alphabet.get_batch_converter()
    features_list = []
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(sequences), batch_size), desc="Extracting Per-Residue ESM-2 Features"):
            batch_seqs = [s for s in sequences[i:i + batch_size] if len(s) > 0]
            if not batch_seqs: continue
            batch_data = [(f"seq_{j}", seq) for j, seq in enumerate(batch_seqs)]
            _, _, batch_tokens = batch_converter(batch_data)
            batch_tokens = batch_tokens.to(device)
            outputs = model(batch_tokens, repr_layers=[33], return_contacts=False)
            token_representations = outputs['representations'][33]
            for j, seq in enumerate(batch_seqs):
                seq_repr = token_representations[j, 1:len(seq) + 1, :].cpu()
                features_list.append(seq_repr)
    return features_list



class ProteinDataset(Dataset):
    def __init__(self, esm_features_list, traditional_features, labels):
        self.esm_features_list = esm_features_list
        self.traditional_features = torch.tensor(traditional_features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.esm_features_list[idx], self.traditional_features[idx], self.labels[idx]


def collate_fn(batch):
    esm_features, traditional_features, labels = zip(*batch)
    esm_padded = pad_sequence(esm_features, batch_first=True, padding_value=0)
    mask = (esm_padded.sum(dim=-1) == 0)
    traditional_stacked = torch.stack(traditional_features, 0)
    labels_stacked = torch.stack(labels, 0)
    return esm_padded, traditional_stacked, labels_stacked, mask




class ACP-EPC(nn.Module):

    def __init__(self, esm_dim, traditional_dim, embed_dim, num_heads, num_encoder_layers=1, dim_feedforward=1024,
                 num_classes=1, dropout=0.3):
        super(ACP-EPC, self).__init__()
        self.query_proj = nn.Linear(traditional_dim, embed_dim)
        self.key_proj = nn.Linear(esm_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=num_encoder_layers
        )
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, num_classes), nn.Sigmoid()
        )

    def forward(self, esm_features, traditional_features, key_padding_mask):
        query = self.query_proj(traditional_features).unsqueeze(1)
        key = self.key_proj(esm_features)
        encoder_output = self.transformer_encoder(key, src_key_padding_mask=key_padding_mask)
        attn_output, _ = self.multihead_attn(
            query=query, key=encoder_output, value=encoder_output, key_padding_mask=key_padding_mask
        )
        context_vector = attn_output.squeeze(1)
        return self.classifier(context_vector)


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss



def train_model_with_path(model, criterion, optimizer, train_loader, device, num_epochs, save_path):
    history = {'train_loss': [], 'epoch_time': []}
    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        running_loss = 0.0
        for esm_padded, trad_stacked, labels_stacked, mask in tqdm(train_loader,
                                                                   desc=f"Epoch {epoch + 1}/{num_epochs}"):
            esm_padded, trad_stacked, labels_stacked, mask = (d.to(device) for d in
                                                              [esm_padded, trad_stacked, labels_stacked, mask])
            optimizer.zero_grad()
            outputs = model(esm_padded, trad_stacked, mask)
            loss = criterion(outputs, labels_stacked.unsqueeze(1))
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * esm_padded.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        end_time = time.time()
        epoch_duration = end_time - start_time
        history['train_loss'].append(epoch_loss)
        history['epoch_time'].append(epoch_duration)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Time: {epoch_duration:.2f}s")
        torch.save(model.state_dict(), save_path)

    print(f"\nFinal model saved to {save_path}")
    avg_epoch_time = np.mean(history['epoch_time'])
    std_epoch_time = np.std(history['epoch_time'])
    print(f"Average Epoch Time: {avg_epoch_time:.2f}s (±{std_epoch_time:.2f}s)")
    return model, history, avg_epoch_time, std_epoch_time


def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for esm_padded, trad_stacked, labels_stacked, mask in tqdm(test_loader, desc="Evaluating"):
            esm_padded, trad_stacked, labels_stacked, mask = (d.to(device) for d in
                                                              [esm_padded, trad_stacked, labels_stacked, mask])
            outputs = model(esm_padded, trad_stacked, mask)
            preds = (outputs > 0.5).float()
            all_preds.extend(preds.cpu().numpy().flatten())
            all_labels.extend(labels_stacked.cpu().numpy().flatten())
            all_probs.extend(outputs.cpu().numpy().flatten())

    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, zero_division=0),
        'recall': recall_score(all_labels, all_preds, zero_division=0),
        'f1_score': f1_score(all_labels, all_preds, zero_division=0),
        'Mcc': matthews_corrcoef(all_labels, all_preds),
        'AUC': roc_auc_score(all_labels, all_probs)
    }
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
    metrics['Sn'] = metrics['recall']
    metrics['Sp'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    metrics.update({'Tn': tn, 'Tp': tp, 'Fn': fn, 'Fp': fp})

    print(
        f"Accuracy: {metrics['accuracy']:.4f}, Sn: {metrics['Sn']:.4f}, Sp: {metrics['Sp']:.4f}, MCC: {metrics['Mcc']:.4f}, AUC: {metrics['AUC']:.4f}")
    return metrics


def plot_metrics(history, save_path):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    fig.suptitle('Training Metrics Over Epochs', fontsize=16)
    epochs = range(1, len(history['train_loss']) + 1)
    ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    ax1.set_title('Training Loss')
    ax1.set_ylabel('Loss')
    ax1.legend();
    ax1.grid(True)
    ax2.plot(epochs, history['epoch_time'], 'r-', label='Epoch Time')
    ax2.set_title('Epoch Running Time')
    ax2.set_xlabel('Epochs');
    ax2.set_ylabel('Time (seconds)')
    ax2.legend();
    ax2.grid(True)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]);
    plt.savefig(save_path);
    plt.close()
    print(f"Training metrics plot saved to {save_path}")


def main():
    set_seed(42)
    train_fasta = 
    test1_fasta = 
    test2_fasta = 

    output_dir = 
    os.makedirs(output_dir, exist_ok=True)

    model_save_path = os.path.join(output_dir, 'best_model_focal.pth')
    scaler_save_path = os.path.join(output_dir, 'scaler.npz')
    metrics_plot_path = os.path.join(output_dir, 'training_metrics_focal.png')
    results_csv_path = os.path.join(output_dir, 'test_metrics_focal.csv')

    batch_size = 32
    num_epochs = 50
    learning_rate = 1e-4
    weight_decay = 1e-5
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_properties = 9
    print(f"Using device: {device}")

    train_data = parse_fasta(train_fasta)
    sequences_train = train_data['sequence'].tolist()
    y_train = train_data['label'].values

    start_trad_train = time.time()
    traditional_features_train = extract_traditional_features(sequences_train)
    print(f"Training traditional feature extraction took: {time.time() - start_trad_train:.2f}s")

    properties_train = traditional_features_train[:, -num_properties:]
    scaler_mean = np.mean(properties_train, axis=0)
    scaler_std = np.std(properties_train, axis=0)
    scaler_std[scaler_std == 0] = 1
    np.savez(scaler_save_path, mean=scaler_mean, std=scaler_std)
    print(f"Scaler parameters saved to {scaler_save_path}")
    traditional_features_train[:, -num_properties:] = (properties_train - scaler_mean) / scaler_std

    print("\nLoading ESM-2 model...")
    esm2_model, esm2_alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    esm2_model = esm2_model.to(device)

    start_esm_train = time.time()
    esm2_features_train_list = extract_esm2_features(sequences_train, esm2_model, esm2_alphabet, 4, device)
    print(f"Training ESM-2 feature extraction took: {time.time() - start_esm_train:.2f}s")

    train_dataset = ProteinDataset(esm2_features_train_list, traditional_features_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    model = ACP-EPC(
        esm_dim=1280, traditional_dim=429, embed_dim=256, num_heads=8
    ).to(device)
    print("\nModel Initialized:\n", model)

    criterion = FocalLoss(gamma=2.0, alpha=0.25)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    print("\nStarting training...")
    model, history, avg_time, std_time = train_model_with_path(model, criterion, optimizer, train_loader, device,
                                                               num_epochs, model_save_path)

    def process_and_evaluate(test_fasta, test_name, model, scaler_mean, scaler_std):
        print(f"\n--- Processing {test_name} ---")
        test_data = parse_fasta(test_fasta)
        sequences_test = test_data['sequence'].tolist()
        y_test = test_data['label'].values

        traditional_features_test = extract_traditional_features(sequences_test)

        properties_test = traditional_features_test[:, -num_properties:]
        traditional_features_test[:, -num_properties:] = (properties_test - scaler_mean) / scaler_std

        esm2_features_test_list = extract_esm2_features(sequences_test, esm2_model, esm2_alphabet, 4, device)

        test_dataset = ProteinDataset(esm2_features_test_list, traditional_features_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

        metrics = evaluate_model(model, test_loader, device)
        return metrics

    test1_metrics = process_and_evaluate(test1_fasta, "Test Set 1", model, scaler_mean, scaler_std)
    test2_metrics = process_and_evaluate(test2_fasta, "Test Set 2", model, scaler_mean, scaler_std)





if __name__ == "__main__":
    main()