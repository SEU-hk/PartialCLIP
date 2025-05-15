import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def visualize(partialY, labels):
    # 1. 统计整行为0的比例
    zero_rows = torch.all(partialY == 0, dim=1)
    zero_row_ratio = zero_rows.float().mean()
    N, K = partialY.shape[0], partialY.shape[1]

    # 2. 统计每行1的个数
    ones_count = torch.sum(partialY, dim=1)

    print('Average candidate num: ', ones_count.mean())
    
    # 打印zero_row_ratio
    print(f'empty candidate set ratio: {zero_row_ratio}')

    labels_in_mask = (partialY[torch.arange(partialY.shape[0]), labels.squeeze()] == 1).float()

    proportion = labels_in_mask.mean()
    print(f'Covering Rate: {proportion}')


def remove_zero_rows(partialY, data, labels):
    non_zero_rows = (partialY != 0).any(dim=1)
    new_partialY = []
    new_data = []
    new_labels = []

    for i, is_non_zero in enumerate(non_zero_rows):
        if is_non_zero:
            new_partialY.append(partialY[i].tolist())
            new_data.append(data[i])
            new_labels.append(labels[i].item())

    new_partialY = torch.tensor(new_partialY)
    new_labels = torch.tensor(new_labels)

    return new_partialY, new_data, new_labels

def calculate_top_k_accuracy(outputs, labels, k_percentiles):
    """
    Calculate top-k% accuracy.
    
    Args:
        outputs (torch.Tensor): Model outputs (logits). Shape: (batch_size, num_classes).
        labels (torch.Tensor): True labels. Shape: (batch_size,).
        k_percentiles (list of float): List of percentiles (e.g., [0.1, 0.2, 0.3, 0.4, 0.5]).
    
    Returns:
        dict: Dictionary with top-k% accuracies.
    """
    batch_size, num_classes = outputs.shape
    accuracies = {}
    
    # Convert logits to probabilities
    probs = F.softmax(outputs, dim=1)
    
    # Get the top-k indices for each sample
    for k_percentile in k_percentiles:
        k = int(k_percentile * num_classes)
        
        # Get the top-k indices
        _, top_k_indices = torch.topk(probs, k, dim=1)
        
        # Check if the true label is in the top-k indices
        correct = (top_k_indices == labels.unsqueeze(1).expand_as(top_k_indices)).any(dim=1)
        
        # Calculate accuracy
        accuracy = correct.float().mean().item()
        accuracies[f'top_{k_percentile * 100:.0f}%'] = accuracy
    
    return accuracies


