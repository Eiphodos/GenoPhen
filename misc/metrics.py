import torch

def preprocess_mlm_acc(predictions, labels):
    indices = [[i for i, x in enumerate(labels[row]) if x != -100] for row in range(len(labels))]

    labels = [labels[row][indices[row]] for row in range(len(labels))]
    labels = [item for sublist in labels for item in sublist]

    predictions = [predictions[row][indices[row]] for row in range(len(predictions))]
    predictions = [item for sublist in predictions for item in sublist]

    return torch.stack(predictions), torch.stack(labels)


def oskar_pt_accuracy_calc(predictions, target):
    accuracy_sum = 0
    for sentence_idx in range(len(predictions)):
        idx_list = []
        correct_count = 0
        for i in range(len(target[sentence_idx])):
            if target[sentence_idx][i] != -100:
                idx_list.append(i)
        for idx in idx_list:
            if target[sentence_idx][idx] == predictions[sentence_idx][idx]:
                correct_count += 1
        if len(idx_list) != 0:
            accuracy_sum += correct_count / len(idx_list)

    accuracy = accuracy_sum / len(predictions)
    return accuracy