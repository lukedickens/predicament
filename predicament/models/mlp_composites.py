import torch
import torch.nn as nn
import torch.nn.functional as F


# Sparse MLP is a single hidden layer MLP with a regularization term that
# induces sparse weights on the inputs. This means that only a certain number
# of input features get utilized.
class SparseMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, l1_penalty):
        super(SparseMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.l1_penalty = l1_penalty

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def l1_regularization(self):
        l1_reg = torch.norm(self.fc1.weight, 1)
        return l1_reg

# 
class PiecewiseSparseMLP(nn.Module):
    def __init__(
            self, input_size, hidden_size, output_size, l1_penalty, num_sets):
        super(PiecewiseSparseMLP, self).__init__()
        self.num_sets = num_sets
        self.sets_of_hidden_nodes = nn.ModuleList([
            SparseMLP(
                input_size, hidden_size, output_size, l1_penalty) for _ in range(num_sets)])
        self.prototype_vectors = nn.Parameter(torch.randn(num_sets, input_size))

    def compute_attention_weights(self, input_data):
        distances = torch.norm(
            input_data.unsqueeze(1) - self.prototype_vectors.unsqueeze(0), dim=2)
        attention_weights = F.softmax(-distances, dim=1)
        return attention_weights

    def forward(self, input_data):
        predictions = []
        for idx, set_model in enumerate(self.sets_of_hidden_nodes):
            output = set_model(input_data)
            predictions.append(output)
        predictions = torch.stack(predictions, dim=1)
        
        attention_weights = self.compute_attention_weights(input_data)
        
        aggregated_predictions = torch.sum(
            predictions * attention_weights.unsqueeze(-1), dim=1)
        
        return aggregated_predictions

# Example usage:
input_size = 10
hidden_size = 20
output_size = 1
l1_penalty = 0.001  # Strength of L1 regularization
num_sets = K  # Number of sets of hidden nodes

model = PiecewiseSparseMLP(input_size, hidden_size, output_size, l1_penalty, num_sets)
