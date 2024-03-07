import torch
from torch import nn

class Attention(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Attention, self).__init__()
        self.attention_weights = nn.Parameter(torch.randn(hidden_dim, input_dim))

    def forward(self, x):
        attention_scores = torch.matmul(x, self.attention_weights)
        attention_weights = torch.softmax(attention_scores, dim=-1)
        context_vector = torch.sum(x * attention_weights, dim=1)
        return context_vector


class MuonClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.attention = Attention(input_dim, 512)
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 32)
        self.fc4 = nn.Linear(32, 1)
        self.drop = nn.Dropout(0.1)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x = self.attention(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        return x
    
    # def __init__(self, input_dim):
    #     super().__init__()
    #     self.fc1 = nn.Linear(input_dim, 1024)
    #     self.fc2 = nn.Linear(1024, 256)
    #     self.fc3 = nn.Linear(256, 128)
    #     self.fc4 = nn.Linear(128, 16)
    #     self.fc5 = nn.Linear(16, 1)
    #     self.dropout = nn.Dropout(0.2)
    #     self.relu = nn.ReLU()
    #     # self.bn1 = nn.BatchNorm1d(1024)
    #     # self.bn2 = nn.BatchNorm1d(256)
    #     # self.bn3 = nn.BatchNorm1d(128)
    #     # self.bn4 = nn.BatchNorm1d(16)

    # def forward(self, x):
    #     x = self.fc1(x)
    #     # x = self.bn1(x)
    #     x = self.relu(x)
    #     x = self.dropout(x) # Comment here (?)
    #     x = self.fc2(x)
    #     # x = self.bn2(x)
    #     x = self.relu(x)
    #     x = self.dropout(x)
    #     x = self.fc3(x)
    #     # x = self.bn3(x)
    #     x = self.relu(x)
    #     x = self.dropout(x)
    #     x = self.fc4(x)
    #     # x = self.bn4(x)
    #     x = self.relu(x)
    #     x = self.dropout(x)
    #     x = self.fc5(x)
    #     return x

    def predict(self, x):
        pred = torch.sigmoid(self.forward(x))
        return pred

def train(
    train_dataloader, test_dataloader, model, loss_fn, optimizer, scheduler, device
):
    epoch_loss = 0

    model.train()
    size = len(train_dataloader.dataset)
    for batch, (X, y) in enumerate(train_dataloader):
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_dataloader)

    scheduler.step(avg_loss)
    # scheduler.step()

    print(f"Train | Loss = {avg_loss:.4f} |")

    model.eval()
    test_loss = 0
    with torch.no_grad():
        for X, y in test_dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()

    avg_test_loss = test_loss / len(test_dataloader)

    print(f"Test  | Loss = {avg_test_loss:.4f} |")
    print("--------------------------------------")

    return (avg_loss, avg_test_loss)
