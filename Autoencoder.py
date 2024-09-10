class AnomalyDetectionAE(nn.Module):
    def __init__(self):
        super(AnomalyDetectionAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Example: Training the autoencoder for anomaly detection
def train_autoencoder(normal_data):
    model = AnomalyDetectionAE()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        output = model(normal_data)
        loss = criterion(output, normal_data)
        loss.backward()
        optimizer.step()

    return model

# Example: Normal signals and anomalous signals
normal_signal = torch.randn(64, 128)
anomalous_signal = normal_signal + 0.5 * torch.randn(64, 128)  # Add noise

# Train the model on normal signals
model = train_autoencoder(normal_signal)

# Anomaly detection
reconstructed_signal = model(anomalous_signal)
anomaly_score = torch.mean((anomalous_signal - reconstructed_signal) ** 2, dim=1)
print("Anomaly scores:", anomaly_score)
