import torch
import torch.nn as nn
import torch.optim as optim

class SignalDenoisingCNN(nn.Module):
    def __init__(self):
        super(SignalDenoisingCNN, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU()
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Training the denoising model
def train_denoising_model(signal_data, noisy_data):
    model = SignalDenoisingCNN()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        output = model(noisy_data)
        loss = criterion(output, signal_data)
        loss.backward()
        optimizer.step()

    return model

# Example: Simulated clean and noisy signal data
clean_signal = torch.randn(64, 1, 128)  # Clean signal
noisy_signal = clean_signal + 0.1 * torch.randn(64, 1, 128)  # Noisy signal

# Train the model
model = train_denoising_model(clean_signal, noisy_signal)
