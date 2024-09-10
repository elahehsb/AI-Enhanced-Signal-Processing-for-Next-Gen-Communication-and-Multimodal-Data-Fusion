import torch
import torch.nn as nn

class MultimodalFusionNN(nn.Module):
    def __init__(self):
        super(MultimodalFusionNN, self).__init__()
        self.audio_fc = nn.Linear(128, 64)
        self.video_fc = nn.Linear(256, 64)
        self.fc = nn.Linear(128, 1)  # Final output for fusion

    def forward(self, audio_signal, video_signal):
        audio_out = nn.ReLU()(self.audio_fc(audio_signal))
        video_out = nn.ReLU()(self.video_fc(video_signal))
        fused_out = torch.cat((audio_out, video_out), dim=1)  # Concatenate features
        return self.fc(fused_out)

# Example: Simulated audio and video data
audio_data = torch.randn(32, 128)  # 32 samples of audio data
video_data = torch.randn(32, 256)  # 32 samples of video data

# Initialize the fusion model
fusion_model = MultimodalFusionNN()

# Perform multimodal fusion
output = fusion_model(audio_data, video_data)
print("Fused output:", output)
