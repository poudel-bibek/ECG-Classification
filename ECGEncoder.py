import time
import torch
import torch.nn as nn
import torch.nn.functional as F

class ECGEncoder():
    def __init__(self, ):
        self.best_model_path = './best_model.pth'
        self.model = self.load_model()
        self.model.eval()
    
    def load_model(self, ):
        
        # Model without the last layer
        model = nn.Sequential(
            nn.Conv1d(1, 128, kernel_size=16, stride=2, padding=0, bias=True),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(128, 128, kernel_size=16, stride=2, padding=0, dilation=2, bias=True),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(128, 128, kernel_size=16, stride=1, padding=0, dilation=2, bias=True),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(128, 64, kernel_size=8, stride=1, padding=0, dilation=1, bias=True),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(64, 64, kernel_size=8, stride=1, padding=0, dilation=1, bias=True),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(64, 32, kernel_size=4, stride=1, padding=0, dilation=1, bias=True),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(32, 32, kernel_size=4, stride=1, padding=0, dilation=1, bias=True),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(32, 16, kernel_size=2, stride=1, padding=0, dilation=1, bias=True),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Flatten(),
            nn.Linear(1488, 128),
            nn.ReLU(),
            nn.Linear(128, 8),
            nn.ReLU()
        )

        # Load the model except the last layer
        state_dict = torch.load(self.best_model_path, map_location=torch.device('cpu)) # Since we plan to run this in a CPU at inference.
        model.load_state_dict(state_dict, strict=False)
        return model
        
    def encode(self, input_ecg):
        """
        Make inference on the input ECG signal and return the encoded representation
        """
        with torch.no_grad():
            encoded = self.model(input_ecg)
        return encoded
    
# Example Use case
ecg_encoder = ECGEncoder()
for i in range(100):
    start = time.time()
    input_ecg = torch.randn(1, 1, 650)
    encoded = ecg_encoder.encode(input_ecg)[0]
    print(f"Encoded LDR: {encoded}\n{type(encoded), encoded.shape}", end="\t")
    print(f"Inference time: {time.time() - start}\n"
