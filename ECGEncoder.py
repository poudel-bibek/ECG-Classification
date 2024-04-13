import time
import torch
import torch.nn as nn
import torch.nn.functional as F


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)
    
class ConvNormPool(nn.Module):
    """Conv Skip-connection module"""
    def __init__(
        self,
        input_size,
        hidden_size,
        kernel_size,
        norm_type='bachnorm'
    ):
        super().__init__()
        
        self.kernel_size = kernel_size
        self.conv_1 = nn.Conv1d(
            in_channels=input_size,
            out_channels=hidden_size,
            kernel_size=kernel_size
        )
        self.conv_2 = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=kernel_size
        )
        self.conv_3 = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=kernel_size
        )
        self.swish_1 = Swish()
        self.swish_2 = Swish()
        self.swish_3 = Swish()
        if norm_type == 'group':
            self.normalization_1 = nn.GroupNorm(
                num_groups=8,
                num_channels=hidden_size
            )
            self.normalization_2 = nn.GroupNorm(
                num_groups=8,
                num_channels=hidden_size
            )
            self.normalization_3 = nn.GroupNorm(
                num_groups=8,
                num_channels=hidden_size
            )
        else:
            self.normalization_1 = nn.BatchNorm1d(num_features=hidden_size)
            self.normalization_2 = nn.BatchNorm1d(num_features=hidden_size)
            self.normalization_3 = nn.BatchNorm1d(num_features=hidden_size)
            
        self.pool = nn.MaxPool1d(kernel_size=2)
        
    def forward(self, input):
        conv1 = self.conv_1(input)
        x = self.normalization_1(conv1)
        x = self.swish_1(x)
        x = F.pad(x, pad=(self.kernel_size - 1, 0))
        
        x = self.conv_2(x)
        x = self.normalization_2(x)
        x = self.swish_2(x)
        x = F.pad(x, pad=(self.kernel_size - 1, 0))
        
        conv3 = self.conv_3(x)
        x = self.normalization_3(conv1+conv3)
        x = self.swish_3(x)
        x = F.pad(x, pad=(self.kernel_size - 1, 0))   
        
        x = self.pool(x)
        return x
    
class ECGCNN(nn.Module):
    def __init__(
        self,
        input_size=1,
        hid_size=256,
        kernel_size=5,
    ):
        super().__init__()
        
        self.conv1 = ConvNormPool(
            input_size=input_size,
            hidden_size=hid_size,
            kernel_size=kernel_size,
        )
        self.conv2 = ConvNormPool(
            input_size=hid_size,
            hidden_size=hid_size//2,
            kernel_size=kernel_size,
        )
        self.conv3 = ConvNormPool(
            input_size=hid_size//2,
            hidden_size=hid_size//4,
            kernel_size=kernel_size,
        )
        self.avgpool = nn.AdaptiveAvgPool1d((1))
        self.fc = nn.Linear(in_features=hid_size//4, out_features=8)
        
        self.swish = Swish()
        
    def forward(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.avgpool(x)        
        x = x.view(-1, x.size(1) * x.size(2))
        x = self.swish(self.fc(x))
        
        return x
    
class ECGEncoder():
    def __init__(self, ):
        self.best_model_path = './best_model.pth'
        self.model = self.load_model()
        self.model.eval()
    
    def load_model(self, ):
        
        # Model without the last layer
        # model = nn.Sequential(
        #     nn.Conv1d(1, 128, kernel_size=16, stride=2, padding=0, bias=True),
        #     nn.BatchNorm1d(128),
        #     nn.ReLU(),
        #     nn.Dropout(0.2),
        #     nn.Conv1d(128, 128, kernel_size=16, stride=2, padding=0, dilation=2, bias=True),
        #     nn.BatchNorm1d(128),
        #     nn.ReLU(),
        #     nn.Dropout(0.2),
        #     nn.Conv1d(128, 128, kernel_size=16, stride=1, padding=0, dilation=2, bias=True),
        #     nn.BatchNorm1d(128),
        #     nn.ReLU(),
        #     nn.Dropout(0.2),
        #     nn.Conv1d(128, 64, kernel_size=8, stride=1, padding=0, dilation=1, bias=True),
        #     nn.BatchNorm1d(64),
        #     nn.ReLU(),
        #     nn.Dropout(0.2),
        #     nn.Conv1d(64, 64, kernel_size=8, stride=1, padding=0, dilation=1, bias=True),
        #     nn.BatchNorm1d(64),
        #     nn.ReLU(),
        #     nn.Dropout(0.2),
        #     nn.Conv1d(64, 32, kernel_size=4, stride=1, padding=0, dilation=1, bias=True),
        #     nn.BatchNorm1d(32),
        #     nn.ReLU(),
        #     nn.Dropout(0.2),
        #     nn.Conv1d(32, 32, kernel_size=4, stride=1, padding=0, dilation=1, bias=True),
        #     nn.BatchNorm1d(32),
        #     nn.ReLU(),
        #     nn.Dropout(0.2),
        #     nn.Conv1d(32, 16, kernel_size=2, stride=1, padding=0, dilation=1, bias=True),
        #     nn.BatchNorm1d(16),
        #     nn.ReLU(),
        #     nn.Dropout(0.2),
        #     nn.Flatten(),
        #     nn.Linear(1488, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 8),
        #     nn.ReLU()
        # )

        model = ECGCNN(input_size=1, hid_size=256, kernel_size=5)
        # Load the model except the last layer
        state_dict = torch.load(self.best_model_path, map_location=torch.device('cpu')) # Since we plan to run this in a CPU at inference.
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
# ecg_encoder = ECGEncoder()
# for i in range(100):
#     start = time.time()
#     input_ecg = torch.randn(1, 1, 650)
#     encoded = ecg_encoder.encode(input_ecg)[0]
#     print(f"Encoded LDR: {encoded}\n{type(encoded), encoded.shape}", end="\t")
#     print(f"Inference time: {time.time() - start}\n")
