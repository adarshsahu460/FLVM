import torch
from transformers import ViTModel
import os

class ViTForAlzheimers(torch.nn.Module):
    """Vision Transformer model for Alzheimer's classification."""
    def __init__(self, num_labels=4):
        super(ViTForAlzheimers, self).__init__()
        hf_token = os.getenv("HF_TOKEN")
        self.vit = ViTModel.from_pretrained('facebook/deit-tiny-patch16-224', use_auth_token=hf_token)
        # Freeze ViT backbone to reduce computation
        for param in self.vit.parameters():
            param.requires_grad = False
        # Unfreeze the last 6 encoder blocks for fine-tuning
        if hasattr(self.vit, 'encoder') and hasattr(self.vit.encoder, 'layer'):
            for param in self.vit.encoder.layer[-6:].parameters():
                param.requires_grad = True
        # Deepen the classifier head
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(self.vit.config.hidden_size, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(256, num_labels)
        )
    
    def forward(self, pixel_values):
        outputs = self.vit(pixel_values=pixel_values)
        pooled_output = outputs.last_hidden_state[:, 0]
        logits = self.classifier(pooled_output)
        return logits