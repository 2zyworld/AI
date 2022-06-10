import torch

from transformers import (
    ElectraForSequenceClassification,
    ElectraConfig,
)
from transformers.models.electra.modeling_electra import ElectraClassificationHead


class KoELECTRAModel(torch.nn.Module):
    def __init__(self, num_labels, hidden_size=256):
        super().__init__()
        self.model = ElectraForSequenceClassification.from_pretrained(
            "monologg/koelectra-small-v3-discriminator"
        )
        config = ElectraConfig(hidden_size=hidden_size, hidden_dropout_prob=0.1, num_labels=num_labels)
        self.model.classifier = ElectraClassificationHead(config)
