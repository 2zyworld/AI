import torch
from torch.utils.data import DataLoader
import gluonnlp as nlp
from kobert.pytorch_kobert import get_pytorch_kobert_model
from kobert.utils import get_tokenizer
from AI.kobert.models import BERTClassifier
from AI.kobert.data import BERTDatasetInference


def inference(sentence):
    bert_model, vocab = get_pytorch_kobert_model()
    model = BERTClassifier(bert_model, dr_rate=0.5)
    model.load_state_dict(torch.load("model.pt"))
    model.eval()

    tokenizer = get_tokenizer()
    token = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)
    device = torch.device("cpu")

    dataset = BERTDatasetInference(sentence, token, 256, True, False)
    loader = DataLoader(dataset, batch_size=1, num_workers=4)

    out = []
    for token_ids, valid_length, segment_ids in loader:
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)
        out = model(token_ids, valid_length, segment_ids)

    return out
