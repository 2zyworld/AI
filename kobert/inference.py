import torch
from torch.utils.data import DataLoader
import gluonnlp as nlp
import numpy as np
from kobert.pytorch_kobert import get_pytorch_kobert_model
from kobert.utils import get_tokenizer
from AI.kobert.models import BERTClassifier
from AI.kobert.data import BERTDatasetInference


def inference(sentence, model_path="model1.pt"):
    bert_model, vocab = get_pytorch_kobert_model()
    model = BERTClassifier(bert_model, dr_rate=0.5)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    tokenizer = get_tokenizer()
    token = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)
    device = torch.device("cpu")

    dataset = BERTDatasetInference(sentence, token, 256, True, False)
    loader = DataLoader(dataset, batch_size=1, num_workers=4)

    output = []
    for token_ids, valid_length, segment_ids in loader:
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)
        output.append(model(token_ids, valid_length, segment_ids))
    softmax = torch.nn.Softmax(dim=1)
    output = softmax(output[0])[0].tolist()
    return output


def opacity(color_code, amp, bg=False):
    modified = []
    bg = 255 if bg else 0
    for i in range(3):
        c = (1 - amp) * bg + amp * color_code[i]
        modified.append(int(c))
    return modified


def emotion(output, encoder):
    temp1 = []
    for idx in range(60):
        if idx + 10 == 51:
            code = 31
        elif idx + 10 == 59:
            code = 34
        else:
            code = idx + 10

        row = encoder[encoder.code == code].iloc[0, 2:10]
        temp2 = [output[idx] * x for x in row]
        temp1.append(temp2)
    summation = np.sum(np.array(temp1), axis=0)
    return summation


def colorize(output, encoder):
    color_codes = [
        [255, 0, 0],
        [255, 100, 0],
        [255, 255, 0],
        [128, 255, 128],
        [0, 128, 50],
        [0, 225, 255],
        [0, 100, 225],
        [156, 0, 225]
    ]

    temp1 = []

    for idx in range(60):
        if idx + 10 == 51:
            code = 31
        elif idx + 10 == 59:
            code = 34
        else:
            code = idx + 10

        row = encoder[encoder.code == code].iloc[0, 2:10]
        for i, value in enumerate(row):
            modified = opacity(color_codes[i], value * output[idx])
            temp1.append(modified)

    temp2 = np.sum(np.array(temp1), axis=0)
    temp2 = [int(x) for x in temp2]
    return temp2
