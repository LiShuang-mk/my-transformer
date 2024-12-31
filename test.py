import torch
import time
import random
from nltk.tokenize import word_tokenize
from nltk.translate import bleu_score

import MyTrans as trans
from model import *
from utils import *
from dataloader import *
from tqdm import tqdm as tqnb

params = {
    "model_dim": 64,
    "num_layers": 2,
    "num_heads": 8,
    "ff_dim": 128,
    "dropout": 0.1,
    "batch_size": 32,
    "learning_rate": 0.005,
    "num_epochs": 100,
    "seed": 2003,
    "model_path": "./models/",
    "early_stop": 10,
    "model_weights_file": "model_2024-12-30-22-49-16_2003_e19_d10.pth",
}


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"use {device}")

seed = params["seed"]
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


ds = DataSet.from_csv("./data/large_eng_fra.csv")
ds.set_split("test")
dl = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=False, num_workers=4)

model = TransTrans(
    src_vocab_size=ds.get_src_vocab_size(),
    tgt_vocab_size=ds.get_tgt_vocab_size(),
    max_src_len=ds.get_max_src_len(),
    max_tgt_len=ds.get_max_tgt_len(),
    num_heads=params["num_heads"],
    num_encoders=params["num_layers"],
    num_decoders=params["num_layers"],
    dim_feedforward=params["ff_dim"],
    dim_model=params["model_dim"],
    device=device,
    padding_idx=ds.get_src_vocab().msk_idx,
    dropout=params["dropout"],
)
model.load_state_dict(
    torch.load(params["model_path"] + params["model_weights_file"], weights_only=True),
    assign=True,
)
model.to(device)
model.eval()

TT = torch.Tensor


def normalize_sizes(y_pred: TT, y_true: TT):
    if len(y_pred.size()) == 3:
        y_pred = y_pred.contiguous().view(-1, y_pred.size(2))
    if len(y_true.size()) == 2:
        y_true = y_true.contiguous().view(-1)
    return y_pred, y_true


def compute_loss(y_pred: TT, y_true: TT, mask_idx):
    y_pred, y_true = normalize_sizes(y_pred, y_true)
    return torch.nn.functional.cross_entropy(y_pred, y_true, ignore_index=mask_idx)


def compute_accuracy(y_pred: TT, y_true: TT, mask_index):
    bsz, seq = y_pred.size()[:2]
    y_pred, y_true = normalize_sizes(y_pred, y_true)

    y_pred_indices = torch.argmax(y_pred, dim=1)

    correct_indices = torch.eq(y_pred_indices, y_true).float()
    valid_indices = torch.ne(y_true, mask_index).float()

    n_correct = (correct_indices * valid_indices).sum().item()
    n_valid = valid_indices.sum().item()

    return n_correct / n_valid * 100, y_pred_indices.reshape((bsz, seq))


def sentence_from_indices(
    indices: list, vocab: Vocabulary, strict: bool = True, return_string: bool = True
) -> str | list:
    out = []
    for index in indices:
        if index == vocab.sos_idx and strict:
            continue
        elif index == vocab.eos_idx and strict:
            break
        else:
            out.append(vocab.lookup_idx(index))
    if return_string:
        return " ".join(out)
    else:
        return out


loss_test, acc_test, batch_test = 0, 0, 0
test_bar = tqnb(total=len(dl), desc="test")
src_vocab = ds.get_src_vocab()
tgt_vocab = ds.get_tgt_vocab()

bleus = []
chencherry = bleu_score.SmoothingFunction()

for data_dict in dl:
    x_src: TT = data_dict["x_source"].to(device)
    x_tgt: TT = data_dict["x_target"].to(device)
    y_tgt: TT = data_dict["y_target"].to(device)
    len_x_src: TT = data_dict["x_srclen"].to(device)
    len_x_tgt: TT = data_dict["x_tgtlen"].to(device)

    with torch.no_grad():
        y_pred = model(x_src, x_tgt, len_x_src, len_x_tgt)

    loss = compute_loss(y_pred, y_tgt, mask_idx=ds.get_src_vocab().msk_idx)
    acc, y_pred_indices = compute_accuracy(
        y_pred, y_tgt, mask_index=ds.get_src_vocab().msk_idx
    )

    loss_test += loss.detach().cpu().item()
    acc_test += acc

    y_tgt_list = y_tgt.tolist()
    y_pred_indices_list = y_pred_indices.tolist()
    for ref_sent, pred_sent in zip(y_tgt_list, y_pred_indices_list):
        ref_str = sentence_from_indices(ref_sent, vocab=tgt_vocab)
        pred_str = sentence_from_indices(pred_sent, vocab=tgt_vocab)
        score = bleu_score.sentence_bleu(
            references=ref_str,
            hypothesis=pred_str,
            smoothing_function=chencherry.method1,
        )
        bleus.append(score)

    batch_test += 1

    test_bar.set_postfix(
        loss=loss_test / batch_test,
        acc=acc_test / batch_test,
    )
    test_bar.update()

test_bar.close()

print(
    f"Test Loss: {loss_test / batch_test:.4f}, Test Accuracy: {acc_test / batch_test:.2f}%"
)

print(f"平均得分：{np.mean(bleus):.4f}\n" + f"中位数：{np.median(bleus):.4f}")
