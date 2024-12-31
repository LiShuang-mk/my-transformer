# %%
import torch
import time
import random
from nltk.tokenize import word_tokenize

import MyTrans as trans
from model import *
from utils import *
from dataloader import DataSet
from tqdm import tqdm as tqnb

# %%
params = {
    "model_dim": 128,
    "num_layers": 6,
    "num_heads": 8,
    "ff_dim": 512,
    "dropout": 0.1,
    "batch_size": 64,
    "learning_rate": 5e-3,
    "num_epochs": 100,
    "seed": 1229,
    "model_path": "./models/",
    "early_stop": 10,
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

# %%
ds = DataSet.from_csv("./data/eng_fra.csv")
ds.set_split("test")
dl = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=8)

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
    torch.load("./models/model_2024-12-30-16-22-52_1230_e12_d10.pth", weights_only=True), assign=True
)
model.to(device)
model.eval()

# %%
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
    y_pred, y_true = normalize_sizes(y_pred, y_true)

    _, y_pred_indices = y_pred.max(dim=1)

    correct_indices = torch.eq(y_pred_indices, y_true).float()
    valid_indices = torch.ne(y_true, mask_index).float()

    n_correct = (correct_indices * valid_indices).sum().item()
    n_valid = valid_indices.sum().item()

    return n_correct / n_valid * 100, y_pred_indices


loss_test, acc_test, batch_test = 0, 0, 0
test_bar = tqnb(total=len(dl), desc="test")

for data_dict in dl:
    x_src = data_dict["x_source"].to(device)
    x_tgt = data_dict["x_target"].to(device)
    y_tgt = data_dict["y_target"].to(device)
    len_x_src = data_dict["x_srclen"].to(device)
    len_x_tgt = data_dict["x_tgtlen"].to(device)

    with torch.no_grad():
        y_pred = model(x_src, x_tgt, len_x_src, len_x_tgt)

    loss = compute_loss(y_pred, y_tgt, mask_idx=ds.get_src_vocab().msk_idx)
    acc, y_pred_indices = compute_accuracy(y_pred, y_tgt, mask_index=ds.get_src_vocab().msk_idx)

    loss_test += loss.detach().cpu().item()
    acc_test += acc

    batch_test += 1

    test_bar.set_postfix(
        loss=loss_test / batch_test,
        acc=acc_test / batch_test,
    )
    test_bar.update()

test_bar.close()

print(f"Test Loss: {loss_test / batch_test:.4f}, Test Accuracy: {acc_test / batch_test:.2f}%")


# %%
def infer_target(model: TransTrans, src: str):
    src_seq = word_tokenize(src)
    src_seq = " ".join(src_seq)
    vectorizor = ds.get_vectorizer()
    data_dict = vectorizor.vectorize(src_seq, "")

    x_src: TT = torch.tensor(data_dict["source_vector"]).unsqueeze(0).to(device)
    x_tgt: TT = torch.zeros_like(x_src).to(device)
    x_tgt[0, :] = vectorizor.tgt_vocab.msk_idx
    x_tgt[0, 0] = vectorizor.tgt_vocab.sos_idx
    x_src_len: TT = torch.tensor(data_dict["source_length"]).unsqueeze(0).to(device)
    x_tgt_len: TT = torch.tensor(1).unsqueeze(0).to(device)

    while x_tgt_len[0] < vectorizor.max_tgt_len:
        y = model(x_src, x_tgt, x_src_len, x_tgt_len)
        y = torch.argmax(y, dim=-1)
        if y[0, -1] == vectorizor.tgt_vocab.eos_idx:
            x_tgt = y
            break
        x_tgt_len[0] += 1
        x_tgt = y

    y_pred = [
        vectorizor.tgt_vocab.lookup_idx(token.item())
        for token in x_tgt[0, 1 : x_tgt_len[0]]
    ]
    return " ".join(y_pred)


res = infer_target(model, "I love you")
print(res)

# %%
