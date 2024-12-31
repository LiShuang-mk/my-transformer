# %%
import torch
import time
import random

import MyTrans as trans
from utils import *
from dataloader import DataSet
from tqdm import tqdm as tqnb

# %%


class TransTrans(torch.nn.Module):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        max_src_len,
        max_tgt_len,
        num_heads,
        num_encoders,
        num_decoders,
        dim_model,
        dim_feedforward,
        device,
        padding_idx=0,
        dropout=0.1,
    ):
        super(TransTrans, self).__init__()

        self.src_emb = trans.embedding.TransformerEmbedding(
            vocab_size=src_vocab_size,
            max_len=max_src_len,
            d_model=dim_model,
            device=device,
            padding_idx=padding_idx,
        )
        self.tgt_emb = trans.embedding.TransformerEmbedding(
            vocab_size=tgt_vocab_size,
            max_len=max_tgt_len,
            d_model=dim_model,
            device=device,
            padding_idx=padding_idx,
        )

        self.transformer = torch.nn.Transformer(
            d_model=dim_model,
            nhead=num_heads,
            num_encoder_layers=num_encoders,
            num_decoder_layers=num_decoders,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            norm_first=True,
            dropout=dropout,
            device=device,
        )
        self.transformer.to(device)

        self.final_linear = torch.nn.Linear(dim_model, tgt_vocab_size)

        self.pad_idx = padding_idx

    def forward(self, src, tgt, src_len, tgt_len):
        src_key_padding_mask = self.get_key_padding_mask(src)
        tgt_key_padding_mask = self.get_key_padding_mask(tgt)

        src = self.src_emb(src, src_len)
        tgt = self.tgt_emb(tgt, tgt_len)

        # 将准备好的数据送给transformer
        out = self.transformer(
            src,
            tgt,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )

        out = self.final_linear(out)

        return out

    def get_key_padding_mask(self, tokens):
        key_padding_mask = torch.zeros(tokens.size(), device=tokens.device)
        key_padding_mask[tokens == self.pad_idx] = -torch.inf
        return key_padding_mask


# %%
params = {
    "model_dim": 64,
    "num_layers": 1,
    "num_heads": 1,
    "ff_dim": 256,
    "dropout": 0.1,
    "batch_size": 32,
    "learning_rate": 5e-4,
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

    _, y_pred_indeces = y_pred.max(dim=-1)

    correct_indices = torch.eq(y_pred_indeces, y_true).float()
    valid_indices = torch.ne(y_true, mask_index).float()

    n_correct = (correct_indices * valid_indices).sum().item()
    n_valid = valid_indices.sum().item()

    return n_correct / n_valid * 100, y_pred_indeces


def start_train(model, ds, dl, dl_val):
    optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer, mode="min", factor=0.5, patience=1
    )

    epoch_bar = tqnb(desc="training routine", total=params["num_epochs"], position=0)

    ds.set_split("train")
    train_bar = tqnb(
        desc="split=train",
        total=ds.get_num_batches(params["batch_size"]),
        position=1,
        leave=True,
    )
    ds.set_split("val")
    val_bar = tqnb(
        desc="split=val",
        total=ds.get_num_batches(params["batch_size"]),
        position=1,
        leave=True,
    )

    # logger = get_root_logger()
    loss_val_best = 1e10
    today_str = time.strftime("%Y-%m-%d", time.localtime())

    for epoch in range(params["num_epochs"]):

        ds.set_split("train")
        model.train()

        loss_epoch, acc_epoch, batch = 0, 0, 0
        for data_dict in dl:
            x_src = data_dict["x_source"].to(device)
            x_tgt = data_dict["x_target"].to(device)
            y_tgt = data_dict["y_target"].to(device)
            len_x_src = data_dict["x_srclen"].to(device)
            len_x_tgt = data_dict["x_tgtlen"].to(device)

            optimizer.zero_grad()

            y_pred = model(x_src, x_tgt, len_x_src, len_x_tgt)

            loss = compute_loss(y_pred, y_tgt, mask_idx=ds.get_src_vocab().msk_idx)
            acc, y_pred_indeces = compute_accuracy(
                y_pred, y_tgt, mask_index=ds.get_src_vocab().msk_idx
            )

            loss.backward()

            optimizer.step()

            loss_epoch += loss.detach().cpu().item()
            acc_epoch += acc

            batch += 1
            train_bar.set_postfix(
                loss=loss_epoch / batch, acc=acc_epoch / batch, epoch=epoch
            )
            train_bar.update()

        ds.set_split("val")
        model.eval()
        loss_epoch_val, acc_epoch_val, batch_val = 0, 0, 0

        for data_dict in dl_val:
            x_src = data_dict["x_source"].to(device)
            x_tgt = data_dict["x_target"].to(device)
            y_tgt = data_dict["y_target"].to(device)
            len_x_src = data_dict["x_srclen"].to(device)
            len_x_tgt = data_dict["x_tgtlen"].to(device)

            with torch.no_grad():
                y_pred = model(x_src, x_tgt, len_x_src, len_x_tgt)

            loss = compute_loss(y_pred, y_tgt, mask_idx=ds.get_src_vocab().msk_idx)
            acc, y_pred_indeces = compute_accuracy(y_pred, y_tgt, mask_index=ds.get_src_vocab().msk_idx)

            loss_epoch_val += loss.detach().cpu().item()
            acc_epoch_val += acc

            batch_val += 1
            val_bar.set_postfix(
                loss=loss_epoch_val / batch_val,
                acc=acc_epoch_val / batch_val,
                epoch=epoch,
            )
            val_bar.update()

        # logger.info(
        #     f"Epoch {epoch}: train loss: {loss_epoch/batch}, "
        #     + f"train acc: {acc_epoch/batch}, val loss: {loss_epoch_val/batch_val}, "
        #     + f"val acc: {acc_epoch_val/batch_val}"
        # )

        scheduler.step(loss_epoch_val / batch_val)

        if loss_epoch_val / batch_val < loss_val_best:
            loss_val_best = loss_epoch_val / batch_val
            epoch_bar.set_postfix(loss_val_best=loss_val_best)
            es = 0
            # torch.save(
            #     model.state_dict(),
            #     params["model_path"]
            #     + f"model_{today_str}_{params['seed']}_e{epoch}.pth",
            # )
        else:
            es += 1

        if es >= params["early_stop"]:
            print("Early stopping!")
            # logger.warning("early stopping!")
            epoch_bar.close()
            train_bar.close()
            val_bar.close()
            break

        train_bar.n = 0
        val_bar.n = 0
        epoch_bar.update()

    epoch_bar.close()
    train_bar.close()
    val_bar.close()


# %%
ds = DataSet.from_csv("./data/eng_fra.csv")
ds.set_split("train")
dl = torch.utils.data.DataLoader(
    ds, batch_size=params["batch_size"], shuffle=True, num_workers=4
)
ds.set_split("val")
dl_val = torch.utils.data.DataLoader(
    ds, batch_size=params["batch_size"], shuffle=False, num_workers=4
)

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

model = model.to(device)

start_train(model, ds, dl, dl_val)
