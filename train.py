# %%
import torch
import time
import random
import copy
import argparse

import MyTrans as trans
from utils import *
from dataloader import DataSet
from tqdm import tqdm as tqnb

# %%
params = {
    "model_dim": 64,
    "num_layers": 1,
    "num_heads": 4,
    "ff_dim": 256,
    "dropout": 0.1,
    "batch_size": 16,
    "learning_rate": 5e-3,
    "num_epochs": 100,
    "seed": 2003,
    "model_path": "./models/",
    "early_stop": 10,
}

parser = argparse.ArgumentParser()
parser.add_argument("--model_dim", type=int, default=params["model_dim"])
parser.add_argument("--num_layers", type=int, default=params["num_layers"])
parser.add_argument("--num_heads", type=int, default=params["num_heads"])
parser.add_argument("--ff_dim", type=int, default=params["ff_dim"])
parser.add_argument("--dropout", type=float, default=params["dropout"])
parser.add_argument("--batch_size", type=int, default=params["batch_size"])

args = parser.parse_args()
args_dict = args.__dict__
params.update(args_dict)


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

        self.final_linear = torch.nn.Linear(dim_model, tgt_vocab_size)

        self.pad_idx = padding_idx

        self.tgt_mask = torch.tril(torch.ones(max_tgt_len, max_tgt_len, device=device))
        self.tgt_mask[self.tgt_mask == 0] = float("-inf")
        self.tgt_mask[self.tgt_mask == 1] = 0

    def forward(self, src, tgt, src_len, tgt_len):
        src_key_padding_mask = self.get_key_padding_mask(src)
        tgt_key_padding_mask = self.get_key_padding_mask(tgt)

        src = self.src_emb(src, src_len)
        tgt = self.tgt_emb(tgt, tgt_len)

        out = self.transformer(
            src,
            tgt,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            tgt_mask=self.tgt_mask,
        )

        out = self.final_linear(out)

        return out

    def get_key_padding_mask(self, tokens):
        key_padding_mask = torch.zeros(tokens.size(), device=tokens.device)
        key_padding_mask[tokens == self.pad_idx] = -torch.inf
        return key_padding_mask


# %%

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

    y_pred_indices = torch.argmax(y_pred, dim=1)

    correct_indices = torch.eq(y_pred_indices, y_true).float()
    valid_indices = torch.ne(y_true, mask_index).float()

    n_correct = (correct_indices * valid_indices).sum().item()
    n_valid = valid_indices.sum().item()

    return n_correct / n_valid * 100


def start_train(model, ds, dl, dl_val, dl_test):
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

    logger = get_root_logger()
    loss_val_best = 1e10
    acc_val_best = 0
    now_str = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    best_model_state = None
    best_epoch = -1
    save_cnt = 0

    logger.info(params)

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

            #
            # x_tgt = torch.zeros_like(x_src).to(device)
            # x_tgt[:, :] = vectorizor.tgt_vocab.msk_idx
            # x_tgt[:, 0] = vectorizor.tgt_vocab.sos_idx
            # x_src_len = torch.tensor(data_dict["x_srclen"], dtype=torch.int64).to(device)
            # x_tgt_len = torch.zeros_like(data_dict["x_tgtlen"], dtype=torch.int64).to(device)
            # x_tgt_len[:] = 1
            #

            optimizer.zero_grad()

            y_pred = model(x_src, x_tgt, len_x_src, len_x_tgt)
            #
            # for i in range(vectorizor.max_tgt_len):
            #     y_pred = model(x_src, x_tgt, x_src_len, x_tgt_len)
            #     y_pred_indices = torch.argmax(y_pred, dim=-1)
            #     x_tgt_len[:] += 1
            #     x_tgt = y_pred_indices
            #

            loss = compute_loss(y_pred, y_tgt, mask_idx=ds.get_src_vocab().msk_idx)
            acc = compute_accuracy(y_pred, y_tgt, mask_index=ds.get_src_vocab().msk_idx)

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

            #
            # x_tgt = torch.zeros_like(x_src).to(device)
            # x_tgt[:, :] = vectorizor.tgt_vocab.msk_idx
            # x_tgt[:, 0] = vectorizor.tgt_vocab.sos_idx
            # x_src_len = torch.tensor(data_dict["x_srclen"], dtype=torch.int64).to(device)
            # x_tgt_len = torch.zeros_like(data_dict["x_tgtlen"], dtype=torch.int64).to(device)
            # x_tgt_len[:] = 1
            #

            with torch.no_grad():
                y_pred = model(x_src, x_tgt, len_x_src, len_x_tgt)
                #
                # for i in range(vectorizor.max_tgt_len):
                #     y_pred = model(x_src, x_tgt, x_src_len, x_tgt_len)
                #     y_pred_indices = torch.argmax(y_pred, dim=-1)
                #     x_tgt_len[:] += 1
                #     x_tgt = y_pred_indices
                #

            loss = compute_loss(y_pred, y_tgt, mask_idx=ds.get_src_vocab().msk_idx)
            acc = compute_accuracy(y_pred, y_tgt, mask_index=ds.get_src_vocab().msk_idx)

            loss_epoch_val += loss.detach().cpu().item()
            acc_epoch_val += acc

            batch_val += 1
            val_bar.set_postfix(
                loss=loss_epoch_val / batch_val,
                acc=acc_epoch_val / batch_val,
                epoch=epoch,
            )
            val_bar.update()

        logger.info(
            f"Epoch {epoch}: train loss: {loss_epoch/batch}, "
            + f"train acc: {acc_epoch/batch}, val loss: {loss_epoch_val/batch_val}, "
            + f"val acc: {acc_epoch_val/batch_val}"
        )

        scheduler.step(loss_epoch_val / batch_val)

        if acc_val_best < acc_epoch_val / batch_val:
            acc_val_best = acc_epoch_val / batch_val

        if loss_epoch_val / batch_val < loss_val_best:
            loss_val_best = loss_epoch_val / batch_val
            best_epoch = epoch
            epoch_bar.set_postfix(loss_val_best=loss_val_best, best_epoch=best_epoch)
            es = 0
            best_model_state = copy.deepcopy(model.state_dict())
            if epoch > 10:
                torch.save(
                    best_model_state,
                    params["model_path"]
                    + f"model_{now_str}_{params['seed']}_e{best_epoch}_d{int(params["dropout"]*100)}.pth",
                )
                save_cnt += 1
        else:
            es += 1

        if es >= params["early_stop"]:
            print("Early stopping!")
            logger.warning("early stopping!")
            epoch_bar.close()
            train_bar.close()
            val_bar.close()
            if save_cnt == 0:
                torch.save(
                    best_model_state,
                    params["model_path"]
                    + f"model_{now_str}_{params['seed']}_e{best_epoch}_d{int(params["dropout"]*100)}.pth",
                )
            break

        train_bar.n = 0
        val_bar.n = 0
        epoch_bar.update()

    epoch_bar.close()
    train_bar.close()
    val_bar.close()

    # test model

    ds.set_split("test")
    logger.info("... Testing model: use best loss model ...")
    logger.info(
        f"... best epoch: {best_epoch}, best loss: {loss_val_best}, best acc: {acc_val_best} ..."
    )
    model.load_state_dict(best_model_state)
    model.eval()
    loss_test, acc_test, batch_test = 0, 0, 0
    test_bar = tqnb(total=len(dl), desc="test")

    for data_dict in dl_test:
        x_src = data_dict["x_source"].to(device)
        x_tgt = data_dict["x_target"].to(device)
        y_tgt = data_dict["y_target"].to(device)
        len_x_src = data_dict["x_srclen"].to(device)
        len_x_tgt = data_dict["x_tgtlen"].to(device)

        with torch.no_grad():
            y_pred = model(x_src, x_tgt, len_x_src, len_x_tgt)

        loss = compute_loss(y_pred, y_tgt, mask_idx=ds.get_src_vocab().msk_idx)
        acc = compute_accuracy(y_pred, y_tgt, mask_index=ds.get_src_vocab().msk_idx)

        loss_test += loss.detach().cpu().item()
        acc_test += acc

        batch_test += 1

        test_bar.set_postfix(
            loss=loss_test / batch_test,
            acc=acc_test / batch_test,
        )
        test_bar.update()

    test_bar.close()

    logger.info(f"Test loss: {loss_test/batch_test}, Test acc: {acc_test/batch_test}")
    del logger


# %%
ds = DataSet.from_csv("./data/large_eng_fra.csv")
ds.set_split("train")
dl = torch.utils.data.DataLoader(
    ds, batch_size=params["batch_size"], shuffle=True, num_workers=3
)
ds_val = DataSet.from_csv("./data/large_eng_fra.csv")
ds_val.set_split("val")
dl_val = torch.utils.data.DataLoader(
    ds_val, batch_size=params["batch_size"], shuffle=False, num_workers=1
)
ds_test = DataSet.from_csv("./data/large_eng_fra.csv")
ds_test.set_split("test")
dl_test = torch.utils.data.DataLoader(
    ds_test, batch_size=params["batch_size"], shuffle=False, num_workers=1
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

start_train(model, ds, dl, dl_val, dl_test)

# %% [markdown]
#
