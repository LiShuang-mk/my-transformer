import pandas as pd
import numpy as np
import torch


class Vocabulary:
    def __init__(
        self, unk_token="<unk>", msk_token="<msk>", sos_token="<sos>", eos_token="<eos>"
    ):
        self._mask_token = msk_token
        self._unk_token = unk_token
        self._sos_token = sos_token
        self._eos_token = eos_token

        self._token_to_idx = {}
        self._idx_to_token = {}
        self.msk_idx = self.add_token(self._mask_token)
        self.unk_idx = self.add_token(self._unk_token)
        self.sos_idx = self.add_token(self._sos_token)
        self.eos_idx = self.add_token(self._eos_token)

    def add_token(self, token):
        if token not in self._token_to_idx:
            idx = len(self._token_to_idx)
            self._token_to_idx[token] = idx
            self._idx_to_token[idx] = token
            return idx
        else:
            return self._token_to_idx[token]

    def lookup_token(self, token):
        if token in self._token_to_idx:
            return self._token_to_idx[token]
        else:
            return self._token_to_idx[self._unk_token]

    def lookup_idx(self, idx):
        if idx in self._idx_to_token:
            return self._idx_to_token[idx]
        else:
            raise KeyError(f"Index {idx} is not in the Vocabulary")

    def __len__(self):
        return len(self._token_to_idx)

    def mask_token(self):
        return self._mask_token

    def unk_token(self):
        return self._unk_token

    def sos_token(self):
        return self._sos_token

    def eos_token(self):
        return self._eos_token


class Vectorizer:
    def __init__(
        self,
        src_vocab: Vocabulary,
        tgt_vocab: Vocabulary,
        max_src_len: int,
        max_tgt_len: int,
    ):
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_src_len = max_src_len + 2
        self.max_tgt_len = max_tgt_len + 1

    @classmethod
    def from_dataframe(cls, df):

        source_vocab = Vocabulary()
        target_vocab = Vocabulary()

        max_source_length = 0
        max_target_length = 0

        for _, row in df.iterrows():
            source_tokens = row["source_language"].split(" ")
            if len(source_tokens) > max_source_length:
                max_source_length = len(source_tokens)
            for token in source_tokens:
                source_vocab.add_token(token)

            target_tokens = row["target_language"].split(" ")
            if len(target_tokens) > max_target_length:
                max_target_length = len(target_tokens)
            for token in target_tokens:
                target_vocab.add_token(token)

        return cls(source_vocab, target_vocab, max_source_length, max_target_length)

    def _vectorize(self, indices, vector_length=-1, mask_index=0):
        if vector_length < 0:
            vector_length = len(indices)

        vector = np.zeros(vector_length, dtype=np.int64)
        vector[: len(indices)] = indices
        vector[len(indices) :] = mask_index

        return vector

    def _get_source_indices(self, text):
        indices = [self.src_vocab.sos_idx]
        indices.extend(self.src_vocab.lookup_token(token) for token in text.split(" "))
        indices.append(self.src_vocab.eos_idx)
        return indices

    def _get_target_indices(self, text):
        indices = [self.tgt_vocab.lookup_token(token) for token in text.split(" ")]
        x_indices = [self.tgt_vocab.sos_idx] + indices
        y_indices = indices + [self.tgt_vocab.eos_idx]
        return x_indices, y_indices

    def vectorize(self, source_text, target_text, use_dataset_max_lengths=True):
        source_vector_length = -1
        target_vector_length = -1

        if use_dataset_max_lengths:
            source_vector_length = self.max_src_len
            target_vector_length = self.max_tgt_len

        source_indices = self._get_source_indices(source_text)
        source_vector = self._vectorize(
            source_indices,
            vector_length=source_vector_length,
            mask_index=self.src_vocab.msk_idx,
        )

        target_x_indices, target_y_indices = self._get_target_indices(target_text)
        target_x_vector = self._vectorize(
            target_x_indices,
            vector_length=target_vector_length,
            mask_index=self.tgt_vocab.msk_idx,
        )
        target_y_vector = self._vectorize(
            target_y_indices,
            vector_length=target_vector_length,
            mask_index=self.tgt_vocab.msk_idx,
        )
        return {
            "source_vector": source_vector,
            "target_x_vector": target_x_vector,
            "target_y_vector": target_y_vector,
            "source_length": len(source_indices),
            "target_x_length": len(target_x_indices),
        }


class DataSet(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame, vectorizer: Vectorizer):
        self.text_df = df
        self._vectorizer = vectorizer

        self.train_df = self.text_df[self.text_df.split == "train"]
        self.train_size = len(self.train_df)

        self.val_df = self.text_df[self.text_df.split == "val"]
        self.val_size = len(self.val_df)

        self.test_df = self.text_df[self.text_df.split == "test"]
        self.test_size = len(self.test_df)

        self._lookup_dict = {
            "train": (self.train_df, self.train_size),
            "val": (self.val_df, self.val_size),
            "test": (self.test_df, self.test_size),
        }

        self.set_split("train")

    @classmethod
    def from_csv(cls, dataset_csv):
        text_df = pd.read_csv(dataset_csv)
        train_subset = text_df[text_df.split == "train"]
        return cls(text_df, Vectorizer.from_dataframe(train_subset))

    def set_split(self, split="train"):
        self._target_split = split
        self._target_df, self._target_size = self._lookup_dict[split]

    def __len__(self):
        return self._target_size

    def __getitem__(self, index):
        row = self._target_df.iloc[index]

        vector_dict = self._vectorizer.vectorize(
            row.source_language, row.target_language
        )

        return {
            "x_source": vector_dict["source_vector"],
            "x_target": vector_dict["target_x_vector"],
            "y_target": vector_dict["target_y_vector"],
            "x_srclen": vector_dict["source_length"],
            "x_tgtlen": vector_dict["target_x_length"],
        }
        
    def get_src_vocab_size(self):
        return len(self._vectorizer.src_vocab)
    
    def get_tgt_vocab_size(self):
        return len(self._vectorizer.tgt_vocab)
    
    def get_max_src_len(self):
        return self._vectorizer.max_src_len
    
    def get_max_tgt_len(self):
        return self._vectorizer.max_tgt_len
    
    def get_src_vocab(self):
        return self._vectorizer.src_vocab
    
    def get_tgt_vocab(self):
        return self._vectorizer.tgt_vocab
    
    def get_num_batches(self, batch_size):
        return len(self) // batch_size
    
    def get_vectorizer(self):
        return self._vectorizer
