import torch

import MyTrans as trans

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
        
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len

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
        )

        out = self.final_linear(out)

        return out

    def get_key_padding_mask(self, tokens):
        key_padding_mask = torch.zeros(tokens.size(), device=tokens.device)
        key_padding_mask[tokens == self.pad_idx] = -torch.inf
        return key_padding_mask