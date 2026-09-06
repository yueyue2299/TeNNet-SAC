import re
from pathlib import Path

import torch


PATTERN = re.compile(
    r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|/|:|~|@|\?|>|\*|\$|%[0-9]{2}|[0-9])"
)


class MolTranBertTokenizer:
    """SMI-TED's original regex tokenizer without Transformers internals."""

    def __init__(
        self,
        vocab_file: str = "",
        do_lower_case: bool = False,
        unk_token: str = "<pad>",
        sep_token: str = "<eos>",
        pad_token: str = "<pad>",
        cls_token: str = "<bos>",
        mask_token: str = "<mask>",
        **kwargs,
    ):
        del kwargs
        self.do_lower_case = do_lower_case
        self.unk_token = unk_token
        self.sep_token = sep_token
        self.pad_token = pad_token
        self.cls_token = cls_token
        self.mask_token = mask_token

        tokens = Path(vocab_file).read_text(encoding="utf-8").splitlines()
        self.vocab = {token: index for index, token in enumerate(tokens)}
        self.ids_to_tokens = {index: token for token, index in self.vocab.items()}

        missing = [
            token
            for token in (unk_token, sep_token, pad_token, cls_token, mask_token)
            if token not in self.vocab
        ]
        if missing:
            raise ValueError(f"Missing required tokens in vocabulary: {missing}")

        self.unk_token_id = self.vocab[unk_token]
        self.sep_token_id = self.vocab[sep_token]
        self.pad_token_id = self.vocab[pad_token]
        self.cls_token_id = self.vocab[cls_token]
        self.mask_token_id = self.vocab[mask_token]
        self.padding_idx = self.pad_token_id

    def __len__(self):
        return len(self.vocab)

    @property
    def vocab_size(self):
        return len(self.vocab)

    def tokenize(self, text):
        return PATTERN.findall(text)

    def convert_tokens_to_ids(self, tokens):
        if isinstance(tokens, str):
            return self.vocab.get(tokens, self.unk_token_id)
        return [self.vocab.get(token, self.unk_token_id) for token in tokens]

    def convert_ids_to_tokens(self, ids):
        if isinstance(ids, int):
            return self.ids_to_tokens.get(ids, self.unk_token)
        return [self.ids_to_tokens.get(int(index), self.unk_token) for index in ids]

    def __call__(
        self,
        text,
        padding=False,
        truncation=False,
        add_special_tokens=True,
        return_tensors=None,
        max_length=None,
        **kwargs,
    ):
        del kwargs
        is_single = isinstance(text, str)
        batch = [text] if is_single else list(text)
        encoded = [
            self._encode(
                item,
                truncation=truncation,
                add_special_tokens=add_special_tokens,
                max_length=max_length,
            )
            for item in batch
        ]

        if padding == "max_length":
            if max_length is None:
                raise ValueError("max_length is required when padding='max_length'")
            target_length = max_length
        elif padding:
            target_length = max((len(ids) for ids in encoded), default=0)
        else:
            target_length = None

        attention_mask = [[1] * len(ids) for ids in encoded]
        if target_length is not None:
            for ids, mask in zip(encoded, attention_mask):
                padding_length = target_length - len(ids)
                ids.extend([self.pad_token_id] * padding_length)
                mask.extend([0] * padding_length)

        if return_tensors is not None:
            if return_tensors != "pt":
                raise ValueError("MolTranBertTokenizer only supports return_tensors='pt'")
            try:
                return {
                    "input_ids": torch.tensor(encoded, dtype=torch.long),
                    "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                }
            except ValueError as error:
                raise ValueError(
                    "Unable to create a tensor from sequences of different lengths; "
                    "enable padding and/or truncation."
                ) from error

        if is_single:
            return {"input_ids": encoded[0], "attention_mask": attention_mask[0]}
        return {"input_ids": encoded, "attention_mask": attention_mask}

    def _encode(self, text, truncation, add_special_tokens, max_length):
        token_ids = self.convert_tokens_to_ids(self.tokenize(text))

        if truncation and max_length is not None:
            special_token_count = 2 if add_special_tokens else 0
            content_length = max(max_length - special_token_count, 0)
            token_ids = token_ids[:content_length]

        if add_special_tokens:
            token_ids = [self.cls_token_id, *token_ids, self.sep_token_id]

        return token_ids

    def convert_idx_to_tokens(self, idx_tensor):
        return [self.convert_ids_to_tokens(idx) for idx in idx_tensor.tolist()]

    def convert_tokens_to_string(self, tokens):
        stopwords = {self.cls_token, self.sep_token}
        return "".join(word for word in tokens if word not in stopwords)

    def get_padding_idx(self):
        return self.padding_idx

    def idx_to_smiles(self, torch_model, idx):
        tokens = torch_model.tokenizer.convert_idx_to_tokens(idx)
        return torch_model.tokenizer.convert_tokens_to_string(tokens)
