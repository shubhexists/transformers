import torch
from torch.utils.data import Dataset


class BilingualDataset(Dataset):
    def __init__(
        self, dataset, tokenizer_src, tokenizer_target, src_lang, target_lang, seq_len
    ):
        """
        Initializes a new instance of this Dataset. One language pair of the dataset
        https://huggingface.co/datasets/Helsinki-NLP/opus_books
        """
        super().__init__()
        self.seq_len = seq_len
        self.src_lang = src_lang
        self.tokenizer_target = tokenizer_target
        self.tokenizer_src = tokenizer_src
        self.target_lang = target_lang
        self.dataset = dataset

        self.start_of_sentence_token = torch.tensor(
            [tokenizer_target.token_to_id("[SOS]")], dtype=torch.int64
        )
        self.end_of_sentence_token = torch.tensor(
            [tokenizer_target.token_to_id("[EOS]")], dtype=torch.int64
        )
        self.padding_token = torch.tensor(
            [tokenizer_target.token_to_id("[PAD]")], dtype=torch.int64
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        """
        This function takes the text of the sentence from the dataset, tokenizes it using the
        tokenizer_src and the tokenizer_target respectively and constructs the tensors used to pass to the transformer
        """
        src_target_pair = self.dataset[index]
        src_text = src_target_pair["translation"][self.src_lang]
        target_text = src_target_pair["translation"][self.target_lang]

        encoder_input_tokens = self.tokenizer_src.encode(src_text).ids
        decoder_input_tokens = self.tokenizer_target.encode(target_text).ids

        enc_num_padding_tokens = self.seq_len - len(encoder_input_tokens) - 2
        dec_num_padding_tokens = self.seq_len - len(decoder_input_tokens) - 1

        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("Sentence is too long")

        encoder_input = torch.cat(
            [
                self.start_of_sentence_token,
                torch.tensor(encoder_input_tokens, dtype=torch.int64),
                self.end_of_sentence_token,
                torch.tensor(
                    [self.padding_token] * enc_num_padding_tokens, dtype=torch.int64
                ),
            ],
            dim=0,
        )

        decoder_input = torch.cat(
            [
                self.start_of_sentence_token,
                torch.tensor(decoder_input_tokens, dtype=torch.int64),
                torch.tensor(
                    [self.padding_token] * dec_num_padding_tokens, dtype=torch.int64
                ),
            ],
            dim=0,
        )

        label = torch.cat(
            [
                torch.tensor(decoder_input_tokens, dtype=torch.int64),
                self.end_of_sentence_token,
                torch.tensor(
                    [self.padding_token] * dec_num_padding_tokens, dtype=torch.int64
                ),
            ],
            dim=0,
        )

        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input,  # (seq_len)
            "decoder_input": decoder_input,  # (seq_len)
            "encoder_mask": (encoder_input != self.padding_token)
            .unsqueeze(0)
            .unsqueeze(0)
            .int(),  # (1, 1, seq_len) adding the sequence dimension and batch dimension
            "decoder_mask": (decoder_input != self.padding_token).unsqueeze(0).int()
            & causal_mask(
                decoder_input.size(0)
            ),  # (1, seq_len) & (1, seq_len, seq_len),
            "label": label,  # (seq_len)
            "src_text": src_text,
            "tgt_text": target_text,
        }


def causal_mask(size):
    # This returns everything above the diagonal. Hence we reverse it by mask == 0 in return as we need
    # stuff below the diagonal
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0
