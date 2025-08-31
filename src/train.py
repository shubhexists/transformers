import torch
import torch.nn as nn
from config import get_config, get_weights_file_path
from torch.utils.data import random_split, DataLoader
from datasets import load_dataset
from tokenizers import Tokenizer
from dataset import BilingualDataset, causal_mask
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from pathlib import Path
from model import build_transformer, Transformer
from tqdm import tqdm
import warnings


def greedy_decode(
    model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device
):
    sos_idx = tokenizer_tgt.token_to_id("[SOS]")
    eos_idx = tokenizer_tgt.token_to_id("[EOS]")

    encoder_output = model.encode(source, source_mask)
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break

        decoder_mask = (
            causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)
        )

        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        prob = model.projection(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [
                decoder_input,
                torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device),
            ],
            dim=1,
        )

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)


def run_validation(
    model,
    validation_dataset,
    tokenizer_src,
    tokenizer_target,
    max_len,
    device,
    print_msg,
    num_examples=2,
):
    model.eval()
    count = 0

    console_width = 80
    with torch.no_grad():
        for batch in validation_dataset:
            count += 1
            encoder_input = batch["encoder_input"].to(device)  # (b, seq_len)
            encoder_mask = batch["encoder_mask"].to(device)  # (b, 1, 1, seq_len)

            # check that the batch size is 1
            assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"

            model_out = greedy_decode(
                model,
                encoder_input,
                encoder_mask,
                tokenizer_src,
                tokenizer_target,
                max_len,
                device,
            )

            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            model_out_text = tokenizer_target.decode(model_out.detach().cpu().numpy())

            print_msg("-" * console_width)
            print_msg(f"{'SOURCE: ':>12}{source_text}")
            print_msg(f"{'TARGET: ':>12}{target_text}")
            print_msg(f"{'PREDICTED: ':>12}{model_out_text}")

            if count == num_examples:
                print_msg("-" * console_width)
                break


def get_all_sentences(dataset, lang):
    for item in dataset:
        yield item["translation"][lang]


def get_or_build_tokenizer(config, dataset, lang):
    tokenizer_path = Path(config["tokenizer_file"].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(
            special_tokens=["[UNK]", "[SOS]", "[EOS]", "[PAD]"], min_frequency=2
        )
        tokenizer.train_from_iterator(get_all_sentences(dataset, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer


def get_dataset(config):
    dataset_raw = load_dataset(
        "opus_books", f"{config['lang_src']}-{config['lang_target']}", split="train"
    )

    tokenizer_src = get_or_build_tokenizer(config, dataset_raw, config["lang_src"])
    tokenizer_target = get_or_build_tokenizer(
        config, dataset_raw, config["lang_target"]
    )

    train_dataset_size = int(0.9 * len(dataset_raw))
    validation_dataset_size = len(dataset_raw) - train_dataset_size

    train_dataset_raw, validation_dataset_raw = random_split(
        dataset_raw, [train_dataset_size, validation_dataset_size]
    )

    train_dataset = BilingualDataset(
        train_dataset_raw,
        tokenizer_src,
        tokenizer_target,
        config["lang_src"],
        config["lang_target"],
        config["seq_len"],
    )

    validation_dataset = BilingualDataset(
        validation_dataset_raw,
        tokenizer_src,
        tokenizer_target,
        config["lang_src"],
        config["lang_target"],
        config["seq_len"],
    )

    max_len_src = 0
    max_len_target = 0

    for item in dataset_raw:
        src_ids = tokenizer_src.encode(item["translation"][config["lang_src"]]).ids
        target_ids = tokenizer_src.encode(
            item["translation"][config["lang_target"]]
        ).ids

        max_len_src = max(len(src_ids), max_len_src)
        max_len_target = max(len(target_ids), max_len_target)

    train_dataloader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True
    )
    validation_dataloader = DataLoader(
        validation_dataset, batch_size=config["batch_size"], shuffle=True
    )

    return train_dataloader, validation_dataloader, tokenizer_src, tokenizer_target


def get_model(config, vocab_src_len, vocab_target_length) -> Transformer:
    model = build_transformer(
        vocab_src_len,
        vocab_target_length,
        config["seq_len"],
        config["seq_len"],
        d_model=config["d_model"],
        N=4,
        head=4,
        dropout=0.1,
        d_ff=256,
    )

    return model


def train_model(config) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    Path(config["model_folder"]).mkdir(parents=True, exist_ok=True)

    train_dataloader, validation_dataloader, tokenizer_src, tokenizer_target = (
        get_dataset(config)
    )
    model = get_model(
        config, tokenizer_src.get_vocab_size(), tokenizer_target.get_vocab_size()
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], eps=1e-9)
    initial_epoch = 0
    global_step = 0

    if config["preload"]:
        model_filename = get_weights_file_path(config, config["preload"])
        state = torch.load(model_filename)
        initial_epoch = state["epoch"] + 1
        optimizer.load_state_dict(state["optimizer_state_dict"])
        global_step = state["global_step"]

    loss_fn = nn.CrossEntropyLoss(
        ignore_index=tokenizer_src.token_to_id("[PAD]"), label_smoothing=0.1
    ).to(device)

    for epoch in range(initial_epoch, config["num_epochs"]):
        batch_iterator = tqdm(train_dataloader, desc=f"Processing epoch : {epoch:02d}")
        for batch in batch_iterator:
            model.train()
            encoder_input = batch["encoder_input"].to(device)  # (b, seq_len)
            decoder_input = batch["decoder_input"].to(device)  # (B, seq_len)
            encoder_mask = batch["encoder_mask"].to(device)  # (B, 1, 1, seq_len)
            decoder_mask = batch["decoder_mask"].to(device)  # (B, 1, seq_len, seq_len)

            encoder_output = model.encode(
                encoder_input, encoder_mask
            )  # (B, seq_len, d_model)
            decoder_output = model.decode(
                encoder_output, encoder_mask, decoder_input, decoder_mask
            )  # (B, seq_len, d_model)
            proj_output = model.projection(decoder_output)  # (B, seq_len, vocab_size)

            label = batch["label"].to(device)  # (B, seq_len)

            loss = loss_fn(
                proj_output.view(-1, tokenizer_target.get_vocab_size()), label.view(-1)
            )
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1

        run_validation(
            model,
            validation_dataloader,
            tokenizer_src,
            tokenizer_target,
            config["seq_len"],
            device,
            lambda msg: batch_iterator.write(msg),
        )

    model_filename = get_weights_file_path(config, f"{epoch:02d}")
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "global_step": global_step,
        },
        model_filename,
    )


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)
