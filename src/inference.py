import torch
from train import get_model, greedy_decode, get_or_build_tokenizer
from config import get_config

INPUT_TEXT = "sun rises in the night"

def inference():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    config = get_config()

    tokenizer_src = get_or_build_tokenizer(config, None, config["lang_src"])
    tokenizer_tgt = get_or_build_tokenizer(config, None, config["lang_target"])

    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

    model_filename = "weights/tmodel_19.pt"
    state = torch.load(model_filename, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    tokens = tokenizer_src.encode(INPUT_TEXT).ids
    tokens = [tokenizer_src.token_to_id("[SOS]")] + tokens + [tokenizer_src.token_to_id("[EOS]")]
    encoder_input = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
    encoder_mask = (encoder_input != tokenizer_src.token_to_id("[PAD]")).unsqueeze(0).unsqueeze(0).to(device)

    model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, config["seq_len"], device)
    output_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

    print("Source:", INPUT_TEXT)
    print("Predicted:", output_text)


if __name__ == "__main__":
    inference()
