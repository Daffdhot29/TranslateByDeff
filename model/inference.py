from pathlib import Path

import sentencepiece as spm
import torch
from huggingface_hub import hf_hub_download

from .architecture import Encoder, Decoder, Seq2Seq


EMBED = 256
HIDDEN_SIZE = 512
NUM_LAYERS = 2
DROPOUT = 0.3


def load_translator(checkpoint_path, tokenizer_path, device=None):
    device = device or torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    sp = spm.SentencePieceProcessor()

    if not sp.load(str(tokenizer_path)):
        raise RuntimeError(f"Failed to load tokenizer: {tokenizer_path}")

    vocab_size = sp.get_piece_size()
    pad_idx = sp.pad_id()

    encoder = Encoder(
        vocab_size,
        EMBED,
        HIDDEN_SIZE,
        pad_idx,
        NUM_LAYERS,
        DROPOUT,
    )

    decoder = Decoder(
        vocab_size,
        EMBED,
        HIDDEN_SIZE,
        pad_idx,
        NUM_LAYERS,
        DROPOUT,
    )

    model = Seq2Seq(
        encoder,
        decoder,
        pad_idx,
        device,
    ).to(device)

    checkpoint_path = Path(checkpoint_path)

    if checkpoint_path.exists():
        final_checkpoint_path = checkpoint_path
    else:
        final_checkpoint_path = hf_hub_download(
            repo_id="arkandaffa/translateseq2seqbydeff",
            filename="best_seq2seq.pt",
        )

    checkpoint = torch.load(
        str(final_checkpoint_path),
        map_location=device,
    )

    state_dict = (
        checkpoint["model"]
        if isinstance(checkpoint, dict) and "model" in checkpoint
        else checkpoint
    )

    model.load_state_dict(state_dict)
    model.eval()

    return model, sp, device


def translate_beam(
    model,
    sentence,
    sp,
    device,
    beam_size=3,
    max_len=50,
    alpha=0.7,
    reverse_src=True,
):
    sentence = sentence.strip()

    if not sentence:
        return ""

    model.eval()

    sos_id = sp.piece_to_id("<sos>")
    eos_id = sp.piece_to_id("<eos>")

    if reverse_src:
        sentence = " ".join(sentence.split()[::-1])

    src_ids = (
        [sos_id]
        + sp.encode(sentence, out_type=int)
        + [eos_id]
    )

    src = torch.tensor(
        src_ids,
        dtype=torch.long,
    ).unsqueeze(0).to(device)

    with torch.no_grad():
        encoder_outputs, hidden, cell = model.encoder(src)

        hidden = [
            torch.tanh(model.bridge_hidden(h))
            for h in hidden
        ]

        cell = [
            torch.tanh(model.bridge_cell(c))
            for c in cell
        ]

        mask = model.create_mask(src)
        beams = [([sos_id], 0.0, hidden, cell)]

        for _ in range(max_len):
            new_beams = []
            all_finished = True

            for seq, score, h_state, c_state in beams:
                if seq[-1] == eos_id:
                    new_beams.append(
                        (seq, score, h_state, c_state)
                    )
                    continue

                all_finished = False

                input_token = torch.tensor(
                    [seq[-1]],
                    dtype=torch.long,
                    device=device,
                )

                output, new_hidden, new_cell = model.decoder(
                    input_token,
                    h_state,
                    c_state,
                    encoder_outputs,
                    mask,
                )

                log_probs = torch.log_softmax(
                    output,
                    dim=-1,
                )

                topk = torch.topk(
                    log_probs,
                    beam_size,
                )

                for k in range(beam_size):
                    token = topk.indices[0][k].item()
                    token_log_prob = topk.values[0][k].item()

                    new_seq = seq + [token]

                    new_score = (
                        score + token_log_prob
                    ) / (len(new_seq) ** alpha)

                    new_beams.append(
                        (
                            new_seq,
                            new_score,
                            [h.clone() for h in new_hidden],
                            [c.clone() for c in new_cell],
                        )
                    )

            if all_finished:
                break

            beams = sorted(
                new_beams,
                key=lambda x: x[1],
                reverse=True,
            )[:beam_size]

    best_seq = beams[0][0][1:]

    if eos_id in best_seq:
        best_seq = best_seq[
            :best_seq.index(eos_id)
        ]

    return sp.decode(best_seq)