import torch
import sentencepiece as spm
from seq2seq.models import Seq2SeqModel
from seq2seq.beam import BeamSearch, BeamSearchNode
import torch.nn.functional as F


def decode(model: Seq2SeqModel, src_tokens: torch.Tensor, src_pad_mask: torch.Tensor, max_out_len: int,
           tgt_tokenizer: spm.SentencePieceProcessor, args, device: torch.device, beam_size: int = 1):
    """Decodes a sequence without teacher forcing.

    By default (beam_size=1) this uses greedy decoding (existing behaviour). If
    beam_size > 1, performs per-sample beam search using the BeamSearch helpers.

    Note: beam-search currently handles each example in the batch separately
    (runs encoding once per example). This keeps the implementation simple and
    avoids incremental-state machinery in the decoder.
    """
    batch_size = src_tokens.size(0)
    BOS = tgt_tokenizer.bos_id()
    EOS = tgt_tokenizer.eos_id()
    PAD = tgt_tokenizer.pad_id()

    # Fast path: greedy decoding (original behaviour)
    if beam_size == 1:
        generated = torch.full((batch_size, 1), BOS, dtype=torch.long, device=device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        for t in range(max_out_len):
            # Create target padding mask with correct batch dimension
            max_len = model.decoder.pos_embed.size(1)
            if generated.size(1) > max_len:
                generated = generated[:, :max_len]
            trg_pad_mask = (generated == PAD).unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, seq_len)
            # Forward pass: use only the generated tokens so far
            output = model(src_tokens, src_pad_mask, generated, trg_pad_mask).to(device)
            # Get the logits for the last time step
            next_token_logits = output[:, -1, :]  # last time step
            next_tokens = next_token_logits.argmax(dim=-1, keepdim=True)  # greedy

            # Append next token to each sequence
            generated = torch.cat([generated, next_tokens], dim=1)

            # Mark sequences as finished if EOS is generated
            finished = finished | (next_tokens.squeeze(1) == EOS)
            if finished.all():
                break
        # Remove initial BOS token and anything after EOS
        predicted_tokens = []
        for seq in generated[:, 1:].tolist():
            if EOS in seq:
                idx = seq.index(EOS)
                seq = seq[:idx+1]
            predicted_tokens.append(seq)
        return predicted_tokens

    # Beam search path (beam_size > 1)
    predicted_batch = []

    # We'll run beam search independently for each example in the batch.
    for b in range(batch_size):
        # Single example tensors
        src_b = src_tokens[b : b + 1]
        src_mask_b = src_pad_mask[b : b + 1]

        # Run encoder once for this example
        with torch.no_grad():
            # The model.forward calls encoder then decoder; directly call encoder for beam search
            encoder_out = model.encoder(src_b, src_mask_b)

        beam = BeamSearch(beam_size=beam_size, max_len=max_out_len, pad=PAD)

        # Create initial node with BOS
        init_seq = torch.tensor([BOS], dtype=torch.long, device=device)
        init_node = BeamSearchNode(search=None, emb=None, lstm_out=None, final_hidden=None,
                                   final_cell=None, mask=None, sequence=init_seq, logProb=0.0, length=1)
        # add initial node with its score (negative logprob)
        beam.add(-init_node.eval(), init_node)

        for t in range(max_out_len):
            # collect current beams
            current = beam.get_current_beams()
            if len(current) == 0:
                break

            sequences = []
            for _, node in current:
                sequences.append(node.sequence.unsqueeze(0))  # (1, seq_len)
            trg = torch.cat(sequences, dim=0).to(device)  # (num_beams, seq_len)

            # ensure trg fits into decoder pos_embed
            max_len = model.decoder.pos_embed.size(1)
            if trg.size(1) > max_len:
                trg = trg[:, :max_len]

            trg_pad_mask = (trg == PAD).unsqueeze(1).unsqueeze(2)

            # Expand encoder_out to match beam batch size
            enc_expand = encoder_out.expand(trg.size(0), -1, -1)
            src_mask_expand = src_mask_b.expand(trg.size(0), -1, -1, -1) if src_mask_b is not None else None

            with torch.no_grad():
                logits = model.decoder(enc_expand, src_mask_expand, trg, trg_pad_mask)
                # take logits for last timestep for each beam
                last_logits = logits[:, -1, :]
                log_probs = F.log_softmax(last_logits, dim=-1)  # (num_beams, vocab)

            # For each beam, expand to topk candidates
            all_candidates = []  # tuples (score, node)
            for beam_idx, (score_val, node) in enumerate(current):
                node_logp = node.logp
                lp = log_probs[beam_idx]  # (vocab,)
                topk_logp, topk_idx = torch.topk(lp, k=min(beam_size, lp.size(0)))
                for k in range(topk_idx.size(0)):
                    token = int(topk_idx[k].item())
                    token_logp = float(topk_logp[k].item())
                    new_logp = node_logp + token_logp
                    new_seq = torch.cat((node.sequence.cpu(), torch.tensor([token], dtype=torch.long)), dim=0)
                    new_node = BeamSearchNode(search=None, emb=None, lstm_out=None, final_hidden=None,
                                              final_cell=None, mask=None, sequence=new_seq, logProb=new_logp,
                                              length=node.length + 1)
                    # if token is EOS, add to final
                    if token == EOS:
                        beam.add_final(-new_node.eval(), new_node)
                    else:
                        all_candidates.append(( -new_node.eval(), new_node))

            # add candidates to beam
            for sc, nd in all_candidates:
                beam.add(sc, nd)

            # prune to keep only beam_size beams
            try:
                beam.prune()
            except Exception:
                # If there are fewer nodes than expected, ignore prune
                pass

            # stop early if we have enough final hypotheses
            if beam.final.qsize() >= beam_size:
                break

        # pick best node
        best_score, best_node = beam.get_best()
        seq = best_node.sequence.tolist()
        # Remove initial BOS and truncate after EOS
        if len(seq) > 0 and seq[0] == BOS:
            seq = seq[1:]
        if EOS in seq:
            idx = seq.index(EOS)
            seq = seq[: idx + 1]
        predicted_batch.append(seq)

    return predicted_batch
