import torch
import sentencepiece as spm
from seq2seq.models import Seq2SeqModel
import math

def decode(model: Seq2SeqModel, src_tokens: torch.Tensor, src_pad_mask: torch.Tensor, max_out_len: int,
           tgt_tokenizer: spm.SentencePieceProcessor, args, device: torch.device):
    """Decodes a sequence without teacher forcing. Works by relying on the model's own predictions, rather than the ground truth (trg_)"""
    batch_size = src_tokens.size(0)
    BOS = tgt_tokenizer.bos_id()
    EOS = tgt_tokenizer.eos_id()
    PAD = tgt_tokenizer.pad_id()
    generated = torch.full((batch_size, 1), BOS, dtype=torch.long, device=device)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    # Encode the source sentence once, outside of generated loop
    with torch.no_grad():
        encoded_src = model.encoder(src_tokens, src_pad_mask)

    for t in range(max_out_len):
        # Create target padding mask with correct batch dimension
        max_len = model.decoder.pos_embed.size(1)
        if generated.size(1) > max_len:
            generated = generated[:, :max_len]
        # Ensure trg_pad_mask has shape (batch_size, seq_len)
        trg_pad_mask = (generated == PAD).unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, seq_len)
        
        # Forward pass: use only the generated tokens so far
        # decode the output
        output = model.decoder(encoded_src, src_pad_mask, generated, trg_pad_mask).to(device)
        
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

def beam_search_decode(model: Seq2SeqModel, src_tokens: torch.Tensor, src_pad_mask: torch.Tensor, max_out_len: int,
                       tgt_tokenizer: spm.SentencePieceProcessor, args, device: torch.device, beam_size: int = 5, alpha: float = 0.7):
    """Beam Search decoding compatible with Transformer-based Seq2Seq models."""
    model.eval()
    BOS, EOS, PAD = tgt_tokenizer.bos_id(), tgt_tokenizer.eos_id(), tgt_tokenizer.pad_id()
    # __QUESTION 1: what does this line set up and why is the beam represented this way?
    beams = [(torch.tensor([[BOS]], device=device), 0.0)]

    # Encode the source sentence once, outside of generated loop
    with torch.no_grad():
        encoded_src = model.encoder(src_tokens, src_pad_mask)

    for _ in range(max_out_len):
        new_beams = []
        for seq, score in beams:
            if seq[0, -1].item() == EOS:
                new_beams.append((seq, score))
                continue
            with torch.no_grad():
                max_len = model.decoder.pos_embed.size(1)
                if seq.size(1) > max_len:
                    seq = seq[:, :max_len]
                # __QUESTION 2: Why do we need to create trg_pad_mask here and how does it affect the model's predictions?
                trg_pad_mask = (seq == PAD)[:, None, None, :]

                # decode the last newly generated token, using encoded source and previous generated tokens
                logits = model.decoder(encoded_src, src_pad_mask, seq, trg_pad_mask)[:, -1, :]
                
                # __QUESTION 3: Explain the purpose of applying log_softmax and selecting top-k tokens here.
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                topk_log_probs, topk_ids = log_probs.topk(beam_size, dim=-1)

            for k in range(beam_size):
                # __QUESTION 4: explain the tensor shapes and the logic when creating new_seq and new_score below. Is any broadcasting or indexing issue possible?
                new_seq = torch.cat([seq, topk_ids[:, k].unsqueeze(0)], dim=1)
                new_score = score + topk_log_probs[:, k].item()
                new_beams.append((new_seq, new_score))

        # Apply length normalization when sorting beams
        beams = sorted(new_beams, key=lambda x: x[1] / ((5 + x[0].size(1) - 1) ** alpha / (6 ** alpha)), reverse=True)[:beam_size]
        # __QUESTION 5: Why do we check for EOS here and what does it imply for beam search?
        if all(seq[0, -1].item() == EOS for seq, _ in beams):
            break
    best_seq, _ = beams[0]
    # __QUESTION 6: What is returned, and why are we squeezing, converting to list and wrapping in another list here?
    return [best_seq.squeeze(0).tolist()]


def beam_search_decode_relative_pruning(model: Seq2SeqModel, src_tokens: torch.Tensor, src_pad_mask: torch.Tensor, max_out_len: int,
                                        tgt_tokenizer: spm.SentencePieceProcessor, args, device: torch.device, beam_size: int = 5, 
                                        alpha: float = 0.7, rp: float = 0.6):
    """Beam Search with relative pruning as described in paper: prunes candidates with score <= rp * best_score."""
    model.eval()
    BOS, EOS, PAD = tgt_tokenizer.bos_id(), tgt_tokenizer.eos_id(), tgt_tokenizer.pad_id()
    beams = [(torch.tensor([[BOS]], device=device), 0.0)]  # (sequence, score)
    
    for _ in range(max_out_len):
        new_beams = []
        
        for seq, score in beams:
            # If sequence already ended with EOS, keep as is
            if seq[0, -1].item() == EOS:
                new_beams.append((seq, score))
                continue
                
            with torch.no_grad():
                # Handle sequence length constraints
                max_len = model.decoder.pos_embed.size(1)
                if seq.size(1) > max_len:
                    seq = seq[:, :max_len]
                
                trg_pad_mask = (seq == PAD)[:, None, None, :]
                logits = model(src_tokens, src_pad_mask, seq, trg_pad_mask)[:, -1, :]
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                topk_log_probs, topk_ids = log_probs.topk(beam_size, dim=-1)

            # Expand each beam
            for k in range(beam_size):
                new_seq = torch.cat([seq, topk_ids[:, k].unsqueeze(0)], dim=1)
                new_score = score + topk_log_probs[:, k].item()
                new_beams.append((new_seq, new_score))

        # Apply relative pruning before selecting top beams
        if new_beams:
            # Find best raw score (not normalized)
            best_score = max(score for _, score in new_beams)
            
            threshold = best_score + math.log(rp) if rp > 0 else float('-inf')
            
            # Keep candidates with score >= threshold (since higher log prob is better)
            pruned_beams = [(seq, score) for seq, score in new_beams if score >= threshold]
            
            # If pruning removed too many, keep at least beam_size candidates
            if len(pruned_beams) < beam_size:
                # Fall back to top beams without pruning
                pruned_beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_size]
            
            # Apply length normalization and select top beams
            beams = sorted(pruned_beams, 
                          key=lambda x: x[1] / ((5 + x[0].size(1) - 1) ** alpha / (6 ** alpha)), 
                          reverse=True)[:beam_size]
        else:
            break

        # Early stopping if all beams end with EOS
        if all(seq[0, -1].item() == EOS for seq, _ in beams):
            break
    
    best_seq, _ = beams[0]
    return [best_seq.squeeze(0).tolist()]

def beam_search_decode_maximum_candidate(model: Seq2SeqModel, src_tokens: torch.Tensor, src_pad_mask: torch.Tensor, max_out_len: int,
                                         tgt_tokenizer: spm.SentencePieceProcessor, args, device: torch.device, beam_size: int = 5, 
                                         alpha: float = 0.7, mc: int = 3):
    """Beam Search with maximum candidates per node as described in paper: 
       limits number of candidates with same history to mc."""
    model.eval()
    BOS, EOS, PAD = tgt_tokenizer.bos_id(), tgt_tokenizer.eos_id(), tgt_tokenizer.pad_id()
    beams = [(torch.tensor([[BOS]], device=device), 0.0)]  # (sequence, score)
    
    for _ in range(max_out_len):
        new_beams = []
        
        for seq, score in beams:
            # If sequence already ended with EOS, keep as is
            if seq[0, -1].item() == EOS:
                new_beams.append((seq, score))
                continue
                
            with torch.no_grad():
                # Handle sequence length constraints
                max_len = model.decoder.pos_embed.size(1)
                if seq.size(1) > max_len:
                    seq = seq[:, :max_len]
                
                trg_pad_mask = (seq == PAD)[:, None, None, :]
                logits = model(src_tokens, src_pad_mask, seq, trg_pad_mask)[:, -1, :]
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                topk_log_probs, topk_ids = log_probs.topk(beam_size, dim=-1)

            # Expand each beam
            for k in range(beam_size):
                new_seq = torch.cat([seq, topk_ids[:, k].unsqueeze(0)], dim=1)
                new_score = score + topk_log_probs[:, k].item()
                new_beams.append((new_seq, new_score))

        # Group beams by their history (all tokens except the last one)
        history_groups = {}
        for seq, score in new_beams:
            # Extract history: all tokens except the last one
            history = tuple(seq.squeeze(0).tolist()[:-1])
            
            if history not in history_groups:
                history_groups[history] = []
            history_groups[history].append((seq, score))
        
        # For each history group, keep only top mc candidates
        pruned_beams = []
        for history, candidates in history_groups.items():
            # Sort candidates by score within this history group
            candidates_sorted = sorted(candidates, key=lambda x: x[1], reverse=True)
            # Keep only top mc candidates from each history
            pruned_beams.extend(candidates_sorted[:mc])
        
        # Get normalized scores for all candidates
        normalized_scores = []
        for seq, score in pruned_beams:
            norm_score = score / ((5 + seq.size(1) - 1) ** alpha / (6 ** alpha))
            normalized_scores.append((seq, score, norm_score))
        
        # Sort by normalized score and keep top beam_size
        normalized_scores.sort(key=lambda x: x[2], reverse=True)
        beams = [(seq, score) for seq, score, _ in normalized_scores[:beam_size]]

        # Early stopping if all beams end with EOS
        if all(seq[0, -1].item() == EOS for seq, _ in beams):
            break
    
    best_seq, _ = beams[0]
    return [best_seq.squeeze(0).tolist()]