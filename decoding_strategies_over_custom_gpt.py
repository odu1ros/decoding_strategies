import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union

from gpt import Transformer, CharTokenizer

# ==============================================================================
#                                    UTILS                                     #
# ==============================================================================

class LogitsProcessor:
    """
    Helper class with specific logit processing methods
    """

    @staticmethod
    def filter_top_k_top_p(logits: torch.Tensor, top_k: int, top_p: float) -> torch.Tensor:
        """
        Top-k and Top-P filtration

        params:
            - logits: [batch_size, vocab_size]
            - top_k: keep only top k tokens with highest probability
            - top_p: keep the top tokens with cumulative probability >= top_p

        returns:
            - filtered logits [batch_size, vocab_size]
        """

        # -- Top-K -- 
        if top_k > 0:
            # find the topk logit value then unsqueese
            # safety check: k cannot be larger than vocab size
            k = min(top_k, logits.size(-1))
            top_k_values = torch.topk(logits, k)[0][..., -1, None]
            # others nullify
            logits[logits < top_k_values] = -float('Inf')


        # -- Top-P -- 
        if top_p > 0.0:
            # sort logits descending and calculate cusum probs
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # remove tokens with cum prob above top_p
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()  # shift one element right so that cusum > P
            sorted_indices_to_remove[:, 0] = 0                                          # always keep first token
            
            # assign -inf directly to corresponding logits
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = -float('Inf')
            
        return logits
    
    @staticmethod
    def repetition_penalty(logits: torch.Tensor, input_ids: torch.Tensor, penalty: float) -> torch.Tensor:
        """
        Apply repetition penalty to logits
        Full-batch implementation
        """
        if penalty == 1.0:
            return logits
        
        # gather values of logits for indices from input_ids
        score = torch.gather(logits, 1, input_ids)

        # if score < 0 then to reduce the token probabilities we have to multiply, otherwise divide
        score = torch.where(score < 0, score * penalty, score / penalty)

        # place corresponding scores back to logits
        logits.scatter_(1, input_ids, score)
        
        return logits


# ==============================================================================
#                                STATEGIES                                     #
# ==============================================================================

class DecodingStrategy(ABC):
    """
    Base class for generation strategies
    """
    @abstractmethod
    def generate(self, model: Transformer, input_ids: torch.Tensor, 
                 max_new_tokens: int, eos_token_id: int) -> torch.Tensor:
        """
        Generates a sequence of tokens
        
        returns:
            - output_ids: full generated sequence
        """
        pass


class SamplingStrategy(DecodingStrategy):
    """
    Sampling strategies:
        - Greedy
        - Probabiltstic sampling
    """
    def __init__(self, temperature: float, top_k: int, top_p: float, repetition_penalty: float):
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty

    def generate(self, model: Transformer, input_ids: torch.Tensor, 
                 max_new_tokens: int, eos_token_id: int) -> torch.Tensor:
        
        print("Starting Sampling decoding.")
        generated_ids = input_ids
        
        # here, logic supporting KV-Cache is implemented
        # however, gpt.py does not support past_key_values,
        # so we feed the entire current_input at each step
        current_input = input_ids
        
        for _ in range(max_new_tokens):
            # forward pass
            logits = model(current_input)
            next_token_logits = logits[:, -1, :] # [batch_size, vocab_size]
            
            # -- Repetition penalty --
            next_token_logits = LogitsProcessor.repetition_penalty(
                next_token_logits, generated_ids, self.repetition_penalty
            )

            #  -- Temperature -- 
            if self.temperature > 0:
                next_token_logits = next_token_logits / self.temperature
            
            # -- Top-K / Top-P filtering -- 
            next_token_logits = LogitsProcessor.filter_top_k_top_p(
                next_token_logits, self.top_k, self.top_p
            )
            
            # -- Token selection --
            if self.temperature == 0:
                # :Greedy:      the highest probability token
                print("Strategy: Greedy")
                next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            else:
                # :Sampling:    sample from the filtered distribution
                print("Strategy: Probabilitsic Sampling")
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)

            # now current input is the full sequence
            current_input = generated_ids

            # for KV-Cache, use
            # current_input = next_token
            
            # break if end of sentence
            if next_token.item() == eos_token_id:
                break
        
        return generated_ids


class BeamSearchStrategy(DecodingStrategy):
    """
    Strategy:
        - Beam Search (Deterministic)
        - Beam Sampling (Stochastic)
    """
    def __init__(self, num_beams: int, repetition_penalty: float, 
                 do_sample: bool, temperature: float, top_k: int, top_p: float):
        self.num_beams = num_beams
        self.repetition_penalty = repetition_penalty
        self.do_sample = do_sample
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p

    def generate(self, model: Transformer, input_ids: torch.Tensor, 
                 max_new_tokens: int, eos_token_id: int) -> torch.Tensor:
        
        print("Starting Beam Search decoding.")
        beams = [(0.0, input_ids)] # [(score, sequence)]
        finished = []
        
        for _ in range(max_new_tokens):
            candidates = []

            # we want to expand each beam
            for score, seq in beams:
                
                with torch.no_grad():
                    outputs = model(seq)
                logits = outputs[:, -1, :]

                # -- Repetition penalty --
                logits = LogitsProcessor.repetition_penalty(
                    logits, seq, self.repetition_penalty
                )
                
                if self.do_sample:
                    # :Beam Sampling: Stochastic Beam Search
                    print("Strategy: Stochastic Beam Sampling")
                    
                    # -- Temperature --
                    if self.temperature > 0:
                        logits = logits / self.temperature

                    # -- Top-K / Top-P filtering --
                    logits = LogitsProcessor.filter_top_k_top_p(
                        logits, self.top_k, self.top_p
                    )
                    
                    # calculate probs for sampling
                    probs = F.softmax(logits, dim=-1)
                    
                    # sample num_beams candidates from the distribution
                    top_ids = torch.multinomial(probs, num_samples=self.num_beams)
                    
                    # get corresponding log-probs
                    # we need to re-calculate log_softmax on the processed logits to keep scores consistent
                    log_probs = F.log_softmax(logits, dim=-1)
                    top_scores = torch.gather(log_probs, -1, top_ids)
                    
                else:
                    # :Beam Search: Deterministic Beam Search
                    print("Strategy: Deterministic Beam Search")
                    
                    # we want to sum log-probs
                    log_probs = F.log_softmax(logits, dim=-1)
                    
                    # top-k best tokens for current beam (k=num_beams)
                    top_scores, top_ids = torch.topk(log_probs, self.num_beams, dim=-1)
                
                # one beam produces num_beams candidates
                for i in range(self.num_beams):
                    token_id = top_ids[0, i].unsqueeze(-1).unsqueeze(-1)    # [1, 1]
                    token_score = top_scores[0, i].item()                   # scalar
                    
                    new_score = score + token_score                         # cumulative score for the beam
                    new_seq = torch.cat([seq, token_id], dim=-1)            # update sequence
                    
                    # check if end of sentence reached
                    if token_id.item() == eos_token_id:
                        finished.append((new_score, new_seq))
                    else:
                        candidates.append((new_score, new_seq))
            
            # proceed with num_beams best candidates
            candidates.sort(key=lambda x: x[0], reverse=True)
            beams = candidates[:self.num_beams]
            
            # if none, break
            if not beams:
                break
                
        # if max_new_tokens reached, but not the eos_token, consider best num_beams beams as finished
        finished.extend(beams)
        finished.sort(key=lambda x: x[0], reverse=True)
        
        return finished[0][1]


# ==============================================================================
#                                  GENERATION                                  #
# ==============================================================================

class GenerativeModel(Transformer):
    """
    Wrapper around hand written gpt to support strategies:
        1) num_beams=1, do_sample=False -> Greedy
        2) num_beams=1, do_sample=True -> Sampling
        3) num_beams>1, do_sample=False -> Beam Search
        4) num_beams>1, do_sample=True -> Beam Sampling 
    """

    @torch.no_grad()
    def generate(
        self, 
        text_input: str,
        tokenizer: CharTokenizer,
        max_new_tokens: int = 100, 
        device: str = "cuda",
        do_sample: bool = False,
        temperature: float = 1.0, 
        top_k: int = 50, 
        top_p: float = 0.9, 
        repetition_penalty: float = 1.0, 
        num_beams: int = 1
    ) -> str:
        """
        Text generation function
        """
        
        self.eval()
        self.to(device)

        input_ids_list = tokenizer.tokenize_ids(text_input)
        
        # handle EOS tokens from tokenizer
        if isinstance(input_ids_list, list) and len(input_ids_list) > 0:
            if input_ids_list[-1] == tokenizer.end_token_id:
                input_ids_list = input_ids_list[:-1]

        input_tensor = torch.tensor(input_ids_list, dtype=torch.long).unsqueeze(0).to(device)
        
        # --- Strategy selectoin ---
        strategy: DecodingStrategy

        # if num_beams > 1 use Beam Strategy (Search or Sampling)
        if num_beams > 1:
            strategy = BeamSearchStrategy(
                num_beams=num_beams,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )
        # else Standard (Greedy or Probabilistic Sampling)
        else:
            eff_temp = temperature if do_sample else 0.0
            
            strategy = SamplingStrategy(
                temperature=eff_temp,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty
            )
        
        # --- Generation ---
        output_ids = strategy.generate(
            model=self,
            input_ids=input_tensor,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.end_token_id
        )

        result_list = output_ids[0].tolist()
        return tokenizer.decode(result_list)