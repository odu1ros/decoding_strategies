import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

class TextGenerator:
    def __init__(self, model_name, device=None):
        """
        Generator initialization
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16,
            device_map=self.device
        )

        # inference mode
        self.model.eval()

    def _get_next_token_logits(self, input_ids, past_key_values=None):
        """
        Get logits for the next token
        We can use past_key_values to avoid recomputing the entire sequence (KV-Cache)

        params:
            - input_ids: [batch_size, sequence_length]
            - past_key_values: cached key/values from previous steps
            
        returns:
            - next_token_logits: [batch_size, vocab_size]
            - past_key_values: updated cached key/values
        """
        with torch.no_grad():
            outputs = self.model(
                input_ids,
                past_key_values=past_key_values,
                use_cache=True
                ) # [batch_size, sequence_length, vocab_size]

        next_token_logits = outputs.logits[:, -1, :] # [batch_size, vocab_size]
        return next_token_logits, outputs.past_key_values

    def _filter_top_k_top_p(self, logits, top_k:int, top_p:float):
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
            top_k_values = torch.topk(logits, top_k)[0][..., -1, None]
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
    
    def _repetition_penalty(self, logits, input_ids, penalty:float):
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
    

    def _sample_decoding(self, input_ids, max_new_tokens:int, 
                         temperature:float, top_k:int, top_p:float, 
                         repetition_penalty:float):
        """
        Sampling text generation
        Supports streaming only

        Strategies: 
            - Greedy
            - Sampling

        params:
            - input_ids: [1, sequence_length]
            - max_new_tokens: maximum number of tokens to generate
            - temperature: sampling temperature
            - top_k: top-k filtering
            - top_p: top-p filtering
            - repetition_penalty: repetition penalty factor

        yields:
            - new_word: newly generated word (streaming)
        """
        print("Starting Sampling decoding.")
        generated_ids = input_ids

        # we want to use cache to feed only the last generated token at each step
        past_key_values = None

        # for first step, current input is the whole input_ids
        current_input = input_ids
        
        for _ in range(max_new_tokens):
            logits, past_key_values = self._get_next_token_logits(current_input, past_key_values)
            
            # -- Repetition penalty --
            logits = self._repetition_penalty(logits, generated_ids, repetition_penalty)

            #  -- Temperature -- 
            if temperature > 0:
                logits = logits / temperature
            
            # -- Top-K / Top-P filtering -- 
            logits = self._filter_top_k_top_p(logits, top_k, top_p)
            
            # -- Token selection --
            if temperature == 0:
                # :Greedy:      the highest probability token
                next_token = torch.argmax(logits, dim=-1).unsqueeze(-1)
            else:
                # :Sampling:    sample from the filtered distribution
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)

            # now current input is the last generated token
            current_input = next_token

            # decode and yield
            new_word = self.tokenizer.decode(next_token[0], skip_special_tokens=True)
            yield new_word
            
            # break if end of sentence
            if next_token.item() == self.tokenizer.eos_token_id:
                break
                

    def _beam_search_decoding(self, input_ids, max_new_tokens:int, num_beams:int, repetition_penalty:float):
        """
        Beam Search text generation
        Streaming not supported

        Strategy: 
            - Beam Search

        params:
            - input_ids: [1, sequence_length]
            - max_new_tokens: maximum number of tokens to generate
            - num_beams: number of beams

        returns:
            - generated_ids: [1, sequence_length + max_new_tokens]
        """
        print("Starting Beam Search decoding.")
        beams = [(0.0, input_ids)] # (score, sequence)
        finished = []
        
        for _ in range(max_new_tokens):
            candidates = []

            # we want to expand each beam
            for score, seq in beams:
                
                with torch.no_grad():
                    outputs = self.model(seq)
                logits = outputs.logits[:, -1, :]

                # -- Repetition penalty --
                logits = self._repetition_penalty(logits, seq, repetition_penalty)
                
                # we want to sum log-probs
                log_probs = F.log_softmax(logits, dim=-1)
                
                # top-k best tokens for current beam (k=num_beams)
                top_scores, top_ids = torch.topk(log_probs, num_beams, dim=-1)
                
                # one beam produces num_beams candidates
                for i in range(num_beams):
                    token_id = top_ids[0, i].unsqueeze(-1).unsqueeze(-1)    # [1, 1]
                    token_score = top_scores[0, i].item()                   # scalar
                    
                    new_score = score + token_score                         # cumulative score for the beam
                    new_seq = torch.cat([seq, token_id], dim=-1)            # update sequence
                    
                    # check if end of sentence reached
                    if token_id.item() == self.tokenizer.eos_token_id:
                        finished.append((new_score, new_seq))
                    else:
                        candidates.append((new_score, new_seq))
            
            # proceed with num_beams best candidates
            candidates.sort(key=lambda x: x[0], reverse=True)
            beams = candidates[:num_beams]
            
            # if none, break
            if not beams:
                break
                
        # if max_new_tokens reached, but not the eos_token, consider best num_beams beams as finished
        finished.extend(beams)
        finished.sort(key=lambda x: x[0], reverse=True)
        
        return finished[0][1]

    def generate(self, prompt, max_new_tokens:int=100, 
                 temperature:float=1.0, top_k:int=50, top_p:float=0.9, 
                 repetition_penalty:float=1.01, num_beams:int=1, streaming=False):
        """
        Text generation function

        params:
            - prompt: input text prompt
            - max_new_tokens: maximum number of tokens to generate
            - temperature: sampling temperature
            - top_k: top-k filtering
            - top_p: top-p filtering
            - repetition_penalty: repetition penalty factor
            - num_beams: number of beams for Beam Search (used if > 1)
        """
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        # if num_beams > 1 use Beam Search
        if num_beams > 1:
            if streaming:
                print("Beam Search does not support streaming.")

            output_ids = self._beam_search_decoding(input_ids, max_new_tokens, num_beams, repetition_penalty)
            return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # else Sampling
        else:
            streamer = self._sample_decoding(
                input_ids, max_new_tokens, temperature, top_k, top_p, repetition_penalty
            )
            
            # streaming
            if streaming:
                print("Streaming enabled.")
                return streamer
            # full text
            else:
                print("Streaming disabled.")
                full_text = "".join([word for word in streamer])
                return full_text
            
