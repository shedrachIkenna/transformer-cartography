import torch 
from transformers import AutoModelCausalLM, AutoTokenizer 

class ResidualStreamExtractor: 
    """
    Extracts the full residual stream trajectory from GPT-2

    Captures the post-(wte + wpe) embedding (the true input to block 0)
    plus the output for each transformer block, yielding a tensor shape 
    [L+1, S, D] where: 
        L = number of layers 
        S = Sequence length 
        D = hidden dim 

    wte: word token embedding 
    wpe: word position embedding 
    L + 1: L+1 layer states total for L blocks 
    """