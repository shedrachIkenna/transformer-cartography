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

    def __init__(self, model_name: str = "gpt2"): 
        self.tokenizer = AutoTokenizer.from_pretrained(model_name) # tokenizer for GPT-2
        self.model = AutoModelCausalLM.from_pretrained(model_name) # GPT-2 model weights 
        self.model.eval() # tells pytorch that the model is used for inference not training 

        self._residual_stack: list[torch.Tensor] = [] # Where we store the residual we get from a layer 
        self._hooks: list[torch.utils.hooks.RemovableHook] = [] # saves a reference of pytorch internal record of the _hook_fn function which we will use for cleanup 

        self._register_hooks()