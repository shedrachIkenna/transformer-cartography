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

        self._residual_stack: list[torch.Tensor] = [] # Where we store the residual we get after each layer 
        self._hooks: list[torch.utils.hooks.RemovableHook] = [] # saves a reference of pytorch internal record of the _hook_fn function which we will use for cleanup 

        self._register_hooks() # function that fires during the forward pass of each layer 

    def _register_hooks(self) -> None: 
        """ 
        This is a setup function that wires up the hook function (_hook_fn) to each layer

        _hook_fn is a callback function that automatically runs every time a layer completes its forward pass 
        """
        # Wires the hook function to the initial layer (wte + wpe)
        # Connects the hook function to get the initial state of the token vector before it enters the first layer 
        h0 = self.model.transformer.drop.register_forward_hook(self._hook_fn) 

        block_hooks = [
            layer.register_forward_hook(self._hook_fn) for layer in self.model.transformer.h 
        ]

        self._hooks = [h0, *block_hooks]


    def remove_hooks(self) -> None: 
        """Remove all registered hooks. It is called when the extractor is no longer needed"""
        for handle in self._hooks: # loops through each layer hook 
            handle.remove() # detaches each hook from the layer 
        
        self._hooks.clear() # clears the hook's list 

    def _hook_fn(self, module, input, output) -> None: 
        """
        Pytorch will automatically run this function everytime each hooked layer completes its forward pass 

        Our aim is to get the layer's output tensor 
        """
        # Get the output state (its a plain tensor. shape is [S, D]) from each layer. 
        # If a layer's output is a tuple, get the item in the first index (output[0] - the layer's plain tensor) else get the output (the plain tensor)
        state = output[0] if isinstance(output, tuple) else output

        # add each layer's output state tensors to the residual stack list 
        # .detach removes the computational graph attached to each tensor to free up memory 
        # move the tensors to cpu
        self._residual_stack.append(state.detach().cpu())

    def extract(self, prompt: str) -> tuple[torch.Tensor, list[str]]:
        """
        Run the forward pass and return the residual stream tensor and token list 

        Returns: 
            data: torch.Tensor 
                  shape [L+1, S, D] - [layers + 1, Token_List, Hidden_Dim]
            tokens: list[str]
                    The tokenized prompt as human readable token strings 
        """

        # clear stack before a new run to get residuals from all layers 
        self._residual_stack.clear()

        # tokenize the prompt and return pytorch tensors ("pt")
        # Each token in the prompt gets mapped to an integer ID from GPT-2 vocabulary 
        inputs = self.tokenizer(prompt, returns_tensors="pt")

        # The token_ids gets converted back into tokens. 
        # The idea is that the tokenizer from the previous line uses BPE
        # So the token we get in the next line might end up different from the ones from the prompt (because of subwords)
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["inputs_ids"][0])

        with torch.no_grad(): # stop the model from saving the computational graph during the forward pass. we are not training here 
            self.model(**inputs) # pass the tokenized prompt to the model 
