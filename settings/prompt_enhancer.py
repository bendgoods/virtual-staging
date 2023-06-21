import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


class PromptEnhancer:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('Gustavosta/MagicPrompt-Stable-Diffusion')
        self.model = AutoModelForCausalLM.from_pretrained('Gustavosta/MagicPrompt-Stable-Diffusion')
        self.pipe = pipeline('text-generation', model=self.model, tokenizer=self.tokenizer, max_new_tokens=200)

    def __call__(self, prompt, n_prompts=3):
        #inputs = self.tokenizer(prompt, return_tensors='pt')
        #outputs = self.model(**inputs, labels=inputs['input_ids'])
        prompts = []
        while len(prompts) < n_prompts:
            out = self.pipe(prompt)
            out = [o['generated_text'].split('\n') for o in out]
            prompts += [o for out_ in out for o in out_ if len(o) > 0]
        return prompts
