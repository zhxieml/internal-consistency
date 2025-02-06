from typing import Optional

import torch
from peft import AutoPeftModelForCausalLM
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    StoppingCriteriaList,
)

from src.models.prompt_template import whitespace_template

MODEL_CONFIGS = {
    "llama2-7b": dict(
        model_name="meta-llama/Llama-2-7b-hf",
        num_layers=32,
        prompt_template=whitespace_template,
    ),
    "llama2-13b": dict(
        model_name="meta-llama/Llama-2-13b-hf",
        num_layers=40,
        prompt_template=whitespace_template,
    ),
    "mixtral-7b": dict(
        model_name="mistralai/Mistral-7B-v0.1",
        num_layers=32,
        prompt_template=whitespace_template,
    ),
    "mixtral-8x7b": dict(
        # model_name="mistralai/Mixtral-8x7B-v0.1",
        model_name="TheBloke/mixtral-8x7b-v0.1-AWQ",
        num_layers=32,
        prompt_template=whitespace_template,
    ),
}


class HFModel:
    def __init__(
        self,
        modelname: str,
        model_path: Optional[str] = None,
        use_lora: bool = False,
        low_cpu_mem_usage: bool = False,
        torch_dtype: Optional[str] = None,
        device: str = "cuda",
        load_in_8bit: bool = False,
    ):
        # load model and tokenizer
        model_name, num_layers, prompt_template = (
            MODEL_CONFIGS[modelname]["model_name"],
            MODEL_CONFIGS[modelname]["num_layers"],
            MODEL_CONFIGS[modelname]["prompt_template"],
        )
        model_name_or_path = model_path or model_name
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, trust_remote_code=True, padding_side="left"
        )
        if load_in_8bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
        else:
            bnb_config = None

        model_class = AutoPeftModelForCausalLM if use_lora else AutoModelForCausalLM
        model = model_class.from_pretrained(
            model_name_or_path,
            low_cpu_mem_usage=low_cpu_mem_usage,
            torch_dtype=None if load_in_8bit else torch_dtype,
            # quantization_config=bnb_config,
            load_in_8bit=load_in_8bit,
            trust_remote_code=True,
            device_map="auto",
            # attn_implementation="flash_attention_2",
        )
        model.eval()

        # configure padding for tokenizer
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            model.generation_config.pad_token_id = model.generation_config.eos_token_id

        self.model_name = model_name
        self.model = model
        self.tokenizer = tokenizer
        self.num_layers = num_layers
        self.prompt_template = prompt_template

    def make_inputs(self, prompts):
        return self.tokenizer(prompts, padding=True, return_tensors="pt").to(
            self.model.device
        )

    def forward(
        self,
        prompts: list[str],
        **kwargs,
    ):
        # make inputs
        inputs = self.make_inputs(prompts)

        # forward
        outputs = self.model(**inputs, **kwargs)
        outputs["inputs"] = inputs

        return outputs

    def generate(
        self,
        prompts: list[str],
        num_new_tokens: int = 1,
        return_probs: bool = False,
        return_hidden: bool = False,
        return_attention: bool = False,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        top_p: float = 1.0,
        top_k: int = 50,
        temperature: float = 1.0,
        **kwargs,
    ):
        # make inputs
        inputs = self.make_inputs(prompts)
        batch_size, input_length = inputs.input_ids.shape

        # generate
        outputs = self.model.generate(
            **inputs,
            # min_new_tokens=num_new_tokens,
            max_new_tokens=num_new_tokens,
            return_dict_in_generate=True,
            output_scores=return_probs,
            output_hidden_states=return_hidden,
            output_attentions=return_attention,
            do_sample=temperature > 0,
            top_p=top_p,
            top_k=top_k,
            stopping_criteria=stopping_criteria,
            temperature=temperature,
            **kwargs,
        )

        out_sequences = outputs.sequences
        out_scores = outputs.scores
        out_hiddens = outputs.hidden_states
        out_attentions = outputs.attentions

        prompt_tokens = out_sequences[:, :input_length]
        generated_tokens = out_sequences[:, input_length:]
        res = dict(
            prompt=prompt_tokens,
            generation=generated_tokens,
            input_ids=inputs.input_ids,
            logits=out_scores,
        )

        # get probs
        if return_probs:
            transition_scores = self.model.compute_transition_scores(
                out_sequences, out_scores, normalize_logits=True
            )
            log_prob_sums = torch.cumsum(transition_scores, dim=1)
            probs = torch.exp(log_prob_sums)
            res["probs"] = probs

        if return_hidden:
            res["hidden_states"] = out_hiddens
        if return_attention:
            res["attentions"] = out_attentions

        return res

    def set_prompt_template(self, prompts: list[str]):
        return [self.prompt_template(p) for p in prompts]

    @staticmethod
    def get_norm_and_head(model):
        if model.config.is_encoder_decoder:
            pointer = model.decoder
        else:
            pointer = model
        if hasattr(pointer, "final_layer_norm"):
            norm_fn = pointer.final_layer_norm
        elif hasattr(pointer, "gpt_neox"):
            norm_fn = pointer.gpt_neox.final_layer_norm
        elif hasattr(pointer.model, "norm"):
            norm_fn = pointer.model.norm
        elif hasattr(pointer.model, "final_layernorm"):
            norm_fn = pointer.model.final_layernorm
        else:
            raise NotImplementedError

        if hasattr(model, "lm_head"):
            head_fn = model.lm_head
        elif hasattr(model, "embed_out"):
            head_fn = model.embed_out
        else:
            raise NotImplementedError

        return norm_fn, head_fn
