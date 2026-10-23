"""Local official Strands loader and exact MiniLM cross-encoder ranking.

Only construct these objects inside the caller's OS GPU lease after admission.
Importing this module imports neither Torch nor Transformers.
"""
from __future__ import annotations

import gc
import os
from pathlib import Path
import time

from .protocol import INSTRUCTIONS, retrieval_pairs

HOBSON_REVISION = "bb282d786bc251fd4e3068de3ada9ddbb38127cd"
MINILM_REVISION = "233902d25c440f23af6f7d6e94d2946bac0bee0a"
BASE_REVISION = "b1485b2fa6dfa1287294f269f5fb618e03d52d7c"
UPSTREAM_REVISION = "930e6d5de16309be32579f5631a7369a3322140c"
HOBSON_TEMPERATURE = .9107136998460428
PINS = {
    "hobson/head.safetensors": "daad0727152b6185447cee36f78230c144312feb7b19f02749b19645238b5287",
    "hobson/lora/adapter_model.safetensors": "701bdb895887097f7954ec7eb06f7937d3b790035b5462195eb26abd80a4aebc",
    "hobson/hobson_config.json": "2ae86f2ed56975f68e8f2f368104df8ce0ff4b7d9d146dcaf8eec5626d62449d",
    "hobson/lora/adapter_config.json": "eb48e4ff81569664c4dd2a504b53da598eeaa390c265c2269ce4fe7526eab38a",
    "hobson/tokenizer.json": "a2cdd2e108566b09079afa8d266e9e65e7c280f27b771218237a06de5ba9cd86",
    "hobson/tokenizer_config.json": "8671bed7c852ce9e661be94f179a7b4ffd091c2a65aea0363e5501c20318ee45",
    "hobson/chat_template.jinja": "273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80",
    "minilm/model.safetensors": "821d1aa69520101d6e0737f78a042ae25b19e5cb9160701909d10434f4aeb0ae",
    "minilm/config.json": "380e02c93f431831be65d99a4e7e5f67c133985bf2e77d9d4eba46847190bacc",
    "minilm/tokenizer.json": "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66",
    "minilm/tokenizer_config.json": "a5c2e5a7b1a29a0702cd28c08a399b5ecc110c263009d17f7e3b415f25905fd8",
    "minilm/special_tokens_map.json": "3c3507f36dff57bce437223db3b3081d1e2b52ec3e56ee55438193ecb2c94dd6",
    "minilm/vocab.txt": "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3",
    "base/model.safetensors-00001-of-00001.safetensors": "928acbf11878c32185bbd863514d191769285065ab9ea14fbfe431303f5fdf2d",
    "base/config.json": "ed1c1723241f23f7f4e23430759cbd7dcfb4103cbdfe052bfe7626b57c2615b4",
    "base/model.safetensors.index.json": "74d2ddfe79f10f35b27b498632f02b97b60dd9ec39b35d7c5a890c399284e319",
    "upstream/modeling.py": "5e1dd53c6d940a0c238d6651fcf0b2a23bcacbfcbfdda851fe5945ce9a29d511",
    "upstream/infer.py": "c9edda18a9e3ce53fadf1f9ca4e1798314ed19b89040a1fc03c5529292551489",
    "upstream/prompting.py": "10207d68ebee901a31c19515a6efcb8c3d41856ade7fc492b44dc0095c910fe1",
    "upstream/schema.py": "2e10a5ead27627356dde1c2ec985f795a72558da4ab23876a2bc74a8407123fb",
}


def offline_environment():
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        os.environ[key] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "4"


class MiniLMRanker:
    def __init__(self, path):
        offline_environment()
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        torch.set_num_threads(4)
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(str(path),
            local_files_only=True, use_safetensors=True).cpu().eval()
        if self.model.config.num_labels != 1 or self.model.config.architectures != ["BertForSequenceClassification"]:
            raise ValueError("The partner requires the one-score MiniLM cross-encoder")
        self.model.requires_grad_(False)

    def rank(self, public):
        started = time.perf_counter()
        pairs = retrieval_pairs(public)
        batch = self.tokenizer([p[0] for p in pairs], [p[1] for p in pairs],
            padding=True, truncation=True, max_length=512, return_tensors="pt")
        with self.torch.inference_mode():
            scores = self.model(**batch).logits.view(-1).float().tolist()
        return {"scores": scores, "pair_count": len(pairs),
            "input_tokens": int(batch["attention_mask"].sum().item()),
            "padded_tokens": int(batch["input_ids"].numel()),
            "seconds": time.perf_counter() - started, "physical_forwards": 1}

    def close(self):
        self.model = None
        gc.collect()


class OfficialDecider:
    def __init__(self, checkpoint, base, *, kind="hobson"):
        offline_environment()
        import torch
        from transformers import AutoTokenizer
        from peft import PeftModel
        from strands_decider.modeling import (StrandsDeciderConfig, StrandsDeciderModel,
            config_path, load_head_state, ensure_bos, serving_attn_implementation)
        from strands_decider.infer import EngineConfig, SystemOneEngine
        from strands_decider.prompting import render_question
        from strands_decider.schema import NoulQuestion

        torch.set_num_threads(4)
        self.torch, self.engine, self.model = torch, None, None
        checkpoint = Path(checkpoint)
        config = StrandsDeciderConfig.from_json(config_path(str(checkpoint)))
        if kind == "hobson" and (config.pointer_dim != 256 or config.lora_r != 16
                or config.temperature_by_kind.get("noul") != HOBSON_TEMPERATURE):
            raise ValueError("The literal partner Hobson configuration changed")
        if config.base_model != "Qwen/Qwen3.5-2B-Base":
            raise ValueError("This bounded pilot reuses only the verified Qwen3.5 base")
        self.temperature = float(config.temperature_by_kind.get("noul", config.temperature))
        # Change location only in memory. All original checkpoint bytes stay frozen.
        config.base_model, config.base_revision = str(base), None
        tokenizer = AutoTokenizer.from_pretrained(str(checkpoint), local_files_only=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if config.force_bos:
            ensure_bos(tokenizer)
        torso = StrandsDeciderModel._load_torso(config, None, None)
        attention = serving_attn_implementation(torso.config)
        if attention:
            torso.set_attn_implementation(attention)
        torso = PeftModel.from_pretrained(torso, str(checkpoint / "lora"),
                                         is_trainable=False, local_files_only=True)
        model = StrandsDeciderModel(config, torso, tokenizer)
        model.head.load_state_dict(load_head_state(str(checkpoint)), strict=True)
        model.head.float()
        model.requires_grad_(False)
        model.eval()
        self.model = model
        self.rendered = render_question(NoulQuestion(instructions=INSTRUCTIONS))
        self.metadata = {"kind": kind, "pointer_dim": config.pointer_dim, "lora_rank": config.lora_r,
            "temperature": self.temperature, "head_parameters": sum(p.numel() for p in model.head.parameters()),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "context_limit": config.max_length, "prefix_cache": False,
            "precision": "bfloat16_torso_float32_pointer", "api_probability_rounding_decimals": 4}
        self.engine_config, self.engine_class = EngineConfig, SystemOneEngine

    def activate(self):
        # Startup placement precedes the caller's per-call post-load RAM/VRAM guard.
        self.engine = self.engine_class(self.model, self.engine_config(device="cuda",
            strict_window=True, use_prefix_cache=False, max_batch=1))
        self.torch.cuda.synchronize()

    def prepare(self, state):
        from strands_decider.prompting import render_state
        state_ids, question_ids, offsets = self.engine._fit(render_state(state), [self.rendered.text])
        return state_ids, question_ids, offsets

    def score(self, prepared):
        state_ids, question_ids, offsets = prepared
        torch, engine = self.torch, self.engine
        started = time.perf_counter()
        ids, mask = engine._pad([state_ids + question_ids[0]])
        options = engine._option_idx([self.rendered], len(state_ids), offsets)
        with torch.inference_mode():
            output = self.model(input_ids=ids, attention_mask=mask,
                n_slots=torch.tensor([2], device="cuda"), temperature=1.0, opt_idx=options)
            logits = output["logits"][0, :2].float()
            probability = float(torch.softmax(logits / self.temperature, dim=-1)[1].item())
            raw = logits.cpu().tolist()
        torch.cuda.synchronize()
        return {"p_true": round(probability, 4), "p_true_unrounded": probability,
            "raw_logits_false_true": raw, "temperature": self.temperature,
            "input_tokens": int(mask.sum().item()), "seconds": time.perf_counter() - started,
            "physical_forwards": 1, "head_scores": 1}

    def close(self):
        self.torch.cuda.synchronize()
        self.engine = None
        self.model = None
        gc.collect()
        self.torch.cuda.empty_cache()
        self.torch.cuda.synchronize()
        if self.torch.cuda.memory_allocated() != 0:
            raise RuntimeError("Owned native tensors remain allocated after model release")
