"""Bound prefill computation without shortening the writer's full context."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from haltiq_pair.server import (ATTENTION_IMPLEMENTATION, LOGITS_TO_KEEP,
                                MODEL, PREFILL_CHUNK_SIZE, REVISION, Writer)


class WriterMemoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
            import transformers
        except ImportError:
            raise unittest.SkipTest("optional writer model dependencies")

    def test_loader_and_receipt_pin_the_memory_configuration(self):
        import torch
        import transformers
        from types import SimpleNamespace
        model = unittest.mock.Mock()
        model.to.return_value = model
        model.eval.return_value = model
        model.config = SimpleNamespace(_attn_implementation="sdpa")
        with tempfile.TemporaryDirectory() as directory, \
                patch("transformers.AutoTokenizer.from_pretrained"), \
                patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=model) as loader:
            writer = Writer(directory, device="cpu")
            loader.assert_called_once_with(MODEL, revision=REVISION,
                                           torch_dtype=torch.float32,
                                           attn_implementation=ATTENTION_IMPLEMENTATION)
            receipt = json.loads((Path(directory) / "writer.json").read_text())
            self.assertEqual(receipt, writer.metadata)
            self.assertEqual(receipt["max_context"], 16384)
            self.assertEqual(receipt["context_policy"], "reject_overflow_without_truncation")
            self.assertEqual(receipt["attention_implementation"], "sdpa")
            self.assertEqual(receipt["prefill_chunk_size"], 512)
            self.assertEqual(receipt["logits_to_keep"], 1)
            self.assertTrue(receipt["use_cache"])
            self.assertEqual(receipt["transformers"], transformers.__version__)

    def test_complete_passes_all_tokens_and_explicit_generate_options(self):
        import torch
        from transformers import BatchEncoding
        captured = []
        prompt = torch.arange(1037).unsqueeze(0)

        class Tokenizer:
            eos_token_id = 0
            def apply_chat_template(self, messages, **kwargs):
                self.template_kwargs = kwargs
                return BatchEncoding({"input_ids": prompt.clone(),
                                      "attention_mask": torch.ones_like(prompt)})
            def decode(self, ids, **kwargs):
                return "fixture response"

        class Model:
            def generate(self, **kwargs):
                captured.append(kwargs)
                return torch.cat((kwargs["input_ids"], torch.tensor([[2, 3]])), dim=-1)

        with tempfile.TemporaryDirectory() as directory:
            writer = Writer.__new__(Writer)
            writer.output, writer.max_context = Path(directory), 16384
            writer.device, writer.tokenizer, writer.model = "cpu", Tokenizer(), Model()
            reply = writer.complete({"messages": [{"role": "user", "content": "fixture"}],
                                     "max_tokens": 12})
            self.assertEqual(len(captured), 1)
            args = captured[0]
            torch.testing.assert_close(args["input_ids"], prompt)
            self.assertEqual(args["prefill_chunk_size"], PREFILL_CHUNK_SIZE)
            self.assertEqual(args["logits_to_keep"], LOGITS_TO_KEEP)
            self.assertTrue(args["use_cache"])
            self.assertFalse(args["do_sample"])
            self.assertEqual(args["max_new_tokens"], 12)
            self.assertNotIn("truncation", writer.tokenizer.template_kwargs)
            self.assertEqual(reply["usage"]["prompt_tokens"], 1037)
            self.assertEqual(reply["usage"]["completion_tokens"], 2)
            writer.max_context = 1040
            with self.assertRaisesRegex(ValueError, "no silent truncation"):
                writer.complete({"messages": [{"role": "user", "content": "fixture"}],
                                 "max_tokens": 12})
            self.assertEqual(len(captured), 1)

    def test_real_qwen2_chunked_prefill_preserves_prompt_and_greedy_tokens(self):
        import torch
        import transformers
        from transformers import Qwen2Config, Qwen2ForCausalLM
        self.assertEqual(transformers.__version__, "5.17.0",
                         "run this qualification with the pinned Colab Transformers version")
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            torch.manual_seed(42)
            config = Qwen2Config(vocab_size=97, hidden_size=48, intermediate_size=96,
                                 num_hidden_layers=2, num_attention_heads=6,
                                 num_key_value_heads=2, max_position_embeddings=2048,
                                 eos_token_id=None, pad_token_id=0,
                                 attn_implementation="sdpa")
            model = Qwen2ForCausalLM(config).cpu().eval()
            prompt = torch.randint(1, 97, (1, 1037))
            kwargs = {"input_ids": prompt, "attention_mask": torch.ones_like(prompt),
                      "max_new_tokens": 6, "do_sample": False, "use_cache": True,
                      "logits_to_keep": 1, "return_dict_in_generate": True,
                      "output_logits": True}
            with torch.inference_mode():
                full = model.generate(**kwargs)
            observed = []
            def capture(module, args, call_kwargs):
                observed.append(call_kwargs["input_ids"].detach().clone())
            handle = model.register_forward_pre_hook(capture, with_kwargs=True)
            try:
                with torch.inference_mode():
                    chunked = model.generate(**kwargs, prefill_chunk_size=PREFILL_CHUNK_SIZE)
            finally:
                handle.remove()
            self.assertEqual([part.shape[-1] for part in observed[:3]], [512, 512, 13])
            torch.testing.assert_close(torch.cat(observed[:3], dim=-1), prompt)
            torch.testing.assert_close(chunked.sequences[:, :1037], prompt)
            torch.testing.assert_close(chunked.sequences, full.sequences)
            self.assertEqual(chunked.sequences.shape[-1], 1043)
            for full_logits, chunked_logits in zip(full.logits, chunked.logits):
                torch.testing.assert_close(chunked_logits, full_logits, atol=1e-6, rtol=1e-5)
        finally:
            torch.set_num_threads(previous_threads)


if __name__ == "__main__":
    unittest.main()
