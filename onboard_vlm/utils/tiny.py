"""Tiny randomly-initialised Gemma 4 configs for tests and offline dry runs (no Hub access)."""

from __future__ import annotations

from typing import Any

TINY_SPECIAL_TOKENS = dict(
    boi_token_id=300,
    eoi_token_id=301,
    image_token_id=302,
    video_token_id=303,
    boa_token_id=304,
    eoa_token_index=305,
    audio_token_id=306,
)


def tiny_gemma4_config(vision_hidden: int = 48, vision_layers: int = 3, text_hidden: int = 64, vocab: int = 512, **overrides: Any):
    """A ~1M-parameter ``Gemma4Config`` that runs forward/generate on CPU in milliseconds."""
    from transformers import Gemma4Config

    cfg = dict(
        text_config=dict(
            vocab_size=vocab,
            vocab_size_per_layer_input=vocab,
            hidden_size=text_hidden,
            intermediate_size=2 * text_hidden,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=32,
            hidden_size_per_layer_input=8,
            sliding_window=64,
            layer_types=["sliding_attention", "full_attention"],
            pad_token_id=0,
            eos_token_id=1,
            bos_token_id=2,
        ),
        vision_config=dict(
            hidden_size=vision_hidden,
            intermediate_size=2 * vision_hidden,
            num_hidden_layers=vision_layers,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=vision_hidden // 2,
        ),
        audio_config=None,
        **TINY_SPECIAL_TOKENS,
    )
    cfg.update(overrides)
    return Gemma4Config(**cfg)


_TINY_CHAT_TEMPLATE = (
    "{{ bos_token }}{% for m in messages %}<start_of_turn>{{ 'model' if m['role'] == 'assistant' else m['role'] }}\n"
    "{% for c in m['content'] %}{% if c['type'] == 'image' %}<|image|>{% elif c['type'] == 'video' %}<|video|>"
    "{% else %}{{ c['text'] }}{% endif %}{% endfor %}<end_of_turn>\n{% endfor %}"
    "{% if add_generation_prompt %}<start_of_turn>model\n{% endif %}"
)


def tiny_gemma4_processor(vocab_words: str = "user model system the a of is what this in on to and answer question image video please yes no"):
    """An offline ``Gemma4Processor`` (word-level tokenizer) whose special-token ids match
    :func:`tiny_gemma4_config`. Exercises the real chat-template / image-token expansion code."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import Gemma4ImageProcessor, Gemma4Processor, PreTrainedTokenizerFast
    from transformers.models.gemma4.feature_extraction_gemma4 import Gemma4AudioFeatureExtractor
    from transformers.models.gemma4.video_processing_gemma4 import Gemma4VideoProcessor

    t = TINY_SPECIAL_TOKENS
    vocab = {
        "<pad>": 0, "<eos>": 1, "<bos>": 2, "<unk>": 3,
        "<boi>": t["boi_token_id"], "<eoi>": t["eoi_token_id"], "<|image|>": t["image_token_id"],
        "<|video|>": t["video_token_id"], "<boa>": t["boa_token_id"], "<eoa>": t["eoa_token_index"],
        "<|audio|>": t["audio_token_id"], "<start_of_turn>": 307, "<end_of_turn>": 308,
    }
    for i, w in enumerate(vocab_words.split()):
        vocab[w] = 10 + i
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    hf_tok = PreTrainedTokenizerFast(
        tokenizer_object=tok, bos_token="<bos>", eos_token="<eos>", pad_token="<pad>", unk_token="<unk>",
        extra_special_tokens={
            "image_token": "<|image|>", "boi_token": "<boi>", "eoi_token": "<eoi>",
            "audio_token": "<|audio|>", "boa_token": "<boa>", "eoa_token": "<eoa>",
        },
    )
    return Gemma4Processor(
        feature_extractor=Gemma4AudioFeatureExtractor(),
        image_processor=Gemma4ImageProcessor(),
        tokenizer=hf_tok,
        video_processor=Gemma4VideoProcessor(),
        chat_template=_TINY_CHAT_TEMPLATE,
    )
