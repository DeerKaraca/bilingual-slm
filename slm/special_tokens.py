"""Special tokens shared by the tokenizer, the data pipeline and the server (no dependencies, so the
server image does not need the data stack)."""

EOT = "<|endoftext|>"  # document separator
SPECIAL_TOKENS = [EOT, "<|im_start|>", "<|im_end|>"]  # ChatML tokens, reserved for chat models
