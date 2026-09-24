# Tokenizer Implementation Notes

The experiments compare word-level tokenization with UTF-8 byte fallback against SentencePiece Unigram tokenization.

The word-level segmentation is language-specific:

- English: rule-based segmentation
- French: language-aware segmentation
- Chinese: jieba
- Japanese: MeCab / fugashi

The tokenizer assets for the 10k, 50k, and 100k vocabulary settings are provided in `tokenizer_assets/`.

Word-level out-of-vocabulary units are represented with UTF-8 byte fallback tokens.

See the language-specific `mp_tokenizer.py` files for the implemented segmentation and normalization behavior.
