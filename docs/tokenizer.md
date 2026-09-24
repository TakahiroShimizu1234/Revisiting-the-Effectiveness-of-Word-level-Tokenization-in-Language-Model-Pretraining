# Tokenizer Implementation Notes

The experiments compare word-level tokenization with UTF-8 byte fallback against SentencePiece Unigram tokenization.

The word-level segmentation is language-specific:

- English: rule-based segmentation
- French: language-aware segmentation
- Chinese: jieba
- Japanese: MeCab / fugashi

The tokenizer assets for the 10k, 50k, and 100k vocabulary settings are provided in `tokenizer_assets/`.

Word-level out-of-vocabulary units are represented with UTF-8 byte fallback tokens.

For the verified Llama 3.2 1B English checkpoints, use `mp_en_paper_10k`, `mp_en_paper_50k`, and `mp_en_paper_100k`. Their implementation is `tokenization_mp_tokenizer.py`; load each directory with `AutoTokenizer.from_pretrained(..., trust_remote_code=True)`.

English segmentation uses alphabetic runs, single digits, single punctuation/symbol characters, and one `_` per whitespace character. Vocabulary lookup uses lowercase; UTF-8 byte fallback uses the original unmatched unit.


The multilingual token-budget analysis script defaults to the `mp_en_paper_*` assets for English. Its English vocabulary-specific results should not be assumed to represent every other model family until those models' tokenizer mappings have been checked.

For French, Chinese, and Japanese, see their language-specific `mp_tokenizer.py` implementations.
