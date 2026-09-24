import sys

from lm_eval.models.huggingface import HFLM


if not getattr(HFLM, "_acl_lowercase_patch_applied", False):
    _original_tok_encode = HFLM.tok_encode
    _original_tok_batch_encode = HFLM.tok_batch_encode
    _sample_printed = False

    def _report_change(before, after):
        global _sample_printed

        if not _sample_printed and before != after:
            print(
                "[lowercase-eval] sample:",
                repr(before[:120]),
                "->",
                repr(after[:120]),
                file=sys.stderr,
                flush=True,
            )
            _sample_printed = True

    def tok_encode_lowercase(self, string, *args, **kwargs):
        if isinstance(string, str):
            lowered = string.lower()
            _report_change(string, lowered)
            string = lowered

        return _original_tok_encode(self, string, *args, **kwargs)

    def tok_batch_encode_lowercase(self, strings, *args, **kwargs):
        lowered_strings = []

        for string in strings:
            if isinstance(string, str):
                lowered = string.lower()
                _report_change(string, lowered)
                lowered_strings.append(lowered)
            else:
                lowered_strings.append(string)

        return _original_tok_batch_encode(
            self,
            lowered_strings,
            *args,
            **kwargs,
        )

    HFLM.tok_encode = tok_encode_lowercase
    HFLM.tok_batch_encode = tok_batch_encode_lowercase
    HFLM._acl_lowercase_patch_applied = True

    print(
        "[lowercase-eval] HFLM lowercase patch active.",
        file=sys.stderr,
        flush=True,
    )
