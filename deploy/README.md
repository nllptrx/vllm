# Qwen GDN n-gram experiment

`docker-compose.ngram-spec.yml` is a reproducible experiment, not the production
configuration. It layers the GDN speculative-state recovery from upstream PR
#40738 over the APC deadlock-fix image.

The state fix prevents the known mixed prefill/decode corruption, including the
post-verification convolution and SSM state recovery covered by the added unit
test. Live extraction nevertheless rejected n-gram speculation for this
workload:

- WISP 21.05.25: 12.0 s with n-gram versus 10.5 s without it, with identical
  canonical prediction.
- BEX DDT 1443: a repeated `Reference` decode reached 16,384 tokens and 109 s;
  the n-gram proposer accepted the repeated suffix at 100% and accelerated the
  runaway rather than ending it.
- Server-side repetition detection reduced that BEX run to 52 s, but the
  perturbed retry omitted and misbound references, so it is not a quality-safe
  production mitigation.

The deployed runtime should remain the APC-only image until n-gram produces a
gold-equivalent speedup and a repetition stop can be retried without changing
field recall.
