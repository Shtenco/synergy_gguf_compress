from pathlib import Path

from gguf_compress.e2e_quality import _bounded_corpus, parse_perplexity_output


def test_parse_perplexity_and_kld_metrics():
    text = """
    final estimate: PPL = 6.407115 +/- 0.039119
    KLD = 0.031273 +/- 0.000238
    Pearson correlation = 0.98765
    Same top p = 91.25
    RMS probability change = 5.519
    """
    m = parse_perplexity_output(text)
    assert m["perplexity"] == 6.407115
    assert m["kl_divergence"] == 0.031273
    assert m["pearson"] == 0.98765
    assert m["same_top_p"] == 91.25
    assert m["rms_probability_change"] == 5.519


def test_parse_kullback_wording():
    text = "perplexity: 7.25\nKullback-Leibler divergence: 0.125\n"
    m = parse_perplexity_output(text)
    assert m["perplexity"] == 7.25
    assert m["kl_divergence"] == 0.125


def test_bounded_corpus_caps_bytes_and_keeps_utf8(tmp_path: Path):
    src = tmp_path / "corpus.txt"
    src.write_text("Привет мир! " * 1000, encoding="utf-8")
    kld = tmp_path / "baseline.kld"
    bounded, size, changed = _bounded_corpus(src, kld, 1024)
    assert changed is True
    assert bounded != src
    assert 0 < size <= 1024
    bounded.read_text(encoding="utf-8")


def test_small_corpus_is_used_directly(tmp_path: Path):
    src = tmp_path / "small.txt"
    src.write_text("small corpus", encoding="utf-8")
    bounded, size, changed = _bounded_corpus(src, tmp_path / "x.kld", 4096)
    assert bounded == src
    assert size == src.stat().st_size
    assert changed is False
