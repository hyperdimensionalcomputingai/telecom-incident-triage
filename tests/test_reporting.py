"""Report templates preserve mathematical notation and fail on missing values."""

from dataclasses import replace

import pytest

import reporting
from config import Config


def test_named_replacement_preserves_math_and_inserted_content(tmp_path, monkeypatch):
    monkeypatch.setattr(reporting, "REPORT_TEMPLATES", tmp_path)
    (tmp_path / "example.md").write_text(
        "$$h_{\\text{episode}}=\\frac{x}{y}$$\n"
        "{{ table }}\n{{ count }} incidents; {{count}} again.\n"
        "![Figure](figures/example.png)\n"
    )
    table = "| Value |\n|---|\n| {{ untouched }} \\otimes |"
    rendered = reporting.render_template("example.md", {"table": table, "count": 3})
    assert rendered == (
        "$$h_{\\text{episode}}=\\frac{x}{y}$$\n"
        + table
        + "\n3 incidents; 3 again.\n![Figure](figures/example.png)\n"
    )


def test_missing_report_value_fails_before_rendering(tmp_path, monkeypatch):
    monkeypatch.setattr(reporting, "REPORT_TEMPLATES", tmp_path)
    (tmp_path / "example.md").write_text("{{ known }} {{ missing }}")
    with pytest.raises(ValueError, match="Missing report values in example.md: missing"):
        reporting.render_template("example.md", {"known": "value"})


def test_encoder_template_uses_config_without_damaging_latex():
    config = replace(Config(), dimension=8192, levels=64, context_weight=3.0)
    rendered = reporting.encoder_explanation(config)
    assert "8,192-dimensional hypervector" in rendered
    assert "one of 64 numeric levels" in rendered
    assert "(w_S,w_C,w_H)=(1.0,3.0,0.25)" in rendered
    assert r"\frac{x-a_f}{b_f-a_f}" in rendered
    assert not reporting.PLACEHOLDER.search(rendered)
