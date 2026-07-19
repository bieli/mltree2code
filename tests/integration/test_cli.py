from __future__ import annotations

from pathlib import Path

import joblib

from mltree2code.cli import main


def test_cli_stdout(tmp_path: Path, iris_model, capsys):
    model, _ = iris_model
    path = tmp_path / "m.joblib"
    joblib.dump(model, path)
    rc = main([str(path), "python"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "def predict(x):" in out


def test_cli_output_file(tmp_path: Path, iris_model):
    model, _ = iris_model
    path = tmp_path / "m.joblib"
    out = tmp_path / "tree.c"
    joblib.dump(model, path)
    rc = main([str(path), "c", "-o", str(out)])
    assert rc == 0
    text = out.read_text()
    assert "int predict" in text


def test_cli_flags(tmp_path: Path, iris_model, capsys):
    model, _ = iris_model
    path = tmp_path / "m.joblib"
    joblib.dump(model, path)
    rc = main(
        [
            "--model",
            str(path),
            "--lang",
            "rust",
            "--function-name",
            "score",
            "--no-comments",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "pub fn score" in out


def test_cli_list_languages(capsys):
    rc = main(["--list-languages"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "python" in out
    assert "rust" in out


def test_cli_missing_model(capsys):
    rc = main(["/no/such/model.joblib", "python"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "error:" in err
