from triton_nki_transpile.logger import trace_logger


def _list_response_files(path):
    return [p.name for p in path.iterdir() if p.name.endswith("_response.md")]


def test_log_response_saves_markdown_for_code_fence(tmp_path):
    trace_logger.set_log_dir(str(tmp_path))
    s = "```python\nprint('hi')\n```"
    trace_logger.log_response(s)

    files = _list_response_files(tmp_path)
    assert any(f.endswith("_response.md") for f in files)
    md_file = next(str(tmp_path / f) for f in files if f.endswith("_response.md"))
    with open(md_file) as fh:
        content = fh.read()
    assert content.strip() == s


def test_log_response_saves_markdown_for_json_and_plain_text(tmp_path):
    trace_logger.set_log_dir(str(tmp_path))

    json_input = '{"a": 1, "b": [1,2]}'
    txt_input = 'Just some plain text output'

    trace_logger.log_response(json_input)
    trace_logger.log_response(txt_input)

    files = _list_response_files(tmp_path)
    assert len(files) >= 2

    contents = [(f, (tmp_path / f).read_text()) for f in files]
    assert any(json_input in c for _, c in contents)
    assert any(txt_input in c for _, c in contents)

def test_log_version(tmp_path):
    trace_logger.set_log_dir(str(tmp_path))
    version = "1.2.3"
    trace_logger.log_version(version)

    version_file = tmp_path / "VERSION"
    assert version_file.exists()
    assert version_file.read_text() == version
