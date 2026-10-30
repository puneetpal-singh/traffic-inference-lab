from traffic_lab.provenance import file_hash


def test_file_hash_matches_known_sha256(tmp_path):
    path = tmp_path / "weights.bin"
    path.write_bytes(b"abc")
    assert file_hash(path) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_artifact_identity_is_independent_of_workspace_root(tmp_path):
    first, second = tmp_path / "machine-a", tmp_path / "machine-b"
    first.mkdir()
    second.mkdir()
    for directory in (first, second):
        (directory / "weights.bin").write_bytes(b"weights")
        (directory / "metadata.json").write_text("{}")
    assert file_hash(first) == file_hash(second)


def test_artifact_identity_detects_content_and_filename_changes(tmp_path):
    path = tmp_path / "weights.bin"
    path.write_bytes(b"original")
    original = file_hash(tmp_path)
    path.write_bytes(b"changed")
    changed = file_hash(tmp_path)
    assert changed != original
    path.rename(tmp_path / "other.bin")
    assert file_hash(tmp_path) != changed
