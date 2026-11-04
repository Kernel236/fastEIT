"""Tests for reconstruction.build_dataset - the ``ids`` step (one folder per patient)."""

import numpy as np
import pytest

from fasteit.reconstruction import build_dataset


@pytest.fixture()
def fake_pairs(monkeypatch):
    """load_paired returns 100 frames per pair; a file named 'bad' has too many frames."""

    def load_paired(eit_path, bin_path, input_mode="vv", max_frame_diff=1):
        if "bad" in str(eit_path):
            raise ValueError("Frame count mismatch: .eit has 120, .bin has 100.")
        x = np.ones((100, 208 if input_mode == "vv" else 416))
        return x, np.ones((100, 1024))

    monkeypatch.setattr(build_dataset, "load_paired", load_paired)


def make_tree(root, folders):
    for folder, files in folders.items():
        (root / folder).mkdir(parents=True)
        for name in files:
            (root / folder / name).touch()
    return root


def test_one_npz_per_id_folder_with_offset_ids(tmp_path, fake_pairs):
    source = make_tree(
        tmp_path / "src", {"ID3_abc": ["r1.eit", "r1.bin", "r2.eit", "r2.bin"]}
    )
    table = build_dataset.build_from_id_folders(
        source, tmp_path / "out", max_frames=150
    )
    with np.load(tmp_path / "out" / "patient_103.npz") as data:
        assert int(data["patient_id"]) == 103
        assert set(np.unique(data["rec_id"])) == {0, 1}
        assert len(data["Y"]) == 150  # sampled
    assert list(table.status) == ["used", "used"]
    assert set(table.id) == {"ID3"}


def test_missing_bin_and_mismatch_are_logged_not_fatal(tmp_path, fake_pairs):
    source = make_tree(
        tmp_path / "src",
        {"ID4_xyz": ["a.eit", "b.eit", "b.bin", "bad.eit", "bad.bin"]},
    )
    table = build_dataset.build_from_id_folders(source, tmp_path / "out")
    assert dict(zip(table.recording, table.status, strict=True)) == {
        0: "no_bin",
        1: "used",
        2: "frame_mismatch",
    }
    assert (tmp_path / "out" / "recordings.csv").exists()


def test_folders_without_id_are_ignored(tmp_path, fake_pairs):
    source = make_tree(
        tmp_path / "src", {"unworked files": ["r1.eit", "r1.bin"], "ID1_a": []}
    )
    table = build_dataset.build_from_id_folders(source, tmp_path / "out")
    assert table.empty
    assert not list((tmp_path / "out").glob("*.npz"))
