
from splitPhoto import check_and_create_folder, find_subphotos_and_save


def test_imports():
    import cv2
    import numpy

    assert cv2.__version__
    assert numpy.__version__


def test_check_and_create_folder(tmp_path):
    folder = tmp_path / "a" / "b" / "c"
    check_and_create_folder(str(folder))
    assert folder.is_dir()

    # idempotent — no error on second call
    check_and_create_folder(str(folder))
    assert folder.is_dir()


def test_unreadable_image_does_not_crash(tmp_path):
    output_dir = tmp_path / "output"
    processed_dir = tmp_path / "processed"
    output_dir.mkdir()
    processed_dir.mkdir()

    # should not raise
    find_subphotos_and_save(
        input_image_path=str(tmp_path / "nonexistent.jpg"),
        output_dir=str(output_dir),
        processed_dir=str(processed_dir),
        min_contour_area=500_000,
    )

    # no output files created
    assert list(output_dir.iterdir()) == []
