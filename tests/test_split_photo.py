import os

import cv2
import numpy as np

from splitPhoto import check_and_create_folder, find_subphotos_and_save

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _setup_dirs(tmp_path):
    """Create input/output/processed dirs under tmp_path and return paths."""
    input_dir = str(tmp_path / "scan")
    output_dir = str(tmp_path / "output")
    processed_dir = str(tmp_path / "processed")
    for d in [input_dir, output_dir, processed_dir]:
        os.makedirs(d)
    return input_dir, output_dir, processed_dir


def _white_image(height, width):
    return np.ones((height, width, 3), dtype=np.uint8) * 255


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


class TestCheckAndCreateFolder:
    def test_creates_nested_directory(self, tmp_path):
        folder = str(tmp_path / "a" / "b" / "c")
        check_and_create_folder(folder)
        assert os.path.isdir(folder)

    def test_idempotent(self, tmp_path):
        folder = str(tmp_path / "folder")
        check_and_create_folder(folder)
        check_and_create_folder(folder)
        assert os.path.isdir(folder)


class TestUnreadableImage:
    def test_nonexistent_file_does_not_crash(self, tmp_path):
        _, output_dir, processed_dir = _setup_dirs(tmp_path)
        result = find_subphotos_and_save(
            str(tmp_path / "nonexistent.jpg"), output_dir, processed_dir, 500_000
        )
        assert result == 0
        assert os.listdir(output_dir) == []

    def test_corrupt_file_does_not_crash(self, tmp_path):
        input_dir, output_dir, processed_dir = _setup_dirs(tmp_path)
        corrupt_path = os.path.join(input_dir, "corrupt.jpg")
        with open(corrupt_path, "wb") as f:
            f.write(b"not an image")
        result = find_subphotos_and_save(
            corrupt_path, output_dir, processed_dir, 500_000
        )
        assert result == 0


class TestBlankImage:
    def test_white_image_produces_no_output(self, tmp_path):
        input_dir, output_dir, processed_dir = _setup_dirs(tmp_path)
        img = _white_image(500, 500)
        img_path = os.path.join(input_dir, "blank.jpg")
        cv2.imwrite(img_path, img)

        result = find_subphotos_and_save(img_path, output_dir, processed_dir, 500_000)
        assert result == 0
        assert os.listdir(output_dir) == []
        # Original should NOT be moved when zero sub-photos found
        assert os.path.exists(img_path)


class TestSmallContoursFiltered:
    def test_tiny_marks_below_threshold_ignored(self, tmp_path):
        input_dir, output_dir, processed_dir = _setup_dirs(tmp_path)
        img = _white_image(500, 500)
        # Draw a small 20x20 dark square (area=400, below 500_000)
        cv2.rectangle(img, (100, 100), (120, 120), (30, 30, 30), -1)
        img_path = os.path.join(input_dir, "noise.jpg")
        cv2.imwrite(img_path, img)

        result = find_subphotos_and_save(img_path, output_dir, processed_dir, 500_000)
        assert result == 0
        assert os.listdir(output_dir) == []


class TestOutputFilenames:
    def test_filenames_contain_source_name(self, tmp_path):
        input_dir, output_dir, processed_dir = _setup_dirs(tmp_path)
        img = _white_image(2000, 2000)
        cv2.rectangle(img, (100, 100), (1500, 1500), (30, 30, 30), -1)
        img_path = os.path.join(input_dir, "my_scan_001.jpg")
        cv2.imwrite(img_path, img)

        find_subphotos_and_save(img_path, output_dir, processed_dir, 500_000)
        for fname in os.listdir(output_dir):
            assert "my_scan_001" in fname


# ---------------------------------------------------------------------------
# Integration test
# ---------------------------------------------------------------------------


def _create_synthetic_scan():
    """
    Create a 3000x4000 white image with 3 dark rectangles simulating
    photos on a scanner bed.

    Background: 255 (white, above threshold 200)
    Rectangles: 30-50 (dark, below threshold 200)
    """
    img = _white_image(3000, 4000)

    # Rect 1: axis-aligned, 1000x800 = 800k area
    cv2.rectangle(img, (200, 200), (1200, 1000), (30, 30, 30), -1)

    # Rect 2: rotated -10 degrees, ~800k area
    box2 = cv2.boxPoints(((2800, 600), (1000, 800), -10))
    cv2.drawContours(img, [np.intp(box2)], 0, (40, 40, 40), -1)

    # Rect 3: rotated -25 degrees, ~630k area
    box3 = cv2.boxPoints(((2000, 2000), (900, 700), -25))
    cv2.drawContours(img, [np.intp(box3)], 0, (50, 50, 50), -1)

    return img


class TestFullPipeline:
    def test_three_photos_detected_and_saved(self, tmp_path):
        input_dir, output_dir, processed_dir = _setup_dirs(tmp_path)

        img = _create_synthetic_scan()
        img_path = os.path.join(input_dir, "test_scan.jpg")
        cv2.imwrite(img_path, img)

        result = find_subphotos_and_save(img_path, output_dir, processed_dir, 500_000)

        # Exactly 3 sub-photos saved
        output_files = os.listdir(output_dir)
        assert result == 3, f"Expected 3 sub-photos, got {result}"
        assert len(output_files) == 3

        # Each output is a valid, non-empty image
        for fname in output_files:
            fpath = os.path.join(output_dir, fname)
            sub = cv2.imread(fpath)
            assert sub is not None, f"{fname} is not a valid image"
            assert sub.size > 0, f"{fname} is empty"
            assert sub.shape[0] > 50 and sub.shape[1] > 50, (
                f"{fname} too small: {sub.shape}"
            )

        # Original moved to processed
        assert not os.path.exists(img_path)
        processed_files = os.listdir(processed_dir)
        assert len(processed_files) == 1


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_contour_near_image_edge_no_crash(self, tmp_path):
        """Rectangle near the image boundary should not crash."""
        input_dir, output_dir, processed_dir = _setup_dirs(tmp_path)
        img = _white_image(500, 500)
        # Draw a dark rectangle touching the top-left corner
        cv2.rectangle(img, (0, 0), (300, 250), (30, 30, 30), -1)
        img_path = os.path.join(input_dir, "edge.jpg")
        cv2.imwrite(img_path, img)

        # Should not raise even with coordinates near zero
        result = find_subphotos_and_save(img_path, output_dir, processed_dir, 1000)
        assert result >= 0  # no crash is the main assertion
