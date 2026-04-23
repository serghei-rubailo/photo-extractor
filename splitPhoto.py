import glob
import os
import shutil
import uuid
from datetime import datetime

import cv2


def find_subphotos_and_save(
    input_image_path: str,
    output_dir: str,
    processed_dir: str,
    min_contour_area: float,
) -> int:
    """
    Reads an input scanned image containing multiple sub-photos,
    detects each sub-photo, crops, deskews it, and saves each
    sub-photo to the specified output directory.

    :param input_image_path: Path to the input scanned image.
    :param output_dir: Directory where cropped sub-photos are saved.
    :param processed_dir: Directory where processed originals are moved.
    :param min_contour_area: Minimum area a contour must have to be
                             considered a valid sub-photo.
    :return: Number of sub-photos saved.
    """

    image = cv2.imread(input_image_path)
    if image is None:
        print(f"Could not read image at {input_image_path}. Skipping.")
        return 0

    image_height, image_width = image.shape[:2]

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    file_name = os.path.splitext(os.path.basename(input_image_path))[0]
    date_now = datetime.now().strftime("%Y-%m-%d-%H_%M_%S")

    print(f"Start working on: {file_name}")

    saved_count = 0

    for contour in contours:
        area = cv2.contourArea(contour)

        if area < min_contour_area:
            continue

        rect = cv2.minAreaRect(contour)
        (cx, cy), (w, h), angle = rect

        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        rotated = cv2.warpAffine(
            image, M, (image_width, image_height),
            borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255),
        )

        w, h = int(w), int(h)
        x = max(0, int(cx - w / 2))
        y = max(0, int(cy - h / 2))
        x_end = min(image_width, x + w)
        y_end = min(image_height, y + h)

        cropped = rotated[y:y_end, x:x_end]

        if cropped.size == 0:
            print("Warning: empty crop for contour, skipping.")
            continue

        subphoto_id = f"{date_now}_{uuid.uuid4()}"
        subphoto_path = os.path.join(output_dir, f"{file_name}_{subphoto_id}.jpg")

        if cv2.imwrite(subphoto_path, cropped):
            saved_count += 1
            print(f"Saved sub-photo: {subphoto_path}")
        else:
            print(f"ERROR: Failed to write sub-photo: {subphoto_path}")

    if saved_count == 0:
        print(f"WARNING: No sub-photos detected in {file_name}. File NOT moved.")
        return 0

    dest = os.path.join(processed_dir, f"{file_name}_{date_now}_{uuid.uuid4()}.jpg")
    shutil.move(input_image_path, dest)
    print(f"File moved to {processed_dir}.")

    return saved_count


def check_and_create_folder(folder: str):
    os.makedirs(folder, exist_ok=True)


if __name__ == "__main__":
    input_folder = "./scan"
    output_folder = "./output"
    processed_folder = "./processed"

    check_and_create_folder(input_folder)
    check_and_create_folder(output_folder)
    check_and_create_folder(processed_folder)

    extensions = (
        "*.jpg", "*.jpeg", "*.JPG", "*.JPEG",
        "*.png", "*.PNG", "*.tiff", "*.tif", "*.bmp",
    )
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(input_folder, ext)))

    for img_path in image_files:
        find_subphotos_and_save(
            input_image_path=img_path,
            output_dir=output_folder,
            processed_dir=processed_folder,
            min_contour_area=500_000,
        )
