from datetime import datetime
import shutil
import uuid
import cv2
import os


def find_subphotos_and_save(
    input_image_path: str,
    output_dir: str,
    processed_dir: str,
    min_contour_area: float
):
    """
    Reads an input scanned image containing multiple sub-photos,
    detects each sub-photo, crops, deskews it, and saves each
    sub-photo to the specified output directory.

    :param input_image_path: Path to the input scanned image.
    :param output_dir: Directory where cropped sub-photos are saved.
    :param min_contour_area: Minimum area a contour must have to be
                             considered a valid sub-photo.
    """

    # 1. Read the image
    image = cv2.imread(input_image_path)

    if image is None:
        print(f"Could not read image at {input_image_path}. Skipping.")
        return

    # 2. Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 3. Threshold (inverse binary if photos are on white background)
    # Adjust the threshold value or method as needed
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

    # 4. Find contours
    contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    file_name = os.path.splitext(os.path.basename(input_image_path))[0]

    # 5. Loop through contours, filter, deskew, and save each sub-photo
    for contour in contours:
        area = cv2.contourArea(contour)

        if area < min_contour_area:
            # Ignore noise or very small contours
            continue

        # 5a. Compute minimum area rectangle
        # This gives us the center, size, and rotation of the sub-photo
        rect = cv2.minAreaRect(contour)
        (cx, cy), (w, h), angle = rect

        # 5b. Deskew
        #  Create rotation matrix around the rectangle's center
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)

        #  Rotate the entire image
        rotated = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

        # 5c. After rotation, extract the bounding box of the sub-photo
        #     Because we've rotated, we can now treat it like an upright 
        # boundingRect
        w, h = int(w), int(h)
        x = int(cx - w / 2)
        y = int(cy - h / 2)

        # Make sure we don’t go out of image boundaries
        w = min(rotated.shape[1] - x, w)
        h = min(rotated.shape[0] - y, h)

        cropped = rotated[y: y + h, x: x + w]

        if w < h or angle > 45.0:
            cropped = cv2.rotate(cropped, cv2.ROTATE_90_CLOCKWISE)

        date_now = datetime.now().strftime("%Y-%m-%d-%H_%M_%S")
        subphoto_id = f"{date_now}_{uuid.uuid4()}"

        # 5d. Save each sub-photo
        subphoto_path = f"{output_dir}/{file_name}_{subphoto_id}.jpg"
        cv2.imwrite(subphoto_path, cropped)
        print(f"Saved sub-photo: {subphoto_path}")

    shutil.move(
        input_image_path, processed_dir + "/" +
        os.path.basename(input_image_path)
    )
    print(f"File moved to {processed_dir}.")


def check_and_create_folder(folder: str):
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(f"Folder '{folder}' created.")


if __name__ == "__main__":
    import glob

    # Example usage:
    input_folder = "./scan"
    output_folder = "./output"
    processed_folder = "./processed"

    check_and_create_folder(input_folder)
    check_and_create_folder(output_folder)
    check_and_create_folder(processed_folder)

    # Get all image files from input_folder
    image_files = glob.glob(input_folder + "/*.jpeg")  # or *.png, *.tif, etc.

    for img_path in image_files:
        find_subphotos_and_save(
            input_image_path=img_path,
            output_dir=output_folder,
            processed_dir=processed_folder,
            min_contour_area=80000,  # Adjust for your case
        )
