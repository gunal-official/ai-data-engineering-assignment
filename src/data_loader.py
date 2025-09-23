from pathlib import Path

def load_images_and_labels(dataset_dir):
    """
    Recursively load all images from category subfolders and
    create a list of file paths and corresponding category labels.
    
    Args:
        dataset_dir (str or Path): Root directory containing category subfolders.
        
    Returns:
        List of tuples: [(image_path, category_label), ...]
    """
    dataset_path = Path(dataset_dir)
    images_and_labels = []

    # Iterate over each category folder
    for category_folder in dataset_path.iterdir():
        if category_folder.is_dir():
            category_label = category_folder.name
            # Iterate over image files in this category folder
            for image_file in category_folder.glob("*.*"):  # Match all files with any extension
                # Append tuple (image_path, category_label)
                images_and_labels.append((str(image_file.resolve()), category_label))

    return images_and_labels


if __name__ == "__main__":
    dataset_directory = "data/sample_images"
    images_labels = load_images_and_labels(dataset_directory)

    print(f"Loaded {len(images_labels)} images from {dataset_directory}")
    for img_path, label in images_labels[:10]:
        print(f"Image: {img_path} | Category: {label}")
