from PIL import Image

def resize_image(input_path, output_path, size=(512, 512)):
    # Open an image file
    with Image.open(input_path) as img:
        # Resize the image
        img_resized = img.resize(size, Image.ANTIALIAS)
        
        # Save the resized image
        img_resized.save(output_path)
        print(f"Image saved to {output_path}")

# Example usage
resize_image('./test_images/image.png', './test_images/fish_canny.png')

