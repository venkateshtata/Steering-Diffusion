from PIL import Image

def load_image_details(image_path):
    # Load the image
    img = Image.open("/notebooks/erase/stable-diffusion/test_images/images_bird_canny.png")
    
    # Get image details
    print(f"Format: {img.format}")             # Output: Image Format (like PNG, JPEG)
    print(f"Mode: {img.mode}")                 # Output: Image Mode (like RGB, CMYK)
    print(f"Size: {img.size}")                 # Output: Image Size (Width, Height)
    print(f"Width: {img.width} pixels")        # Output: Image Width
    print(f"Height: {img.height} pixels")      # Output: Image Height

    # Close the image file
    img.close()

# Example usage
image_path = 'path/to/your/image.jpg'
load_image_details(image_path)

