from PIL import Image, ImageFilter

def create_blurred_image(input_path, output_path, blur_radius=5):
    """
    Applies a Gaussian blur to the specified image.

    Parameters:
        input_path (str): Path to the input image file.
        output_path (str): Path where the blurred image will be saved.
        blur_radius (int): The radius of the Gaussian blur to apply.
    """
    try:
        # Open the original image
        image = Image.open(input_path)
        
        # Apply Gaussian blur
        blurred_image = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        
        # Save the blurred image
        blurred_image.save(output_path)
        print(f"Blurred image saved as {output_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
input_image_path = '0.png'  # Update this path to your image
output_image_path = 'unconditional.png'      # Desired path for the blurred image
create_blurred_image(input_image_path, output_image_path)

