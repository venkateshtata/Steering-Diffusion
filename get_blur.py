from PIL import Image

def create_blank_image(width, height, output_path, color=(0, 0, 0)):
    """
    Creates a blank image of the specified dimensions and color.

    Parameters:
        width (int): Width of the image.
        height (int): Height of the image.
        output_path (str): Path where the blank image will be saved.
        color (tuple): Color of the image in RGB format (default is white).
    """
    try:
        # Create a new blank image
        image = Image.new('RGB', (width, height), color)
        
        # Save the image
        image.save(output_path)
        print(f"Blank image saved as {output_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
width = 512  # Width of the image
height = 512  # Height of the image
output_image_path = 'unconditional.png'  # Desired path for the blank image
create_blank_image(width, height, output_image_path)
