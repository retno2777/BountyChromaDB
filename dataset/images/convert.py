from PIL import Image
import os
def convert_all_images_to_rgb(input_folder, output_folder):
    """
    Convert all images in the input folder to RGB format and save them to the output folder.
    
    :param input_folder: Path to the folder containing input images.
    :param output_folder: Path to the folder where converted RGB images will be saved.
    """
    # Ensure the output directory exists, if not, create it
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Define valid image extensions
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')

    # Iterate over each file in the input folder
    for file_name in os.listdir(input_folder):
        if file_name.lower().endswith(valid_extensions):
            # Full path to the input image
            input_image_path = os.path.join(input_folder, file_name)
            
            # Full path for the output image (in RGB format)
            output_image_path = os.path.join(output_folder, file_name)

            try:
                # Open the image
                img = Image.open(input_image_path)
                
                # Convert to RGB
                rgb_img = img.convert('RGB')
                
                # Save the converted image
                rgb_img.save(output_image_path)
                print(f"{file_name} successfully converted to RGB and saved.")
            except Exception as e:
                print(f"Error converting {file_name}: {e}")

# Contoh penggunaan
input_folder = 'path/to/input/folder'  
output_folder = 'path/to/output/folder' 

convert_all_images_to_rgb(input_folder, output_folder)
