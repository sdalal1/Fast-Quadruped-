import os
import numpy as np
import mujoco
import glfw
import PIL
from PIL import Image

class MuJoCoCollageGenerator:
    def __init__(self, model_directory, output_path='mujoco_model_collage.png', 
                 grid_layout=None, simulation_duration=2.0, render_width=400, render_height=400,
                 crop_percentage=0.5, padding=10):
        """
        Initialize the MuJoCo Model Collage Generator
        
        :param model_directory: Path to directory containing .xml or .mjc model files
        :param output_path: Path to save the collage image
        :param grid_layout: Optional tuple (rows, cols) to specify grid layout
        :param simulation_duration: Duration (in seconds) to simulate each model
        :param render_width: Width of each model render
        :param render_height: Height of each model render
        :param crop_percentage: Percentage of image to keep from center (0.0 to 1.0)
        :param padding: Pixel width of white space between models
        """
        self.model_directory = model_directory
        self.output_path = output_path
        self.simulation_duration = simulation_duration
        self.render_width = render_width
        self.render_height = render_height
        self.crop_percentage = crop_percentage
        self.padding = padding
        
        # Find all model files
        self.model_files = [
            f for f in os.listdir(model_directory) 
            if f.endswith(('.xml', '.mjc'))
        ]
        
        # Determine grid layout
        if grid_layout is None:
            import math
            total_models = len(self.model_files)
            grid_cols = math.ceil(math.sqrt(total_models))
            grid_rows = math.ceil(total_models / grid_cols)
            self.grid_layout = (grid_rows, grid_cols)
        else:
            self.grid_layout = grid_layout

    def render_model(self, model_path):
        """
        Render a single MuJoCo model
        
        :param model_path: Full path to the model file
        :return: PIL Image of the model simulation
        """
        # Initialize GLFW
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")
        
        # Create window
        glfw.window_hint(glfw.VISIBLE, False)
        window = glfw.create_window(self.render_width, self.render_height, "MuJoCo Render", None, None)
        if not window:
            glfw.terminate()
            raise RuntimeError("Failed to create GLFW window")

        # Make context current
        glfw.make_context_current(window)

        try:
            # Load MuJoCo model
            model = mujoco.MjModel.from_xml_path(model_path)
            data = mujoco.MjData(model)

            # Create renderer
            renderer = mujoco.Renderer(model, height=self.render_height, width=self.render_width)

            # Simulate and render
            mujoco.mj_resetData(model, data)
            
            # Simulate for the specified duration
            while data.time < self.simulation_duration:
                mujoco.mj_step(model, data)

            # Render final frame
            renderer.update_scene(data)
            image_array = renderer.render()
            
            # Convert to PIL Image
            pil_image = Image.fromarray(image_array)

        except Exception as e:
            print(f"Detailed rendering error for {model_path}: {e}")
            # Create a placeholder image if rendering fails
            pil_image = Image.new('RGB', (self.render_width, self.render_height), color='lightgray')
            
        finally:
            # Cleanup
            if 'renderer' in locals():
                renderer.close()
            glfw.destroy_window(window)
            glfw.terminate()

        return pil_image

    def crop_image_center(self, image):
        """
        Crop the center portion of an image
        
        :param image: PIL Image to crop
        :return: Cropped PIL Image
        """
        width, height = image.size
        
        # Calculate crop dimensions
        new_width = int(width * self.crop_percentage)
        new_height = int(height * self.crop_percentage)
        
        # Calculate crop coordinates
        left = (width - new_width) // 2
        top = (height - new_height) // 2
        right = left + new_width
        bottom = top + new_height
        
        return image.crop((left, top, right, bottom))

    def create_collage(self):
        """
        Create a collage of all MuJoCo models in the directory
        """
        rows, cols = self.grid_layout
        
        # Calculate cropped dimensions
        cropped_width = int(self.render_width * self.crop_percentage)
        cropped_height = int(self.render_height * self.crop_percentage)
        
        # Calculate total collage dimensions with padding
        collage_width = cols * (cropped_width + self.padding) + self.padding
        collage_height = rows * (cropped_height + self.padding) + self.padding
        
        # Create blank collage image with black background
        collage = Image.new('RGB', (collage_width, collage_height), color='black')
        
        # Render and place models
        for idx, model_file in enumerate(self.model_files):
            model_path = os.path.join(self.model_directory, model_file)
            
            try:
                # Render the model
                model_image = self.render_model(model_path)
                
                # Crop the center portion
                cropped_image = self.crop_image_center(model_image)
                
                # Calculate grid position with padding
                row = idx // cols
                col = idx % cols
                
                # Calculate paste coordinates
                x = col * (cropped_width + self.padding) + self.padding
                y = row * (cropped_height + self.padding) + self.padding
                
                # Paste cropped model image into collage
                collage.paste(cropped_image, (x, y))
                
                print(f"Rendered and cropped: {model_file}")
            except Exception as e:
                print(f"Error processing {model_file}: {e}")
        
        # Save collage
        collage.save(self.output_path)
        print(f"Collage saved to {self.output_path}")

def generate_mujoco_collage(model_dir, output_path='mujoco_model_collage.png', 
                             crop_percentage=0.5, padding=10):
    """
    Convenience function to generate MuJoCo model collage
    
    :param model_dir: Directory containing MuJoCo model files
    :param output_path: Path to save the collage
    :param crop_percentage: Percentage of image to keep from center (0.0 to 1.0)
    :param padding: Pixel width of white space between models
    """
    generator = MuJoCoCollageGenerator(
        model_directory=model_dir, 
        output_path=output_path,
        simulation_duration=2.0,  # 2 seconds of simulation per model
        render_width=400,
        render_height=400,
        crop_percentage=crop_percentage,
        padding=padding
    )
    generator.create_collage()

# You can call this directly or import and use in another script
if __name__ == "__main__":
    # Replace with your MuJoCo models directory
    MODEL_DIR = "muj_models"
    generate_mujoco_collage(MODEL_DIR, crop_percentage=0.18, padding=10)