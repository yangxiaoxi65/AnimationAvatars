import os

class OpenposeConfig:
    def __init__(self, image_folder: str):
        # input settings
        
        # Output settings
        self.output_json_folder = os.path.join(image_folder, "../openpose_json/")
        self.output_images_folder = os.path.join(image_folder, "../openpose_images/")
        self.write_images_format = "jpg"

        # Pose model settings
        self.model_pose = "BODY_25"
        self.number_people_max = "1"
        self.render_pose = "1"
        self.render_threshold = "0.5"

        # Display settings
        self.display = "0"
        
    def __str__(self):
        return f"""
        OpenposeConfig(
            output_json_folder={self.output_json_folder},
            output_images_folder={self.output_images_folder},
            write_images_format={self.write_images_format},
            model_pose={self.model_pose},
            number_people_max={self.number_people_max},
            render_pose={self.render_pose},
            render_threshold={self.render_threshold},
            display={self.display}
        )
        """
