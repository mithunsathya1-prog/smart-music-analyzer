import os
import subprocess

class StemSplitterService:
    def __init__(self, output_dir="stems_output"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def split_audio(self, input_file: str):
        """
        Splits the given audio file into stems (vocals, drums, bass, other).
        Returns the folder containing the stems.
        """
        # Run demucs via subprocess
        subprocess.run([
            "demucs",
            "-o", self.output_dir,
            input_file
        ], check=True)

        # Path where Demucs saves results
        song_name = os.path.splitext(os.path.basename(input_file))[0]
        model_dir = os.listdir(self.output_dir)[0]  # e.g. "htdemucs"
        result_path = os.path.join(self.output_dir, model_dir, song_name)
        return result_path
