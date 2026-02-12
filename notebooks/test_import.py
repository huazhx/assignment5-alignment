import os
import sys
import pathlib

os.environ['CUDA_VISIBLE_DEVICES'] = '7'

# Add project root to sys.path and import in the same cell
project_root = pathlib.Path("/home/xuzhenhua/git/assignment5-alignment")
sys.path.insert(0, str(project_root))

# Import Generator immediately after modifying sys.path
from core.engine.generator import Generator

# # Initialize the Generator
# generator = Generator(
#     temperature=1.0,
#     max_tokens=1024,
#     stop=["</answer>"],
#     model="/home/xuzhenhua/models/Qwen2.5-Math-1.5B"
# )