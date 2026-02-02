from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

# 定位项目根目录：config.py 的父目录的父目录
ROOT_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):

    # Path Config 
    datasets_dir: Path = ROOT_DIR / "datasets"
    outputs_dir: Path = ROOT_DIR / "outputs"
    eval_file: Path = datasets_dir / "sft-reason" / "val.jsonl"
    model: Path = '/home/xuzhenhua/models/Qwen2.5-Math-1.5B'
    r1_zero_prompt_file: Path = ROOT_DIR / "cs336_alignment" / "prompts" / "r1_zero.prompt"

    # Vllm Config 
    temperature: float = 0.8
    max_tokens: int = 2048
    stop: list[str] = ["</answer>"]
    include_input: bool = True


    # GPU Config
    cuda_visible_devices: str = "7"  # 默认使用 7 号卡
    
    # Other Config
    app_env: str = "development"

    # 配置 Pydantic 加载行为
    model_config = SettingsConfigDict(
        # 显式指向根目录下的 .env 文件
        env_file=ROOT_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore" # 忽略 .env 中多余的变量
    )

# 实例化
settings = Settings()