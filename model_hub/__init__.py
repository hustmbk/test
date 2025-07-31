
# Import DeepSeekV2Model first since it's the primary focus
from .deepseek_v2_model import DeepSeekV2Model

# Optional imports for other models (commented out to avoid dependency issues)
try:
    from .llama import LlamaModel
except ImportError:
    LlamaModel = None

try:
    from .qwen import QwenModel  
except ImportError:
    QwenModel = None

from .deepseek_v2_attention import RMSNorm