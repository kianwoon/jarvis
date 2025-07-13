"""
Model presets API endpoints for LLM configuration
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, List, Optional
from pydantic import BaseModel
from app.core.llm_settings_cache import get_llm_settings, get_main_llm_full_config

router = APIRouter()

class ModelPreset(BaseModel):
    model_name: str
    display_name: str
    context_length: int
    recommended_max_tokens: int
    default_temperature: float
    default_top_p: float
    supports_thinking: bool
    notes: str
    recommended_system_prompts: Dict[str, str]

# Define model presets
MODEL_PRESETS = {
    "deepseek-r1:8b": ModelPreset(
        model_name="deepseek-r1:8b",
        display_name="DeepSeek R1 8B",
        context_length=128000,
        recommended_max_tokens=16384,
        default_temperature=0.8,
        default_top_p=0.95,
        supports_thinking=True,
        notes="Reasoning-optimized model. Uses ~66% tokens for thinking. For detailed visible output, use non-thinking mode or increase max_tokens.",
        recommended_system_prompts={
            "detailed": "You are Jarvis, an AI assistant. Always provide detailed, comprehensive responses with thorough explanations, examples, and step-by-step breakdowns. Be verbose and informative. Use clean markdown formatting.",
            "balanced": "You are Jarvis, an AI assistant. Provide clear, well-structured responses that balance detail with conciseness. Include key information and examples. Use clean markdown formatting.",
            "concise": "You are Jarvis, an AI assistant. Provide accurate, direct responses. Be concise while maintaining clarity. Use clean markdown formatting."
        }
    ),
    "qwen2.5:8b": ModelPreset(
        model_name="qwen2.5:8b",
        display_name="Qwen 2.5 8B",
        context_length=32768,
        recommended_max_tokens=8192,
        default_temperature=0.7,
        default_top_p=0.9,
        supports_thinking=False,
        notes="Balanced model with good verbosity and accuracy. Excellent for general-purpose tasks.",
        recommended_system_prompts={
            "detailed": "You are Jarvis, an AI assistant. Provide comprehensive, well-structured responses with examples and explanations. Use clean markdown formatting.",
            "balanced": "You are Jarvis, an AI assistant. Provide clear, informative responses. Use clean markdown formatting.",
            "concise": "You are Jarvis, an AI assistant. Provide accurate, concise responses. Use clean markdown formatting."
        }
    ),
    "qwen2.5:14b": ModelPreset(
        model_name="qwen2.5:14b",
        display_name="Qwen 2.5 14B",
        context_length=32768,
        recommended_max_tokens=8192,
        default_temperature=0.7,
        default_top_p=0.9,
        supports_thinking=False,
        notes="Larger Qwen model with enhanced capabilities. Better for complex tasks.",
        recommended_system_prompts={
            "detailed": "You are Jarvis, an AI assistant. Provide comprehensive, nuanced responses with detailed analysis and examples. Use clean markdown formatting.",
            "balanced": "You are Jarvis, an AI assistant. Provide clear, well-reasoned responses. Use clean markdown formatting.",
            "concise": "You are Jarvis, an AI assistant. Provide accurate, efficient responses. Use clean markdown formatting."
        }
    ),
    "llama3.1:8b": ModelPreset(
        model_name="llama3.1:8b",
        display_name="Llama 3.1 8B",
        context_length=8192,
        recommended_max_tokens=4096,
        default_temperature=0.7,
        default_top_p=0.9,
        supports_thinking=False,
        notes="Fast and efficient for general tasks. Limited context window.",
        recommended_system_prompts={
            "detailed": "You are Jarvis, an AI assistant. Provide helpful, detailed responses within the context limitations. Use clean markdown formatting.",
            "balanced": "You are Jarvis, an AI assistant. Provide clear, helpful responses. Use clean markdown formatting.",
            "concise": "You are Jarvis, an AI assistant. Provide brief, accurate responses. Use clean markdown formatting."
        }
    ),
    "mistral:7b": ModelPreset(
        model_name="mistral:7b",
        display_name="Mistral 7B",
        context_length=8192,
        recommended_max_tokens=4096,
        default_temperature=0.7,
        default_top_p=0.9,
        supports_thinking=False,
        notes="Efficient model with good performance for its size.",
        recommended_system_prompts={
            "detailed": "You are Jarvis, an AI assistant. Provide comprehensive responses with clear explanations. Use clean markdown formatting.",
            "balanced": "You are Jarvis, an AI assistant. Provide helpful, clear responses. Use clean markdown formatting.",
            "concise": "You are Jarvis, an AI assistant. Provide brief, accurate responses. Use clean markdown formatting."
        }
    )
}

class SettingsValidation(BaseModel):
    is_valid: bool
    errors: List[str]
    warnings: List[str]

@router.get("/")
async def get_all_presets():
    """Get all available model presets"""
    return {
        "presets": [preset.dict() for preset in MODEL_PRESETS.values()]
    }

@router.get("/{model_name}")
async def get_model_preset(model_name: str):
    """Get preset for a specific model"""
    # Handle both with and without tag versions
    if model_name in MODEL_PRESETS:
        return MODEL_PRESETS[model_name].dict()
    
    # Try without tag
    base_name = model_name.split(":")[0]
    for key, preset in MODEL_PRESETS.items():
        if key.startswith(base_name):
            return preset.dict()
    
    raise HTTPException(status_code=404, detail=f"No preset found for model: {model_name}")

@router.post("/validate")
async def validate_settings(settings: dict) -> SettingsValidation:
    """Validate LLM settings configuration"""
    errors = []
    warnings = []
    
    # Extract values
    max_tokens = int(settings.get("max_tokens", 0))
    context_length = int(settings.get("context_length", 128000))
    temperature = float(settings.get("temperature", 0.7))
    model = settings.get("model", "").lower()
    
    # Validations
    if max_tokens <= 0:
        errors.append("max_tokens must be greater than 0")
    
    if max_tokens > context_length:
        errors.append(f"max_tokens ({max_tokens}) cannot exceed context_length ({context_length})")
    
    if max_tokens > context_length / 2:
        warnings.append(f"Very high max_tokens ({max_tokens}) relative to context may cause issues")
    
    if max_tokens > 32768:
        warnings.append("max_tokens > 32k is extremely high and may cause performance issues")
    
    # Model-specific checks
    if "deepseek" in model:
        if max_tokens < 8192:
            warnings.append("DeepSeek R1 benefits from higher max_tokens (8192-16384) for detailed responses")
        if max_tokens > 100000:
            errors.append("max_tokens appears to be set to context window size. Use 16384 for output tokens.")
    
    if temperature > 1.5:
        warnings.append(f"High temperature ({temperature}) may cause inconsistent responses")
    
    if temperature < 0.1:
        warnings.append(f"Very low temperature ({temperature}) may cause repetitive responses")
    
    return SettingsValidation(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings
    )

@router.post("/apply")
async def apply_preset_to_settings(model_name: str, current_settings: dict):
    """Apply model preset to current settings"""
    preset = None
    
    # Find matching preset
    if model_name in MODEL_PRESETS:
        preset = MODEL_PRESETS[model_name]
    else:
        # Try base name
        base_name = model_name.split(":")[0]
        for key, p in MODEL_PRESETS.items():
            if key.startswith(base_name):
                preset = p
                break
    
    if not preset:
        raise HTTPException(status_code=404, detail=f"No preset found for model: {model_name}")
    
    # Apply preset values
    updated_settings = current_settings.copy()
    updated_settings["model"] = model_name
    updated_settings["max_tokens"] = str(preset.recommended_max_tokens)
    updated_settings["context_length"] = str(preset.context_length)
    
    # Update mode settings using centralized helper functions
    # Update thinking mode parameters
    if "thinking_mode_params" in updated_settings:
        updated_settings["thinking_mode_params"]["temperature"] = preset.default_temperature
        updated_settings["thinking_mode_params"]["top_p"] = preset.default_top_p
    
    # Update non-thinking mode parameters  
    if "non_thinking_mode_params" in updated_settings:
        updated_settings["non_thinking_mode_params"]["temperature"] = preset.default_temperature
        updated_settings["non_thinking_mode_params"]["top_p"] = preset.default_top_p
    
    # Update main LLM configuration
    if "main_llm" in updated_settings:
        updated_settings["main_llm"]["model"] = model_name
        updated_settings["main_llm"]["max_tokens"] = preset.recommended_max_tokens
    
    # Legacy schema support (deprecated)
    if "thinking_mode" in updated_settings:
        updated_settings["thinking_mode"]["temperature"] = str(preset.default_temperature)
        updated_settings["thinking_mode"]["top_p"] = str(preset.default_top_p)
    
    if "non_thinking_mode" in updated_settings:
        updated_settings["non_thinking_mode"]["temperature"] = str(preset.default_temperature)
        updated_settings["non_thinking_mode"]["top_p"] = str(preset.default_top_p)
    
    # Add recommended system prompt to main_llm
    if "main_llm" not in updated_settings:
        updated_settings["main_llm"] = {}
    if "system_prompt" not in updated_settings["main_llm"] or not updated_settings["main_llm"]["system_prompt"]:
        updated_settings["main_llm"]["system_prompt"] = preset.recommended_system_prompts["balanced"]
    
    return {
        "settings": updated_settings,
        "preset_applied": preset.display_name,
        "notes": preset.notes
    }