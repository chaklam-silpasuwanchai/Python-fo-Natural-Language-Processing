from huggingface_hub import get_safetensors_metadata

model_id = "mistralai/Mistral-7B-Instruct-v0.1"
precision = "F8"
dtype_bytes = {"F32": 4, "F16": 2, "BF16": 2, "F8": 1, "INT8": 1, "INT4": 0.5}

metadata = get_safetensors_metadata(model_id)
memory = ((sum(metadata.parameter_count.values()) * dtype_bytes[precision]) / (1024**3)) * 1.18
print(f"{model_id=} requires {memory=}GB")