import os
import struct
import torch
import numpy as np
from safetensors.torch import save_file
from gguf import dequantize
from typing import Dict, Optional, List, Any, BinaryIO, Tuple
from transformers import PretrainedConfig
import logging
import json
import tempfile
import shutil
import transformers
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --- Constants ---
GGUF_MAGIC = b"GGUF"
TEMP_DIR = "./tensormemtmpswap"  # Temporary directory for tensor data


# Define the metadata value types
class GGUFValueType:
    UINT8 = 0
    INT8 = 1
    UINT16 = 2
    INT16 = 3
    UINT32 = 4
    INT32 = 5
    FLOAT32 = 6
    BOOL = 7
    STRING = 8
    ARRAY = 9
    UINT64 = 10
    INT64 = 11
    FLOAT64 = 12


# --- Helper Functions (Parsing) ---
def read_bytes(file: BinaryIO, num_bytes: int) -> bytes:
    data = file.read(num_bytes)
    if len(data) != num_bytes:
        raise ValueError(f"Expected {num_bytes} bytes, but got {len(data)}.")
    return data


def read_u8(file: BinaryIO) -> int:
    return struct.unpack("<B", read_bytes(file, 1))[0]


def read_i8(file: BinaryIO) -> int:
    return struct.unpack("<b", read_bytes(file, 1))[0]


def read_u16(file: BinaryIO) -> int:
    return struct.unpack("<H", read_bytes(file, 2))[0]


def read_i16(file: BinaryIO) -> int:
    return struct.unpack("<h", read_bytes(file, 2))[0]


def read_u32(file: BinaryIO) -> int:
    return struct.unpack("<I", read_bytes(file, 4))[0]


def read_i32(file: BinaryIO) -> int:
    return struct.unpack("<i", read_bytes(file, 4))[0]


def read_u64(file: BinaryIO) -> int:
    return struct.unpack("<Q", read_bytes(file, 8))[0]


def read_i64(file: BinaryIO) -> int:
    return struct.unpack("<q", read_bytes(file, 8))[0]


def read_f32(file: BinaryIO) -> float:
    return struct.unpack("<f", read_bytes(file, 4))[0]


def read_f64(file: BinaryIO) -> float:
    return struct.unpack("<d", read_bytes(file, 8))[0]


def read_bool(file: BinaryIO) -> bool:
    return read_u8(file) != 0


def read_string(file: BinaryIO) -> str:
    length = read_u64(file)
    data = read_bytes(file, length)
    return data.decode("utf-8", errors="replace")


def read_array(file: BinaryIO) -> List[Any]:
    value_type = read_u32(file)
    length = read_u64(file)
    return [read_value(file, value_type) for _ in range(length)]


def read_value(file: BinaryIO, value_type: int) -> Any:
    if value_type == GGUFValueType.UINT8:
        return read_u8(file)
    elif value_type == GGUFValueType.INT8:
        return read_i8(file)
    elif value_type == GGUFValueType.UINT16:
        return read_u16(file)
    elif value_type == GGUFValueType.INT16:
        return read_i16(file)
    elif value_type == GGUFValueType.UINT32:
        return read_u32(file)
    elif value_type == GGUFValueType.INT32:
        return read_i32(file)
    elif value_type == GGUFValueType.FLOAT32:
        return read_f32(file)
    elif value_type == GGUFValueType.UINT64:
        return read_u64(file)
    elif value_type == GGUFValueType.INT64:
        return read_i64(file)
    elif value_type == GGUFValueType.FLOAT64:
        return read_f64(file)
    elif value_type == GGUFValueType.BOOL:
        return read_bool(file)
    elif value_type == GGUFValueType.STRING:
        return read_string(file)
    elif value_type == GGUFValueType.ARRAY:
        return read_array(file)
    else:
        raise ValueError(f"Invalid GGUF value type: {value_type}")


def read_metadata(file: BinaryIO) -> Dict[str, Any]:
    metadata = {}
    metadata_count = read_u64(file)
    logging.info(f"Metadata count: {metadata_count}")

    for _ in range(metadata_count):
        key = read_string(file)
        value_type = read_u32(file)
        value = read_value(file, value_type)
        metadata[key] = value

    return metadata


def read_tensor_info(file: BinaryIO) -> Tuple[str, List[int], int, int]:
    name = read_string(file)
    n_dims = read_u32(file)
    dims = [read_u64(file) for _ in range(n_dims)]
    tensor_type = read_u32(file)  # GGML type
    offset = read_u64(file)
    return name, dims, tensor_type, offset


from gguf.constants import GGML_QUANT_SIZES


def dequantize_tensor_data(data: bytes, tensor_type: int) -> np.ndarray:
    """Dequantizes tensor data based on GGML type."""
    if tensor_type not in GGML_QUANT_SIZES:
        raise ValueError(f"Unsupported GGML tensor type: {tensor_type}")

    block_size, type_size = GGML_QUANT_SIZES[tensor_type]

    # Handle F32 and F16 as special cases (no dequantization needed)
    if tensor_type == 0:  # GGML_TYPE_F32
        return np.frombuffer(data, dtype=np.float32)
    elif tensor_type == 1:  # GGML_TYPE_F16
        return np.frombuffer(data, dtype=np.float16).astype(np.float32)
    else:
        # Correctly dequantize using gguf.dequantize
        return dequantize(np.frombuffer(data, dtype=np.uint8), tensor_type)


def format_metadata_value(value: Any, max_length: int = 80) -> str:
    if isinstance(value, str):
        if len(value) > max_length:
            return value[:max_length - 3] + "..."
        return value
    elif isinstance(value, bytes):
        try:
            decoded_value = value.decode("utf-8", errors="replace")
            if len(decoded_value) > max_length:
                return decoded_value[:max_length - 3] + "..."
            return decoded_value
        except:
            return f"Binary data ({len(value)} bytes)"
    elif isinstance(value, list):
        if not value:
            return "[]"
        return f"List (length {len(value)}): {', '.join(str(format_metadata_value(item, max_length // 2)) for item in value[:3])}" + (
            "..." if len(value) > 3 else ""
        )
    else:
        return str(value)


def print_metadata_table(metadata: Dict[str, Any]) -> None:
    print("-" * 80)
    print(f"{'Key':<40} | {'Value Type':<15} | {'Value Preview':<25}")
    print("-" * 80)
    for key, value in metadata.items():
        value_type_name = type(value).__name__
        formatted_value = format_metadata_value(value)
        print(f"{key:<40} | {value_type_name:<15} | {formatted_value:<25}")
    print("-" * 80)


def convert_gguf_to_safetensors(gguf_path: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(TEMP_DIR, exist_ok=True)  # Ensure TEMP_DIR exists *before* mkdtemp
    temp_dir = tempfile.mkdtemp(dir=TEMP_DIR)

    try:
        with open(gguf_path, "rb") as f:
            magic = read_bytes(f, 4)
            if magic != GGUF_MAGIC:
                raise ValueError("Invalid GGUF file (magic number mismatch).")

            version = read_u32(f)
            logging.info(f"GGUF Version: {version}")
            tensor_count = read_u64(f)
            logging.info(f"Tensor count: {tensor_count}")

            metadata = read_metadata(f)
            print_metadata_table(metadata)

            tensor_infos = []
            for _ in range(tensor_count):
                tensor_infos.append(read_tensor_info(f))

            tensor_files = {}
            for name, dims, tensor_type, offset in tensor_infos:
                logging.info(f"Processing tensor: {name}, offset: {offset}, dims: {dims}")
                f.seek(offset)
                block_size, type_size = GGML_QUANT_SIZES[tensor_type]
                num_elements = np.prod(dims)

                if tensor_type in (0, 1):
                    size_in_bytes = num_elements * type_size
                else:
                    size_in_bytes = (num_elements // block_size) * type_size
                logging.info(f"Tensor {name}, Size in bytes: {size_in_bytes}")

                tensor_data = read_bytes(f, size_in_bytes)
                dequantized_data = dequantize_tensor_data(tensor_data, tensor_type)
                dequantized_data = dequantized_data.reshape(dims)

                filepath = os.path.join(temp_dir, f"{name}.safetensors")
                save_file({name: torch.from_numpy(dequantized_data).to(dtype=torch.float16)}, filepath)
                tensor_files[name] = filepath

        # Use safetensors' streaming mechanism
        safetensors_path = os.path.join(output_dir, "model.safetensor")
        try:
            with open(safetensors_path, "wb") as out_file:
                header = {}
                header_size = 0
                out_file.write(b" " * 8)

                for name, filepath in tensor_files.items():
                    with open(filepath, "rb") as in_file:
                        tensor_data = in_file.read()

                    with open(filepath, "rb") as f_temp:
                        header_temp = json.loads(f_temp.read(struct.unpack("<Q", f_temp.read(8))[0]))
                    tensor_shape = header_temp[name]["shape"]
                    tensor_dtype = header_temp[name]["dtype"]
                    data_offsets = header_temp[name]["data_offsets"]

                    header[name] = {
                        "dtype": tensor_dtype,
                        "shape": tensor_shape,
                        "data_offsets": [
                            data_offsets[0] + header_size + 8,
                            data_offsets[1] + header_size + 8,
                        ],
                    }

                    out_file.write(tensor_data[data_offsets[0] + 8 :])
                    os.remove(filepath)  # Clean up

                header_str = json.dumps(header, separators=(",", ":")) + "\0"
                header_bytes = header_str.encode("utf-8")
                header_size = len(header_bytes)

                out_file.seek(0)
                out_file.write(struct.pack("<Q", header_size))
                out_file.write(header_bytes)
            logging.info(f"SafeTensors file saved to: {safetensors_path}")

        except Exception as e:
            logging.error(f"Failed to save SafeTensors file: {e}")
            traceback.print_exc()
            return

    except Exception as e:
        logging.error(f"Error during GGUF parsing: {e}")
        traceback.print_exc()
        return

    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

    # --- Configuration File Generation ---
    config_path = os.path.join(output_dir, "config.json")
    try:
        arch = metadata.get("general.architecture", "unknown")
        if isinstance(arch, bytes):
            arch = arch.decode("utf-8", errors="ignore")

        n_embd = None
        n_head = None
        n_layer = None
        n_positions = None
        rope_dimension_count = None
        vocab_size = None
        intermediate_size = None

        for key, value in metadata.items():
            if "embedding_length" in key:
                n_embd = value
            elif "head_count" in key and "head_count_kv" not in key:
                n_head = value
            elif "block_count" in key:
                n_layer = value
            elif "context_length" in key:
                n_positions = value
            elif "rope.dimension_count" in key:
                rope_dimension_count = value
            elif "feed_forward_length" in key:
                intermediate_size = value
            elif key == "general.file_type":
                if isinstance(value, (int, float)):
                    vocab_size = int(value)

        if not any([n_embd, n_head, n_layer, n_positions]):
            logging.warning("No essential parameters")
        architecture_map = {
            "llama": "LlamaForCausalLM",
            "mistral": "MistralForCausalLM",
            "gemma": "GemmaForCausalLM",
            "phi3": "Phi3ForCausalLM",
        }
        model_type = arch
        arch_class = architecture_map.get(arch.lower(), "AutoModelForCausalLM")
        tokenizer_class = "AutoTokenizer"  # Default
        if arch_class == "AutoModelForCausalLM":
            logging.warning(f"Architecture '{arch}' not found in architecture_map.  Using AutoModel.")
        if arch_class == "LlamaForCausalLM":
            tokenizer_class = "LlamaTokenizer"
        elif arch_class == "MistralForCausalLM":
            tokenizer_class = "MistralTokenizer"
        elif arch_class == "Phi3ForCausalLM":
            tokenizer_class = "Phi3Tokenizer"

        config_dict = {
            "_name_or_path": arch,
            "architectures": [arch_class],
            "model_type": model_type,
            "auto_map": {
                "AutoConfig": f"configuration_{model_type}.{arch_class.replace('ForCausalLM', 'Config')}",
                "AutoModelForCausalLM": f"modeling_{model_type}.{arch_class}",
            },
            "tokenizer_class": tokenizer_class,
            "use_cache": True,
            "attention_dropout": 0.0,
            "embd_pdrop": 0.0,
            "resid_pdrop": 0.0,
            "tie_word_embeddings": False,
            "attention_bias": False,
            "rope_theta": 10000.0,
            "rms_norm_eps": 1e-05,
        }
        if n_embd:
            config_dict["hidden_size"] = n_embd
        if n_head:
            config_dict["num_attention_heads"] = n_head
        if n_head:
            config_dict["num_key_value_heads"] = n_head
        if n_layer:
            config_dict["num_hidden_layers"] = n_layer
        if n_positions:
            config_dict["max_position_embeddings"] = n_positions
            config_dict["original_max_position_embeddings"] = n_positions
        if rope_dimension_count:
            config_dict["rope_scaling"] = {"type": "dynamic", "factor": float(rope_dimension_count)}
        if intermediate_size:
            config_dict["intermediate_size"] = intermediate_size
        if vocab_size:
            config_dict["vocab_size"] = vocab_size

        config_dict["hidden_act"] = "silu"
        config = PretrainedConfig.from_dict(config_dict)
        config.save_pretrained(output_dir)
        logging.info(f"config.json saved to: {config_path}")

        with open(config_path, "r") as f:  # Debugging: Print the config
            config_loaded = json.load(f)
        print(json.dumps(config_loaded, indent=4))

    except Exception as e:
        logging.error(f"Error creating config.json: {e}")
        traceback.print_exc()
        return

    # --- Tokenizer File Generation (Refactored and Corrected) ---

    tokenizer_dir = output_dir
    tokenizer_config_path = os.path.join(tokenizer_dir, "tokenizer_config.json")
    tokenizer_json_path = os.path.join(tokenizer_dir, "tokenizer.json")
    special_tokens_map_path = os.path.join(tokenizer_dir, "special_tokens_map.json")

    try:
        # --- Helper Functions (for Tokenizer) ---

        def _get_tokens(metadata: Dict[str, Any]) -> List[str]:
            """Safely extracts and decodes tokens from metadata."""
            tokens = metadata.get("tokenizer.ggml.tokens", [])
            if not isinstance(tokens, list):
                logging.warning("tokenizer.ggml.tokens is not a list.")
                return []
            return [
                t.decode("utf-8", errors="replace") if isinstance(t, bytes) else str(t) for t in tokens
            ]

        def _get_token_type(metadata: Dict[str, Any], index: int, tokenizer_type: str) -> str:
            """Determines the token type based on metadata and tokenizer type."""
            token_types = metadata.get("tokenizer.ggml.token_type")
            if (
                isinstance(token_types, list)
                and len(token_types) > index
                and tokenizer_type.lower() == "llama"
            ):
                type_map = {
                    1: "Unknown",
                    2: "Control",
                    3: "UserDefined",
                    6: "Byte",
                }
                return type_map.get(token_types[index], "Normal")  # Default to "Normal"
            return "Normal"  # Default if not a Llama-like tokenizer or type not found

        def _is_special_token(metadata: Dict[str, Any], index: int) -> bool:
            """Checks if a token at a given index is a special token."""
            for key, value in metadata.items():
                if "tokenizer.ggml" not in key:
                    continue
                if "token_id" in key and value == index:
                    return True
            return False

        # --- Extract Tokenizer Information ---
        tokenizer_model_gguf = metadata.get("tokenizer.ggml.model", "unknown")
        if isinstance(tokenizer_model_gguf, bytes):
            tokenizer_model_gguf = tokenizer_model_gguf.decode("utf-8", errors="replace")

        tokenizer_type = metadata.get("tokenizer.ggml.model", "BPE")  # Default to BPE
        if isinstance(tokenizer_type, bytes):
            tokenizer_type = tokenizer_type.decode("utf-8", errors="replace")
        tokens = _get_tokens(metadata)
        scores = metadata.get("tokenizer.ggml.scores", [])
        if not isinstance(scores, list):
            scores = []
            logging.warning("tokenizer.ggml.scores did not return a list")

        merges = metadata.get("tokenizer.ggml.merges", [])
        if isinstance(merges, list):
            merges = [m.decode("utf-8", errors="replace") if isinstance(m, bytes) else str(m) for m in merges]
        else:
            merges = []
            logging.warning("tokenizer.ggml.merges is not a list")

        # --- special_tokens_map.json ---
        special_tokens_map = {}
        for key, value in metadata.items():
            if "tokenizer.ggml" in key and "token_id" in key:
                token_name = key.replace("tokenizer.ggml.", "").replace("_token_id", "")
                if isinstance(value, int) and 0 <= value < len(tokens):
                    token_info = {
                        "content": tokens[value],
                        "lstrip": False,
                        "normalized": False,
                        "rstrip": False,
                        "single_word": False,
                    }
                    if _get_token_type(metadata, value, tokenizer_type) == "UserDefined":
                        token_info["normalized"] = True
                    if _get_token_type(metadata, value, tokenizer_type) == "Byte":
                        token_info["normalized"] = True

                    special_tokens_map[token_name + "_token"] = token_info

        with open(special_tokens_map_path, "w") as f:
            json.dump(special_tokens_map, f, indent=2)
        logging.info(f"special_tokens_map.json saved to: {special_tokens_map_path}")

        # --- tokenizer_config.json ---
        tokenizer_config = {
            "tokenizer_class": tokenizer_class,  # Use derived class names
        }

        if "tokenizer.chat_template" in metadata:
            tokenizer_config["chat_template"] = metadata["tokenizer.chat_template"]
        if "tokenizer.ggml.add_bos_token" in metadata:
            tokenizer_config["add_bos_token"] = metadata["tokenizer.ggml.add_bos_token"]
        if "tokenizer.ggml.add_eos_token" in metadata:
            tokenizer_config["add_eos_token"] = metadata["tokenizer.ggml.add_eos_token"]

        for key in ["bos_token", "eos_token", "unk_token", "pad_token"]:
            token_key = key + "_token"  # Corrected key:  "bos_token_token" -> "bos_token"
            if token_key in special_tokens_map:
                tokenizer_config[key] = special_tokens_map[token_key]["content"]

        # --- added_tokens_decoder (CORRECTLY POPULATED) ---
        added_tokens_decoder = {}
        for i, token in enumerate(tokens):
            token_info = {
                "content": token,
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": _is_special_token(metadata, i),
            }
            if _get_token_type(metadata, i, tokenizer_type) == "UserDefined":
                token_info["normalized"] = True
            if _get_token_type(metadata, i, tokenizer_type) == "Byte":
                token_info["normalized"] = True
            added_tokens_decoder[str(i)] = token_info
        tokenizer_config["added_tokens_decoder"] = added_tokens_decoder

        with open(tokenizer_config_path, "w") as f:
            json.dump(tokenizer_config, f, indent=2)
        logging.info(f"tokenizer_config.json saved to: {tokenizer_config_path}")

        # --- tokenizer.json ---
        tokenizer_json_content = {
            "version": "1.0",
            "truncation": None,
            "padding": None,
            "added_tokens": [],  # This will be populated correctly
            "normalizer": None,
            "pre_tokenizer": None,
            "post_processor": None,
            "decoder": None,
            "model": {
                "type": tokenizer_type,
                "vocab": {token: i for i, token in enumerate(tokens)},  # Correct vocab
                "merges": merges,
            },
        }
        if scores:
            tokenizer_json_content["model"]["scores"] = scores

        # --- added_tokens (for tokenizer.json) ---
        added_tokens_list = []
        for i, token in enumerate(tokens):
            token_info = {
                "id": i,
                "content": token,
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": _is_special_token(metadata, i),
            }
            if _get_token_type(metadata, i, tokenizer_type) == "UserDefined":
                token_info["normalized"] = True
            if _get_token_type(metadata, i, tokenizer_type) == "Byte":
                token_info["normalized"] = True

            added_tokens_list.append(token_info)
        tokenizer_json_content["added_tokens"] = added_tokens_list

        with open(tokenizer_json_path, "w") as f:
            json.dump(tokenizer_json_content, f, indent=2)
        logging.info(f"tokenizer.json saved to: {tokenizer_json_path}")

    except Exception as e:
        logging.error(f"Tokenizer Files Exception: {e}")
        traceback.print_exc()
        return

    # --- Generation Config ---
    generation_config_path = os.path.join(output_dir, "generation_config.json")
    try:
        bos_token_id = metadata.get("tokenizer.ggml.bos_token_id")
        eos_token_id = metadata.get("tokenizer.ggml.eos_token_id")
        pad_token_id = metadata.get("tokenizer.ggml.pad_token_id")
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]

        generation_config = {
            "_from_model_config": True,
            "bos_token_id": bos_token_id,
            "eos_token_id": eos_token_id,
            "pad_token_id": pad_token_id,
            "transformers_version": transformers.__version__,
        }
        with open(generation_config_path, "w") as f:
            json.dump(generation_config, f, indent=2)
        logging.info(f"generation_config.json saved to: {generation_config_path}")

    except Exception as e:
        logging.error(f"Failed to create generation_config.json: {e}")
        traceback.print_exc()
        return

    finally:
        init_file_path = os.path.join(output_dir, "__init__.py")
        if not os.path.exists(init_file_path):
            open(init_file_path, "a").close()
            logging.info(f"Created empty __init__.py in {output_dir}")
    logging.info("Conversion complete!")



import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import sys
import traceback

def perform_inference(model_path: str, prompt: str, max_new_tokens: int = 128, use_cpu: bool = False) -> str:
    """
    Performs inference using a Hugging Face model loaded from a specified path.

    Args:
        model_path: Path to the directory containing the converted model (SafeTensors, config.json, etc.).
        prompt: The input text prompt.
        max_new_tokens: The maximum number of new tokens to generate.
        use_cpu: if True, performs inference on CPU.  Defaults to GPU if available.

    Returns:
        The generated text.
    """

    # Determine device
    device = "cuda:0" if torch.cuda.is_available() and not use_cpu else "cpu"
    print(f"Using device: {device}")

    # Load tokenizer and model
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,  # Use float16 for efficiency
            trust_remote_code=True,
            device_map=device,  #  Explicitly set device
            use_safetensors=True #use safetensors
        )

        #Load generation config
        model.generation_config = GenerationConfig.from_pretrained(model_path)
    except Exception as e:
        print(f"Error loading model or tokenizer: {e}")
        traceback.print_exc()  # Print the full traceback
        return ""


    # Tokenize input prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    # Perform inference
    try:
        with torch.no_grad():  # Disable gradient calculation for inference
            output_ids = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,  # Specify pad_token_id if needed
                eos_token_id=tokenizer.eos_token_id,   # Specify eos_token_id if needed
                # Other generation parameters can be added here, e.g.:
                # do_sample=True,
                # top_k=50,
                # top_p=0.95,
                # temperature=0.7,
                # repetition_penalty=1.2,
            )
    except Exception as e:
        print(f"Error during inference: {e}")
        traceback.print_exc() # Print the full traceback
        return ""

    # Decode generated text
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return generated_text


def main():
    gguf_file_path = "./source_test.gguf"
    output_directory = "./converted_model"
    convert_gguf_to_safetensors(gguf_file_path, output_directory)

    user_prompt = "Explain the importance of low latency LLMs"

    # --- Run Inference (GPU, if available) ---
    print("Running inference on GPU (if available)...")
    generated_output = perform_inference(output_directory, user_prompt)
    print(f"Generated Text (GPU/Default):\n{generated_output}\n")

    # --- Run Inference (CPU) ---
    print("Running inference on CPU...")
    generated_output_cpu = perform_inference(output_directory, user_prompt, use_cpu=True)
    print(f"Generated Text (CPU):\n{generated_output_cpu}")

if __name__ == "__main__":
    main()