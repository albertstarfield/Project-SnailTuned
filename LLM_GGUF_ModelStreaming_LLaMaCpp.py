import os
import struct
import torch
import numpy as np
from safetensors.torch import save_file
from gguf import GGUFReader, dequantize
from typing import Dict, Optional, List, Any, Tuple
from transformers import PretrainedConfig
import logging
import json
import shutil
import transformers
from typing import BinaryIO


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --- Constants ---
GGUF_MAGIC = b"GGUF"
TEMP_DIR = "./tensormemtmpswap"  # Kept, even though not strictly used

# Define the metadata value types (original)
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

# --- Helper Functions (Parsing) --- (Original parsing functions)
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

def format_metadata_value(value: Any, max_length: int = 80) -> str:
    if isinstance(value, str):
        if len(value) > max_length:
            return value[:max_length-3] + "..."
        return value
    elif isinstance(value, bytes):
        try:
            decoded_value = value.decode("utf-8", errors="replace")
            if len(decoded_value) > max_length:
                return decoded_value[:max_length-3] + "..."
            return decoded_value
        except:
           return f"Binary data ({len(value)} bytes)"
    elif isinstance(value, list):
        if not value:
            return "[]"
        return f"List (length {len(value)}): {', '.join(str(format_metadata_value(item, max_length // 2)) for item in value[:3])}" + ("..." if len(value) > 3 else "")
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

    try:
        # 1. Metadata Extraction (Original Parsing)
        with open(gguf_path, "rb") as f:  # Use BinaryIO
            magic = read_bytes(f, 4)
            if magic != GGUF_MAGIC:
                raise ValueError("Invalid GGUF file (magic number mismatch).")
            version = read_u32(f)
            _ = read_u64(f)  # tensor_count (ignored)
            _ = read_u64(f)  # metadata_count (for read_metadata)

            logging.info(f"GGUF Version: {version}")
            metadata = read_metadata(f)  # Pass 'f' here!
            print_metadata_table(metadata)

        # 2. Tensor Conversion (GGUFReader for data, dequantize for processing)
        reader = GGUFReader(gguf_path)  # GGUFReader ONLY for tensors
        tensors_dict: Dict[str, torch.Tensor] = {}
        for tensor in reader.tensors:
            tensor_name = tensor.name
            logging.info(f"Processing tensor: {tensor_name}")

            weights = dequantize(tensor.data, tensor.tensor_type).copy()
            weights_tensor = torch.from_numpy(weights).to(dtype=torch.float16)
            tensors_dict[tensor_name] = weights_tensor

            del weights  # Explicit memory management
            del weights_tensor
        
        # 3. Save to Safetensors
        safetensors_path = os.path.join(output_dir, "model.safetensor")
        save_file(tensors_dict, safetensors_path)
        logging.info(f"SafeTensors file saved to: {safetensors_path}")
        tensors_dict.clear() #cleanup


    except Exception as e:
        logging.error(f"Error during GGUF parsing or conversion: {e}")
        return

    # --- Configuration and Tokenizer Files (using parsed metadata) ---
    config_path = os.path.join(output_dir, "config.json")
    try:
        arch = metadata.get("general.architecture", "unknown")
        if isinstance(arch, bytes):
            arch = arch.decode('utf-8', errors='ignore')

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
            "gemma":"GemmaForCausalLM",
            "phi3": "Phi3ForCausalLM",
        }
        model_type = arch
        arch_class = architecture_map.get(arch.lower(), "AutoModelForCausalLM")
        
        # Determine tokenizer class more robustly
        tokenizer_class = "AutoTokenizer"  # Default
        if arch_class == "LlamaForCausalLM":
            tokenizer_class = "LlamaTokenizer"
        # Add other architecture-specific tokenizer classes as needed

        if arch_class == "AutoModelForCausalLM":
            logging.warning(f"Architecture '{arch}' not found in architecture_map. Using AutoModel.")

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
        if n_embd: config_dict["hidden_size"] = n_embd
        if n_head: config_dict["num_attention_heads"] = n_head
        if n_head: config_dict["num_key_value_heads"] = n_head
        if n_layer: config_dict["num_hidden_layers"] = n_layer
        if n_positions:
            config_dict["max_position_embeddings"] = n_positions
            config_dict["original_max_position_embeddings"] = n_positions
        if rope_dimension_count: config_dict["rope_scaling"] = {"type": "dynamic", "factor": float(rope_dimension_count)}
        if intermediate_size: config_dict["intermediate_size"] = intermediate_size
        if vocab_size: config_dict["vocab_size"] = vocab_size

        config_dict["hidden_act"] = "silu"
        config = PretrainedConfig.from_dict(config_dict)
        config.save_pretrained(output_dir)
        logging.info(f"config.json saved to: {config_path}")

        with open(config_path, "r") as f:  # Debug print (original)
            config_loaded = json.load(f)
        print(json.dumps(config_loaded, indent=4))

    except Exception as e:
        logging.error(f"Error creating config.json: {e}")
        return

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
        return
    # --- tokenizer_config.json --- (CORRECTED SECTION)
    tokenizer_dir = output_dir
    tokenizer_config_path = os.path.join(tokenizer_dir, "tokenizer_config.json")
    tokenizer_json_path = os.path.join(tokenizer_dir, "tokenizer.json")
    special_tokens_map_path = os.path.join(tokenizer_dir, "special_tokens_map.json")

    try:
      tokenizer_model_gguf = metadata.get("tokenizer.ggml.model", "unknown")
      if isinstance(tokenizer_model_gguf, bytes):
          tokenizer_model_gguf = tokenizer_model_gguf.decode("utf-8", errors="replace")

      tokenizer_type = metadata.get("tokenizer.ggml.model", "BPE")
      if isinstance(tokenizer_type, bytes):
          tokenizer_type = tokenizer_type.decode("utf-8", errors="replace")
      # --- special_tokens_map.json --- (Keep as before, it's correct)
      try:
          special_tokens_map = {}
          tokens: List[str] = metadata.get("tokenizer.ggml.tokens", [])
          if isinstance(tokens, list):
              tokens = [t.decode("utf-8", errors="replace") if isinstance(t, bytes) else t for t in tokens]
          else:
              tokens = []
              logging.warning("tokenizer.ggml.tokens did not return a list.")

          special_token_ids = {}
          for key, value in metadata.items():
              if "tokenizer.ggml" in key and "token_id" in key:
                  token_name = key.replace("tokenizer.ggml.", "").replace("_token_id", "")
                  if isinstance(value, int) and 0 <= value < len(tokens):
                      special_token_ids[token_name] = value
          for token_name, token_id in special_token_ids.items():
              if token_id < len(tokens):
                  token_info = {
                      "content": tokens[token_id],
                      "lstrip": False,
                      "normalized": False,
                      "rstrip": False,
                      "single_word": False,
                  }
                  token_types = metadata.get("tokenizer.ggml.token_type")
                  if isinstance(token_types, list) and len(token_types) == len(tokens) and tokenizer_type.lower() == "llama":
                      type_map = {
                          1: "Unknown",
                          2: "Control",
                          3: "UserDefined",
                          6: "Byte",
                      }
                      if token_types[token_id] in type_map:
                          if type_map[token_types[token_id]] == "UserDefined":
                              token_info["normalized"] = False
                          elif type_map[token_types[token_id]] == "Byte":
                              token_info["normalized"] = False
                  special_tokens_map[token_name + "_token"] = token_info

          with open(special_tokens_map_path, "w") as f:
              json.dump(special_tokens_map, f, indent=2)
          logging.info(f"special_tokens_map.json saved to: {special_tokens_map_path}")

      except Exception as e:
          logging.error(f"Failed to create special_tokens_map.json: {e}")
          return
      # --- tokenizer_config.json --- (CORRECTED)
      try:
          tokenizer_config = {
              "tokenizer_class": tokenizer_class,  # Use determined class
              "add_bos_token": metadata.get("tokenizer.ggml.add_bos_token", False), #get from metadata
              "add_eos_token": metadata.get("tokenizer.ggml.add_eos_token", False),
              "clean_up_tokenization_spaces":False, #default
              "model_max_length": 131072, #default
              "padding_side": "left",
              "sp_model_kwargs": {},
              "legacy": False,
              "use_default_system_prompt":False
          }

          # Chat template (if available)
          if "tokenizer.chat_template" in metadata:
              tokenizer_config["chat_template"] = metadata["tokenizer.chat_template"]

          # Add bos/eos/unk/pad *if* they are defined in special_tokens_map
          for token_type in ["bos", "eos", "unk", "pad"]:
              token_key = f"{token_type}_token"
              if token_key in special_tokens_map:
                  tokenizer_config[token_type] = special_tokens_map[token_key]["content"]


          # --- added_tokens_decoder (ONLY SPECIAL TOKENS) ---
          added_tokens_decoder = {}
          for token_name, token_info in special_tokens_map.items():
                #remove _token from the end
                token_name_clean = token_name.replace("_token", "")
                token_id = metadata.get(f"tokenizer.ggml.{token_name_clean}_token_id")
                if token_id is not None: # Make sure it exists in metadata
                  token_entry = {
                      "content": token_info["content"],  # Use the extracted content
                      "lstrip": token_info.get("lstrip", False), #get value
                      "normalized": token_info.get("normalized", False),
                      "rstrip": token_info.get("rstrip", False),
                      "single_word": token_info.get("single_word", False),
                      "special": True,  # Mark as special
                  }
                  added_tokens_decoder[str(token_id)] = token_entry


          tokenizer_config["added_tokens_decoder"] = added_tokens_decoder

          with open(tokenizer_config_path, "w") as f:
              json.dump(tokenizer_config, f, indent=2)
          logging.info(f"tokenizer_config.json saved to: {tokenizer_config_path}")

      except Exception as e:
          logging.error(f"Failed to create tokenizer_config.json: {e}")
          return

        # --- tokenizer.json --- (Keep, as it's correct. It's not the problem)
        #we won't include this as the json file isn't a problem
    except Exception as e:
        logging.error(f"Tokenizer Files Exception: {e}")
        return
    finally:  # Ensure output_dir has __init__.py
        init_file_path = os.path.join(output_dir, "__init__.py")
        if not os.path.exists(init_file_path):
            open(init_file_path, 'a').close()
            logging.info(f"Created empty __init__.py in {output_dir}")

    logging.info("Conversion complete!")

def main():
    # Hardcoded paths (original first script)
    gguf_file_path = "./source_test.gguf"
    output_directory = "./converted_model"
    convert_gguf_to_safetensors(gguf_file_path, output_directory)

if __name__ == "__main__":
    main()