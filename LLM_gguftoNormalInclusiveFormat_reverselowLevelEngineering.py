import os
import struct
import torch
import numpy as np
from safetensors.torch import save_file
from gguf import dequantize
from typing import Dict, Optional
from transformers import AutoConfig, PretrainedConfig
import logging
from typing import BinaryIO, List, Any, Tuple
import json
import sentencepiece as spm
import tempfile  # Import tempfile

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
    if tensor_type not in GGML_QUANT_SIZES:
        raise ValueError(f"Unsupported GGML tensor type: {tensor_type}")

    block_size, type_size = GGML_QUANT_SIZES[tensor_type]

    if tensor_type == 0:
        return np.frombuffer(data, dtype=np.float32)
    elif tensor_type == 1:
        return np.frombuffer(data, dtype=np.float16).astype(np.float32)
    else:
        return dequantize(np.frombuffer(data, dtype=np.uint8), tensor_type)

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

def create_tokenizer_model_from_json(
    tokenizer_json_path: str,
    special_tokens_map_path: str,
    tokenizer_config_path: str,
    output_model_path: str = "tokenizer.model",
):
    with open(tokenizer_json_path, "r", encoding="utf-8") as f:
        tokenizer_json = json.load(f)
    with open(special_tokens_map_path, "r", encoding="utf-8") as f:
        special_tokens_map = json.load(f)
    with open(tokenizer_config_path, "r", encoding="utf-8") as f:
        tokenizer_config = json.load(f)

    vocab = tokenizer_json["model"]["vocab"]
    scores = tokenizer_json["model"].get("scores", [])
    if len(scores) < len(vocab):
        scores.extend([0.0] * (len(vocab) - len(scores)))
    elif len(scores) > len(vocab):
        scores = scores[:len(vocab)]

    tokenizer_type = tokenizer_json["model"]["type"].lower()

    model = spm.SentencePieceModel()

    for token, score in zip(vocab, scores):
        piece = model.pieces.add()
        piece.piece = token
        piece.score = float(score)
        piece.type = 1

    unk_id = tokenizer_config.get("unk_token")
    bos_id = tokenizer_config.get("bos_token")
    eos_id = tokenizer_config.get("eos_token")
    pad_id = tokenizer_config.get("pad_token")

    if unk_id is not None:
      if isinstance(unk_id, dict):
          unk_id_value = tokenizer_json["model"]["vocab"].get(unk_id["content"])
      else:
        unk_id_value = tokenizer_json["model"]["vocab"].get(unk_id)

      if unk_id_value is not None:
        model.unk_id = unk_id_value
        model.pieces[unk_id_value].type = 0
      else:
        logging.warning(f"unk_id {unk_id} not found")

    if bos_id is not None:
      if isinstance(bos_id, dict):
        bos_id_value = tokenizer_json["model"]["vocab"].get(bos_id["content"])
      else:
        bos_id_value = tokenizer_json["model"]["vocab"].get(bos_id)

      if bos_id_value is not None:
        model.bos_id = bos_id_value
        model.pieces[bos_id_value].type = 1
      else:
        logging.warning(f"bos_id {bos_id} not found")

    if eos_id is not None:
      if isinstance(eos_id, dict):
        eos_id_value = tokenizer_json["model"]["vocab"].get(eos_id["content"])
      else:
        eos_id_value = tokenizer_json["model"]["vocab"].get(eos_id)

      if eos_id_value is not None:
        model.eos_id = eos_id_value
        model.pieces[eos_id_value].type = 2
      else:
        logging.warning(f"eos_id {eos_id} not found")

    if pad_id is not None:
      if isinstance(pad_id, dict):
        pad_id_value = tokenizer_json["model"]["vocab"].get(pad_id["content"])
      else:
        pad_id_value = tokenizer_json["model"]["vocab"].get(pad_id)
      if pad_id_value is not None:
            model.pieces[pad_id_value].type = 3

    for token_name, token_info in special_tokens_map.items():
        token_content = token_info["content"]
        if token_content in vocab:
            token_id = vocab[token_content]
            model.pieces[token_id].type = 2 #control

    if "merges" in tokenizer_json["model"] and tokenizer_type == "bpe":
        merges = tokenizer_json["model"]["merges"]

        def process_merge(merge_str: str) -> Optional[str]:
          parts = merge_str.split(" ")
          if len(parts) != 2:
            return None

          new_parts = []
          for part in parts:
            if part.startswith("<0x") and part.endswith(">"):
                try:
                    byte_value = int(part[4:-1], 16)
                    new_parts.append(chr(byte_value))
                except ValueError:
                    return None
            else:
              new_parts.append(part.replace("Ġ", " "))
          return " ".join(new_parts)

        for merge in merges:
          processed_merge = process_merge(merge)
          if processed_merge:
            model.pieces.add(piece=processed_merge, score=0.0, type=6) #type 6 for merges
    else:
        logging.warning(
            "No merges found in tokenizer.json, or tokenizer type is not BPE. "
            "The recreated tokenizer.model will only work at the character/byte level."
        )
        if tokenizer_type != "bpe":
            pass
        else:
          for i in range(256):
            for j in range(256):
              model.pieces.add(piece = chr(i) + " " + chr(j), score = 0.0, type = 6)

    with open(output_model_path, "wb") as f:
        f.write(model.SerializeToString())

    print(f"tokenizer.model created at {output_model_path}")

def convert_gguf_to_safetensors(gguf_path: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(TEMP_DIR, exist_ok=True)  # Create the temporary directory

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

            # No longer store tensors in a dictionary directly
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

                # Write tensor data to a temporary file
                with tempfile.NamedTemporaryFile(dir=TEMP_DIR, delete=False) as temp_file:
                    torch.save(torch.from_numpy(dequantized_data).to(dtype=torch.float16), temp_file) #save it to temp first
                    tensor_files[name] = temp_file.name  # Store the file path


        # Now, build the SafeTensors dictionary by loading from the temporary files
        tensors_dict = {}
        for name, filepath in tensor_files.items():
            tensors_dict[name] = torch.load(filepath)  # Load tensor
            os.remove(filepath)  # And delete the temporary file

        safetensors_path = os.path.join(output_dir, "model.safetensor")
        try:
            save_file(tensors_dict, safetensors_path)
            logging.info(f"SafeTensors file saved to: {safetensors_path}")
        except Exception as e:
            logging.error(f"Failed to save SafeTensors file: {e}")
            return
        # Clean up any remaining temporary files (shouldn't be any, but just in case)
        for name, filepath in tensor_files.items():
            if os.path.exists(filepath):
                os.remove(filepath)

    except Exception as e:
        logging.error(f"Error during GGUF parsing: {e}")
        return


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
        if not any([n_embd, n_head, n_layer, n_positions]):
            logging.warning("No essential parameters")

        try:
            config = AutoConfig.from_pretrained(arch, trust_remote_code=True)
            if n_embd: config.hidden_size = n_embd
            if n_head: config.num_attention_heads = n_head
            if n_layer: config.num_hidden_layers = n_layer
            if n_positions: config.max_position_embeddings = n_positions
            if rope_dimension_count: config.rope_theta = rope_dimension_count
            config.tokenizer_class = arch

        except Exception as e:
            logging.warning(f"AutoConfig failed ({arch}): {e}. Creating manually...")
            config = PretrainedConfig()
            config.architectures = [arch]
            if n_embd: config.hidden_size = n_embd
            if n_head: config.num_attention_heads = n_head
            if n_layer: config.num_hidden_layers = n_layer
            if n_positions: config.max_position_embeddings = n_positions
            if rope_dimension_count: config.rope_theta = rope_dimension_count
            config.tokenizer_class = arch

        config.save_pretrained(output_dir)
        logging.info(f"config.json saved to: {config_path}")

    except Exception as e:
        logging.error(f"Error creating config.json: {e}")
        return

    generation_config_path = os.path.join(output_dir, "generation_config.json")
    try:
        with open(generation_config_path, "w") as f:
            f.write("{}\n")
        logging.info(f"generation_config.json saved to: {generation_config_path}")
    except Exception as e:
        logging.error(f"Failed to create generation_config.json: {e}")
        return

    tokenizer_dir = output_dir
    tokenizer_config_path = os.path.join(tokenizer_dir, "tokenizer_config.json")
    tokenizer_model_path = os.path.join(tokenizer_dir, "tokenizer.model")
    tokenizer_json_path = os.path.join(tokenizer_dir, "tokenizer.json")
    special_tokens_map_path = os.path.join(tokenizer_dir, "special_tokens_map.json")

    try:
        tokenizer_model_gguf = metadata.get("tokenizer.ggml.model", "unknown")
        if isinstance(tokenizer_model_gguf, bytes):
            tokenizer_model_gguf = tokenizer_model_gguf.decode("utf-8", errors="replace")

        tokenizer_type = metadata.get("tokenizer.ggml.model", "BPE")
        if isinstance(tokenizer_type, bytes):
            tokenizer_type = tokenizer_type.decode("utf-8", errors="replace")

        # --- special_tokens_map.json ---
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
        # --- tokenizer_config.json ---
        try:
            tokens = metadata.get("tokenizer.ggml.tokens", [])
            if isinstance(tokens, list):
                tokens = [t.decode("utf-8", errors="replace") if isinstance(t, bytes) else t for t in tokens]
            else:
                tokens = []
                logging.warning("tokenizer.ggml.tokens did not return a list.")


            tokenizer_config = {
              "tokenizer_class": arch.capitalize() + "Tokenizer" if tokenizer_model_gguf == "unknown" else tokenizer_model_gguf.capitalize() + "Tokenizer",
            }

            if "tokenizer.chat_template" in metadata:
                tokenizer_config["chat_template"] = metadata["tokenizer.chat_template"]
            if "tokenizer.ggml.add_bos_token" in metadata:
                tokenizer_config["add_bos_token"] = metadata["tokenizer.ggml.add_bos_token"]
            if "tokenizer.ggml.add_eos_token" in metadata:
                tokenizer_config["add_eos_token"] = metadata["tokenizer.ggml.add_eos_token"]

            for key in ["bos_token", "eos_token", "unk_token", "pad_token"]:
                if key in special_tokens_map:
                    tokenizer_config[key] = special_tokens_map[key]["content"]

            if tokens:
                added_tokens_decoder = {}
                for i, token in enumerate(tokens):
                    token_info = {
                        "content": token,
                        "lstrip": False,
                        "normalized": False,
                        "rstrip": False,
                        "single_word": False,
                        "special": False,
                    }

                    if any(metadata.get(f"tokenizer.ggml.{name}_token_id") == i for name in ["bos", "eos", "unknown", "padding"]):
                        token_info["special"] = True
                    for key, value in metadata.items():
                      if "tokenizer.ggml" not in key:
                          continue
                      if "token_id" in key and value == i:
                            token_info["special"] = True
                            break

                    token_types = metadata.get("tokenizer.ggml.token_type")
                    if isinstance(token_types, list) and len(token_types) == len(tokens) and tokenizer_type.lower() == "llama":
                        type_map = {
                            1: "Unknown",
                            2: "Control",
                            3: "UserDefined",
                            6: "Byte",
                        }
                        if token_types[i] in type_map:
                            if type_map[token_types[i]] == "UserDefined":
                                token_info["special"] = True
                                token_info["normalized"] = False
                            elif type_map[token_types[i]] == "Byte":
                                token_info["normalized"] = False

                    added_tokens_decoder[str(i)] = token_info

                tokenizer_config["added_tokens_decoder"] = added_tokens_decoder

            with open(tokenizer_config_path, "w") as f:
                json.dump(tokenizer_config, f, indent=2)
            logging.info(f"tokenizer_config.json saved to: {tokenizer_config_path}")

        except Exception as e:
            logging.error(f"Failed to create tokenizer_config.json: {e}")
            return

        # --- tokenizer.json ---
        try:
            tokens = metadata.get("tokenizer.ggml.tokens", [])
            if isinstance(tokens, list):
                tokens = [t.decode("utf-8", errors="replace") if isinstance(t, bytes) else t for t in tokens]
            else:
                tokens = []
                logging.warning("tokenizer.ggml.tokens did not return a list.")

            if tokens:
                vocab = {token: i for i, token in enumerate(tokens)}
                tokenizer_json_content = {
                    "version": "1.0",
                    "truncation": None,
                    "padding": None,
                    "added_tokens": [],
                    "normalizer": None,
                    "pre_tokenizer": None,
                    "post_processor": None,
                    "decoder": None,
                    "model": {
                        "type": tokenizer_type,
                        "vocab": vocab,
                        "merges": [],
                    }
                }

                added_tokens_list = []
                for i, token in enumerate(tokens):
                    token_info = {
                        "id": i,
                        "content": token,
                        "single_word": False,
                        "lstrip": False,
                        "rstrip": False,
                        "normalized": False,
                        "special": False,
                    }

                    if any(metadata.get(f"tokenizer.ggml.{name}_token_id") == i for name in ["bos", "eos", "unknown", "padding"]):
                        token_info["special"] = True
                    for key, value in metadata.items():
                      if "tokenizer.ggml" not in key:
                        continue
                      if "token_id" in key and value == i:
                            token_info["special"] = True
                            break

                    added_tokens_list.append(token_info)
                tokenizer_json_content["added_tokens"] = added_tokens_list

                scores = metadata.get("tokenizer.ggml.scores")
                if isinstance(scores, list) and len(scores) == len(tokens):
                    tokenizer_json_content["model"]["scores"] = scores

                token_types = metadata.get("tokenizer.ggml.token_type")
                if isinstance(token_types, list) and len(token_types) == len(tokens) and tokenizer_type.lower() == "llama":
                        type_map = {
                            1: "Unknown",
                            2: "Control",
                            3: "UserDefined",
                            6: "Byte",
                        }

                        added_tokens_list = []
                        for i, token_type in enumerate(token_types):
                            token_info = {
                                "id": i,
                                "content": tokens[i],
                                "single_word": False,
                                "lstrip": False,
                                "rstrip": False,
                                "normalized": False,
                                "special": False,
                            }

                            if token_type in type_map:
                                if type_map[token_type] == "UserDefined":
                                    token_info["special"] = True
                                    token_info["normalized"] = False
                                elif type_map[token_type] == "Byte":
                                    token_info["normalized"] = False

                            if any(metadata.get(f"tokenizer.ggml.{name}_token_id") == i for name in ["bos", "eos", "unknown", "padding"]):
                                token_info["special"] = True

                            for key, value in metadata.items():
                              if "tokenizer.ggml" not in key:
                                continue
                              if "token_id" in key and value == i:
                                    token_info["special"] = True
                                    break

                            added_tokens_list.append(token_info)
                        tokenizer_json_content["added_tokens"] = added_tokens_list
                if "tokenizer.ggml.merges" in metadata:
                    merges = metadata.get("tokenizer.ggml.merges")
                    if isinstance(merges, list):
                         tokenizer_json_content["model"]["merges"] = [
                            m.decode("utf-8", errors="replace") if isinstance(m, bytes) else m for m in merges
                        ]
                    else:
                        logging.warning("tokenizer.ggml.merges is not a list")

                with open(tokenizer_json_path, "w") as f:
                    json.dump(tokenizer_json_content, f, indent=2)
                logging.info(f"tokenizer.json (minimal vocabulary) saved to: {tokenizer_json_path}")
            else:
                with open(tokenizer_json_path, "w") as f:
                    json.dump({}, f)
                logging.warning("tokenizer.json is empty, tokens extraction failed.")
                logging.info(f"tokenizer.json (EMPTY) saved to: {tokenizer_json_path}")

        except Exception as e:
            logging.error(f"Failed to create tokenizer.json: {e}")
            return

        # --- tokenizer.model (Recreation) ---
        try:
          create_tokenizer_model_from_json(
                tokenizer_json_path,
                special_tokens_map_path,
                tokenizer_config_path,
                tokenizer_model_path
            )
          # --- Self-Test of tokenizer.model ---
          print("Testing tokenizer.model...")
          sp = spm.SentencePieceProcessor()
          sp.load(tokenizer_model_path)

          print(f"bos id: {sp.bos_id()}")
          print(f"eos id: {sp.eos_id()}")
          print(f"unk id: {sp.unk_id()}")
          print(f"pad id: {sp.pad_id()}")

          test_sentences = [
              "This is a test sentence.",
              "Another test.",
              "<s> This should have special tokens. </s>",
              "Out-of-vocab-word",
          ]
          for sentence in test_sentences:
              print(f"Original: {sentence}")
              print(f"  Pieces: {sp.encode_as_pieces(sentence)}")
              print(f"  IDs: {sp.encode_as_ids(sentence)}")
        except Exception as e:
          logging.error(f"Failed to create tokenizer.model {e}")
          return

    except Exception as e:
        logging.error(f"Tokenizer Files Exception: {e}")
        return

    logging.info("Conversion complete!")

def main():
    gguf_file_path = "./source_test.gguf"
    output_directory = "./converted_model"
    convert_gguf_to_safetensors(gguf_file_path, output_directory)

if __name__ == "__main__":
    main()