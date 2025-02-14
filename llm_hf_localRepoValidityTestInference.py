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
                #pad_token_id=tokenizer.pad_token_id,  # Specify pad_token_id if needed
                #eos_token_id=tokenizer.eos_token_id,   # Specify eos_token_id if needed
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


if __name__ == "__main__":
    model_directory = "./converted_model"  # Replace with the actual path
    user_prompt = "Explain the importance of low latency LLMs"

    # --- Run Inference (GPU, if available) ---
    print("Running inference on GPU (if available)...")
    generated_output = perform_inference(model_directory, user_prompt)
    print(f"Generated Text (GPU/Default):\n{generated_output}\n")


    # --- Run Inference (CPU) ---
    print("Running inference on CPU...")
    generated_output_cpu = perform_inference(model_directory, user_prompt, use_cpu=True)
    print(f"Generated Text (CPU):\n{generated_output_cpu}")