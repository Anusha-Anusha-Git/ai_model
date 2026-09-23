import time
import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


def load_model():
    print(f"Loading {MODEL_NAME} ...")
    start = time.time()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)

    elapsed = time.time() - start
    print(f"Model loaded in {elapsed:.2f} seconds")
    return model, tokenizer


def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 200):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    start = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False
        )
    elapsed = time.time() - start

    response = tokenizer.decode(
        output_ids[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True
    )

    tokens_generated = output_ids.shape[-1] - inputs["input_ids"].shape[-1]
    print(f"Generated {tokens_generated} tokens in {elapsed:.2f}s "
          f"({tokens_generated/elapsed:.2f} tokens/sec)")

    return response


if __name__ == "__main__":
    process = psutil.Process()
    mem_before = process.memory_info().rss / (1024 ** 2)

    model, tokenizer = load_model()

    mem_after = process.memory_info().rss / (1024 ** 2)
    print(f"RAM used by model load: {mem_after - mem_before:.2f} MB")

    if torch.cuda.is_available():
        vram_used = torch.cuda.memory_allocated() / (1024 ** 2)
        vram_total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
        print(f"VRAM used by model: {vram_used:.2f} MB / {vram_total:.2f} MB total")

    test_prompt = "Explain in one sentence what a helical pile is used for in foundation engineering."
    print("\nPrompt:", test_prompt)
    response = generate_response(model, tokenizer, test_prompt)
    print("\nResponse:", response)
