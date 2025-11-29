"""
Inference script for MoE model trained on 8xH100
Loads a checkpoint and generates text
"""
import torch
import torch.nn as nn
from transformers import AutoTokenizer
from models.moe_llm import MoEMinimalLLM


def load_checkpoint(checkpoint_path, device='cuda'):
    """Load model from checkpoint"""
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    config = checkpoint['config']
    model = MoEMinimalLLM(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"✅ Model loaded (Step {checkpoint.get('step', 'unknown')})")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    if 'metrics' in checkpoint:
        metrics = checkpoint['metrics']
        print(f"   Val Loss: {metrics.get('val_loss', 'N/A'):.4f}")
        print(f"   Val PPL: {metrics.get('val_perplexity', 'N/A'):.2f}")
    
    return model, config


def generate_text(model, tokenizer, prompt, max_length=100, temperature=1.0, top_k=50, device='cuda'):
    """Generate text from a prompt"""
    model.eval()
    
    # Encode prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    generated = input_ids
    
    with torch.no_grad():
        for _ in range(max_length):
            # Forward pass
            output = model(generated)
            if isinstance(output, tuple):
                logits = output[0]
            else:
                logits = output
            
            # Get next token logits and apply temperature
            next_token_logits = logits[:, -1, :] / temperature
            
            # Top-k sampling
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Sample from the distribution
            probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to generated sequence
            generated = torch.cat([generated, next_token], dim=1)
            
            # Stop if EOS token is generated
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    return tokenizer.decode(generated[0], skip_special_tokens=True)


def main():
    # Configuration
    checkpoint_path = "./checkpoints/final_model_8xh100.pt"  # Final checkpoint
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "HuggingFaceTB/SmolLM-135M",
        cache_dir="./hf_cache"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model, config = load_checkpoint(checkpoint_path, device)
    
    # Interactive generation
    print("\n" + "="*70)
    print("🤖 MoE Model Inference - Type 'quit' to exit")
    print("="*70)
    
    prompts = [
        "The future of artificial intelligence",
        "Once upon a time, in a distant galaxy",
        "The key to success is",
        "In the field of medicine,",
    ]
    
    print("\n📝 Generating samples from preset prompts...\n")
    for prompt in prompts:
        print(f"Prompt: {prompt}")
        generated = generate_text(
            model, tokenizer, prompt, 
            max_length=80, 
            temperature=0.8, 
            top_k=40,
            device=device
        )
        print(f"Generated: {generated}")
        print("-" * 70)
    
    # Interactive mode
    print("\n💬 Interactive mode - Enter your prompts:\n")
    while True:
        try:
            prompt = input("\nPrompt (or 'quit'): ").strip()
            if prompt.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not prompt:
                continue
            
            generated = generate_text(
                model, tokenizer, prompt,
                max_length=100,
                temperature=0.8,
                top_k=40,
                device=device
            )
            print(f"\n{generated}\n")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break


if __name__ == "__main__":
    main()
