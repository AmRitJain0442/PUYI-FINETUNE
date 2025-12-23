"""
Upload trained LoRA model to Hugging Face Hub
"""
import os
from pathlib import Path

def upload_to_huggingface():
    """Upload the trained model to Hugging Face Hub"""
    
    # Check if huggingface_hub is installed
    try:
        from huggingface_hub import HfApi, login
    except ImportError:
        print("Installing huggingface_hub...")
        import subprocess
        subprocess.check_call(["pip", "install", "huggingface_hub"])
        from huggingface_hub import HfApi, login
    
    # Model path
    model_path = Path("LLaMA-Factory/saves/ERNIE-4.5-0.3B-PT/lora/train_2025-12-23-16-11-58")
    
    if not model_path.exists():
        print(f"Error: Model path not found: {model_path}")
        return
    
    print(f"Found model at: {model_path}")
    print("\nModel files:")
    for file in sorted(model_path.glob("*")):
        if file.is_file():
            size_mb = file.stat().st_size / (1024 * 1024)
            print(f"  - {file.name} ({size_mb:.2f} MB)")
    
    # Get HF token
    print("\n" + "="*60)
    print("HUGGING FACE AUTHENTICATION")
    print("="*60)
    print("\nOptions:")
    print("1. Enter your Hugging Face token manually")
    print("2. Login via browser (huggingface-cli login)")
    print("\nTo get a token:")
    print("  - Go to: https://huggingface.co/settings/tokens")
    print("  - Create a token with 'write' access")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    if choice == "1":
        token = input("\nEnter your Hugging Face token: ").strip()
        if token:
            login(token=token)
            print("✓ Logged in successfully!")
        else:
            print("Error: No token provided")
            return
    elif choice == "2":
        print("\nRunning: huggingface-cli login")
        import subprocess
        subprocess.run(["huggingface-cli", "login"])
    else:
        print("Invalid choice")
        return
    
    # Get repository details
    print("\n" + "="*60)
    print("REPOSITORY DETAILS")
    print("="*60)
    
    username = input("\nEnter your Hugging Face username: ").strip()
    if not username:
        print("Error: Username required")
        return
    
    default_repo_name = "puyi-memories-lora"
    repo_name = input(f"Enter repository name [{default_repo_name}]: ").strip()
    if not repo_name:
        repo_name = default_repo_name
    
    repo_id = f"{username}/{repo_name}"
    
    # Model description
    is_private = input("\nMake repository private? (y/n) [n]: ").strip().lower()
    private = is_private == 'y'
    
    # Create README content
    readme_content = f"""---
license: apache-2.0
base_model: Ernie-4.5-0.3B
tags:
- lora
- finetuned
- personal-memories
- chinese-history
library_name: peft
---

# {repo_name}

This is a LoRA (Low-Rank Adaptation) adapter for the Ernie-4.5-0.3B model, fine-tuned on historical memories and personal narratives.

## Model Details

- **Base Model:** Ernie-4.5-0.3B
- **Training Method:** LoRA (Low-Rank Adaptation)
- **Training Framework:** LLaMA-Factory
- **Training Date:** December 23, 2025
- **Dataset:** Puyi historical memories and narratives

## Usage

### With LLaMA-Factory

```bash
# Inference
python src/cli_demo.py \\
    --model_name_or_path /path/to/base/model \\
    --adapter_name_or_path {repo_id} \\
    --template default
```

### With PEFT

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained("Ernie-4.5-0.3B")
tokenizer = AutoTokenizer.from_pretrained("Ernie-4.5-0.3B")

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, "{repo_id}")

# Generate
inputs = tokenizer("Your prompt here", return_tensors="pt")
outputs = model.generate(**inputs, max_length=200)
print(tokenizer.decode(outputs[0]))
```

## Training Details

See `training_args.yaml` and `all_results.json` for complete training configuration and results.

## Files

- `adapter_model.safetensors` - LoRA adapter weights
- `adapter_config.json` - Adapter configuration
- `tokenizer*` - Tokenizer files
- Training logs and results

## License

Apache 2.0
"""
    
    # Save README to model directory
    readme_path = model_path / "README_HF.md"
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content)
    print(f"\n✓ Created README at: {readme_path}")
    
    # Upload
    print("\n" + "="*60)
    print("UPLOADING TO HUGGING FACE")
    print("="*60)
    print(f"\nRepository: {repo_id}")
    print(f"Private: {private}")
    print(f"\nUploading files from: {model_path}")
    
    confirm = input("\nProceed with upload? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Upload cancelled")
        return
    
    try:
        api = HfApi()
        
        # Create repository
        print("\n1. Creating repository...")
        api.create_repo(
            repo_id=repo_id,
            repo_type="model",
            private=private,
            exist_ok=True
        )
        print(f"✓ Repository created/verified: https://huggingface.co/{repo_id}")
        
        # Upload folder
        print("\n2. Uploading files...")
        api.upload_folder(
            folder_path=str(model_path),
            repo_id=repo_id,
            repo_type="model",
            commit_message="Upload trained LoRA adapter"
        )
        
        print("\n" + "="*60)
        print("✓ UPLOAD COMPLETE!")
        print("="*60)
        print(f"\nYour model is available at:")
        print(f"https://huggingface.co/{repo_id}")
        print("\nYou can now:")
        print("1. View your model on Hugging Face")
        print("2. Share the model with others")
        print("3. Use it in your projects with the code examples above")
        
    except Exception as e:
        print(f"\n✗ Error during upload: {e}")
        print("\nTroubleshooting:")
        print("1. Check your token has 'write' permissions")
        print("2. Verify your username is correct")
        print("3. Try running: huggingface-cli login")

if __name__ == "__main__":
    upload_to_huggingface()
