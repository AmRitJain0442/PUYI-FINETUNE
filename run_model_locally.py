"""
Run the fine-tuned ERNIE model locally with LoRA adapter
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os
from pathlib import Path

class LocalModelRunner:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def load_model(self, adapter_path=None):
        """Load the base model and LoRA adapter"""
        
        print("="*60)
        print("LOADING MODEL")
        print("="*60)
        
        # Model paths
        base_model_name = "baidu/ERNIE-4.5-0.3B-PT"
        
        if adapter_path is None:
            adapter_path = Path("LLaMA-Factory/saves/ERNIE-4.5-0.3B-PT/lora/train_2025-12-23-16-11-58")
        
        print(f"\nBase Model: {base_model_name}")
        print(f"LoRA Adapter: {adapter_path}")
        print(f"Device: {self.device}")
        
        # Load tokenizer
        print("\n1. Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            adapter_path,
            trust_remote_code=True
        )
        print("✓ Tokenizer loaded")
        
        # Load base model
        print("\n2. Loading base model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )
        print("✓ Base model loaded")
        
        # Load LoRA adapter
        print("\n3. Loading LoRA adapter...")
        self.model = PeftModel.from_pretrained(
            base_model,
            adapter_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )
        print("✓ LoRA adapter loaded")
        
        # Set to evaluation mode
        self.model.eval()
        
        print("\n" + "="*60)
        print("✓ MODEL READY")
        print("="*60)
        
    def generate(self, prompt, max_length=512, temperature=0.7, top_p=0.9):
        """Generate response from the model"""
        
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Prepare input
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if self.device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the prompt from response
        if response.startswith(prompt):
            response = response[len(prompt):].strip()
        
        return response
    
    def chat(self):
        """Interactive chat interface"""
        
        print("\n" + "="*60)
        print("CHAT MODE")
        print("="*60)
        print("\nCommands:")
        print("  - Type your message to chat")
        print("  - 'quit' or 'exit' to end")
        print("  - 'clear' to clear screen")
        print("\n" + "="*60 + "\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if user_input.lower() in ['quit', 'exit']:
                    print("\nGoodbye!")
                    break
                
                if user_input.lower() == 'clear':
                    os.system('cls' if os.name == 'nt' else 'clear')
                    continue
                
                if not user_input:
                    continue
                
                print("\nModel: ", end="", flush=True)
                response = self.generate(user_input)
                print(response)
                print()
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}\n")

def main():
    """Main function"""
    
    runner = LocalModelRunner()
    
    try:
        # Load model
        runner.load_model()
        
        # Start chat
        runner.chat()
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
