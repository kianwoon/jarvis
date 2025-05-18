import sys
from scripts.download_model import download_model

if __name__ == "__main__":
    print("Starting Qwen model download...")
    print("This will download the Qwen-7B-Chat model (approximately 7GB)")
    print("Press Ctrl+C to cancel")
    
    try:
        model, tokenizer = download_model()
        print("\nModel download completed successfully!")
        print(f"Model size: {model.get_memory_footprint() / 1024**3:.2f} GB")
        print(f"Model device: {model.device}")
    except KeyboardInterrupt:
        print("\nDownload interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\nDownload failed: {str(e)}")
        sys.exit(1) 