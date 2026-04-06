import os
from models_config import MODEL_REGISTRY

def main():
    for name, model_class in MODEL_REGISTRY.items():
        # Define the expected weight path
        # Note: BaseModel creates the 'outputs' folder automatically
        m = model_class()
        weights_path = os.path.join(m.output_dir, f"{m.model_name}.pth")

        print(f"\n--- Checking Model: {name} ---")

        # Skip logic: check if the .pth file exists
        if os.path.exists(weights_path):
            print(f"⏭️  Weights found at {weights_path}. Skipping training phase.")
            continue
        
        print(f"🚀 No weights found. Starting training for {name}...")
        
        # 1. Load the 815MB raw data cache
        m.load_and_split_data() 
        
        # 2. Build the model-specific tensors/datasets
        m.preprocess() 
        
        # 3. Execute training (5 epochs for LSTM, 50 for Linear)
        epochs = 5 if "lstm" in name else 50
        m.train(epochs=epochs)
        
        print(f"✅ {name} training complete and weights saved.")

if __name__ == "__main__":
    main()