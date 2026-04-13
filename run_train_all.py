import os
from models_config import MODEL_REGISTRY

def main():
    for name, model_class in MODEL_REGISTRY.items():
        # Initialize model - this now creates outputs/model_name/ via BaseModel
        m = model_class()
        
        # --- FIX: Handle different weight naming conventions ---
        # LightGBM uses nn_embedding.pth
        # Other models might use .ckpt or .pth
        if "LightGBM" in m.model_name:
            possible_files = ["nn_embedding.pth"]
        else:
            possible_files = [f"{m.model_name}.ckpt", f"{m.model_name}.pth"]

        weights_path = None
        # Check which of the possible files actually exists
        for check_file in possible_files:
            temp_path = os.path.join(m.output_dir, check_file)
            if os.path.exists(temp_path):
                weights_path = temp_path
                break
        
        # If none exist, default to the first one for logging purposes
        if weights_path is None:
            weights_path = os.path.join(m.output_dir, possible_files[0])

        print(f"\n--- Checking Model: {name} ---")
        print(f"Target Directory: {m.output_dir}")

        # Skip logic: now checks the correct subfolder and multiple extensions
        if os.path.exists(weights_path):
            print(f"⏭️  Weights found at {weights_path}. Skipping training phase.")
            continue
        
        print(f"No weights found. Starting training for {name}...")
        
        # 1. Load the 815MB raw data cache
        m.load_and_split_data() 
        
        # 2. Build the model-specific tensors/datasets
        m.preprocess() 
        
        # 3. Execute training
        # We can dynamically set epochs based on the model name
        epochs = 5 if ("lstm" in name.lower() or "lightgbm" in name.lower()) else 50
        m.train(epochs=epochs)
        
        print(f"✅ {name} training complete. Weights saved in {m.output_dir}")

if __name__ == "__main__":
    main()