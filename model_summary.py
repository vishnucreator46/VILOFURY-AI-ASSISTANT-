# vilofury_parameters.py
try:
    import torch
    from transformers import AutoModelForCausalLM
    print("🔹 Framework: PyTorch detected")

    # Load the fine-tuned Vilofury model
    model = AutoModelForCausalLM.from_pretrained("./vilofury_finetuned")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n🧠 Total parameters: {total_params:,}")
    print(f"⚙️ Trainable parameters: {trainable_params:,}")

except ImportError:
    try:
        from tensorflow.keras.models import load_model
        print("🔹 Framework: TensorFlow / Keras detected")

        # Load your Vilofury model here 👇
        # Example: model = load_model("vilofury_model.h5")
        model = ...  # <-- Replace this with your Vilofury model

        model.summary()

    except ImportError:
        try:
            from transformers import AutoModel
            print("🔹 Framework: Hugging Face Transformers detected")

            # Load your Vilofury model here 👇
            # Example: model = AutoModel.from_pretrained("Vilofury")
            model = ...  # <-- Replace this with your Vilofury model

            print(f"\n🧠 Total parameters: {model.num_parameters():,}")

        except Exception as e:
            print("❌ Could not detect model framework or load Vilofury.")
            print("Error:", e)

