import torch
from PIL import Image
from tqdm import tqdm
import transformers

# Import necessary functions from llava_sae.py
from llava_sae import get_acts, show_top, model, sae, prompt, processor, run_model

# Load the cat image
cat_image = Image.open("image.png")

# Get activations for the cat image
def get_cat_image_acts(idx=586):
    """
    Get activations for a single cat image at a specific index in the SAE
    
    Args:
        idx: The index in the SAE to get activations for (default: 586)
    
    Returns:
        Tensor of activations
    """
    acts = torch.zeros(sae.cfg.d_sae, device=model.device)
    # For a single image, we don't need to sum multiple activations
    acts = sae.encode(run_model(prompt, cat_image, stop_at_layer=sae.cfg.hook_layer + 1)[1][sae.cfg.hook_name])[0, idx]
    return acts

# Get the activations
cat_acts = get_cat_image_acts()

# Show the top activations
print("Top activations for cat_image:")
# Use the show_top function to display the top activations
show_top(cat_acts)

# Alternative approach: get all activations across all SAE features
def get_all_cat_image_acts():
    """
    Get activations for all features in the SAE for the cat image
    
    Returns:
        Tensor of all activations
    """
    hidden_states = run_model(prompt, cat_image, stop_at_layer=sae.cfg.hook_layer + 1)[1][sae.cfg.hook_name]
    all_acts = sae.encode(hidden_states)[0]  # Get activations for all features
    return all_acts

# Get all activations
all_cat_acts = get_all_cat_image_acts()

# Show top activations across all features
print("\nTop activations across all SAE features:")
top_k = 20  # Show top 20 activations
top_values, top_indices = torch.topk(all_cat_acts, top_k)

for i, (idx, val) in enumerate(zip(top_indices.tolist(), top_values.tolist())):
    if (d := descs.get(idx)):
        print(f"{i+1}. Feature {idx}: {val:.4f} - {d}")
    else:
        print(f"{i+1}. Feature {idx}: {val:.4f}") 