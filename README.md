# llava-interpret

To main bulk of work is reproducible within the llava_sae notebook. The notebook

1. Loads a hooked version of LLaVA so that we can explore and intervene on model activations in the resiudal stream
2. Loads a pretrained SAE for Gemma-2B using SAELens
3. We run the model on imagenet images and the prompt "describe the image."
4. From there, we can use the pretrained SAE to analyze the activations
5. We also provide a function to choose an interpretable feature and subtract that from the residual stream in the original LLaVA model. We include some examples in the notebook.
