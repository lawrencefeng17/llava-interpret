import contextlib
import requests
import torch
import transformer_lens
import transformers
from tempfile import TemporaryDirectory
from PIL import Image

es = contextlib.ExitStack()
es.enter_context(torch.inference_mode())

model_name = "Intel/llava-gemma-2b"
model = transformers.AutoModelForImageTextToText.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
)
processor = transformers.AutoProcessor.from_pretrained(model_name)
processor.patch_size = model.config.vision_config.patch_size
processor.vision_feature_select_strategy = model.config.vision_feature_select_strategy

def get_hooked_model(model, tokenizer):
    with TemporaryDirectory() as model_name:
        model.config.save_pretrained(model_name)
        cfg = transformer_lens.loading.get_pretrained_model_config(
            model_name,
            device=model.device,
            dtype=model.dtype,
        )
        state_dict = transformer_lens.loading.get_pretrained_state_dict(
            model_name,
            cfg,
            model,
        )
        for k, v in state_dict.items():
            if v.device != model.device:
                state_dict[k] = v.to(model.device)
    hooked_model = transformer_lens.HookedTransformer(cfg, tokenizer)
    hooked_model.load_and_process_state_dict(
        state_dict,
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
        fold_value_biases=False,
    )
    return hooked_model

hooked_model = get_hooked_model(model.language_model, processor.tokenizer)

def get_input_embeds(input_ids, pixel_values):
    input_embeds = model.get_input_embeddings()(input_ids)
    image_features = model.get_image_features(
        pixel_values,
        model.config.vision_feature_layer,
        model.config.vision_feature_select_strategy,
    )
    input_embeds[input_ids == model.config.image_token_index] = image_features
    return input_embeds

def run_model(prompt, image):
    inp = processor(image, prompt, return_tensors="pt").to(model.device)
    input_embeds = get_input_embeds(inp.input_ids, inp.pixel_values)
    input_embeds *= hooked_model.cfg.d_model ** 0.5
    return hooked_model.run_with_cache(input_embeds, start_at_layer=0)

conversation = [{"role": "user", "content": processor.image_token + "\ndescribe the image"}]
prompt = processor.tokenizer.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)

image = requests.get("https://llava-vl.github.io/static/images/view.jpg", stream=True)
image = Image.open(image.raw)

import sys
mode = "C"

if mode == "A":
    logits, activations = run_model(prompt, image)
    print(logits)
    print(activations)
else:
    inp = processor(image, prompt, return_tensors="pt").to(model.device)
    input_embeds = get_input_embeds(inp.input_ids, inp.pixel_values)

    if mode == "B":
        while True:
            next_token = model.language_model(inputs_embeds=input_embeds, use_cache=False).logits[0, -1].argmax()
            print(end=processor.tokenizer.decode(next_token), flush=True)
            input_embeds = torch.cat([input_embeds, model.get_input_embeddings()(next_token).unsqueeze(0).unsqueeze(0)], 1)
    elif mode == "C":
        input_embeds *= hooked_model.cfg.d_model ** 0.5
        while True:
            next_token = hooked_model(input_embeds, start_at_layer=0)[0, -1].argmax()
            print(end=processor.tokenizer.decode(next_token), flush=True)
            input_embeds = torch.cat([input_embeds, hooked_model.embed(next_token).unsqueeze(0).unsqueeze(0)], 1)
