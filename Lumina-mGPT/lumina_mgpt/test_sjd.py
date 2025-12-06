import os
os.environ['CUDA_VISIBLE_DEVICES']='6'

from inference_solver import FlexARInferenceSolver
from PIL import Image
import pdb

# ******************** Image Generation ********************
model_path_='/your-ckpt/Alpha-VLLM/Lumina-mGPT-7B-512'
img_size=512
inference_solver = FlexARInferenceSolver(
    model_path=model_path_,
    precision="bf16",
    target_size=img_size,
)
w=img_size;h=img_size
max_num_new_tokens = 16
guidance_scale = 3.0
multi_token_init_scheme = 'random' # 'repeat_horizon'
image_top_k = 2000 
text_top_k = 10
guidance_scale = 3.0
prefix_token_sampler_scheme = 'speculative_jacobi' # 'jacobi', 'speculative_jacobi'
# w=512;h=512

from jacobi_iteration import renew_pipeline_sampler
print(inference_solver.__class__)
inference_solver = renew_pipeline_sampler(
    inference_solver,
    jacobi_loop_interval_l = 3,
    jacobi_loop_interval_r = (h // 16)**2 + w // 16 - 10,
    max_num_new_tokens = max_num_new_tokens,
    guidance_scale = guidance_scale,
    seed = 42,
    multi_token_init_scheme = multi_token_init_scheme,
    do_cfg=  True,
    image_top_k=image_top_k, 
    text_top_k=text_top_k,
    prefix_token_sampler_scheme = prefix_token_sampler_scheme,
)

prompt = f"Image of a dog playing water, and a waterfall is in the background."

# generated: tuple of (generated response, list of generated images)
generated = inference_solver.generate(
    images=[],
    qas=[[f"Generate an image of {w}x{h} according to the following prompt:\n{prompt}", None]],
    max_gen_len=8192,
    temperature=1.0,
    logits_processor=inference_solver.create_logits_processor(cfg=guidance_scale, image_top_k=image_top_k),
)

a1, new_image = generated[0], generated[1][0]
new_image.save('./test_sjd.png')

