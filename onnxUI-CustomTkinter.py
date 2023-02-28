import customtkinter as ctk
import functools
import PIL
from PIL import Image
from math import ceil
import numpy as np
import os
import re
import gc
import lpw_pipe
import pythread as pt

from diffusers import (
    OnnxRuntimeModel,
    OnnxStableDiffusionPipeline,
    OnnxStableDiffusionImg2ImgPipeline,
    OnnxStableDiffusionInpaintPipeline,
    OnnxStableDiffusionInpaintPipelineLegacy,
    DDIMScheduler,
    PNDMScheduler,
    LMSDiscreteScheduler,
    DDPMScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
    KDPM2DiscreteScheduler,
    KDPM2AncestralDiscreteScheduler,
    HeunDiscreteScheduler,
    DPMSolverSinglestepScheduler,
    DEISMultistepScheduler,
)

from diffusers import __version__ as _df_version
from packaging import version


def resize_and_crop(
        input_image: PIL.Image.Image,
        height: int,
        width: int
):
    input_width, input_height = input_image.size

    # nearest neighbor for upscaling
    if (input_width * input_height) < (width * height):
        resample_type = Image.NEAREST
    # lanczos for downscaling
    else:
        resample_type = Image.LANCZOS

    if height / width > input_height / input_width:
        adjust_width = int(input_width * height / input_height)
        input_image = input_image.resize((adjust_width, height),
                                         resample=resample_type)
        left = (adjust_width - width) // 2
        right = left + width
        input_image = input_image.crop((left, 0, right, height))
    else:
        adjust_height = int(input_height * width / input_width)
        input_image = input_image.resize((width, adjust_height),
                                         resample=resample_type)
        top = (adjust_height - height) // 2
        bottom = top + height
        input_image = input_image.crop((0, top, width, bottom))
    return input_image


def step_adjustment(
        unadjusted_steps,
        denoise,
        pipeline,
        scheduler,
):
    # adjust step count to account for denoise in img2img
    if pipeline == "img2img":
        steps_old = unadjusted_steps
        steps = ceil(unadjusted_steps / denoise)
        if (steps > 1000) and (scheduler == "DPMSM" or "DPMSS" or "DEIS"):
            steps_unreduced = steps
            steps = 1000
            print()
            print(
                f"Adjusting steps to account for denoise. From {steps_old} "
                f"to {steps_unreduced} steps internally."
            )
            print(
                f"Without adjustment the actual step count would be "
                f"~{ceil(steps_old * denoise)} steps."
            )
            print()
            print(
                f"INTERNAL STEP COUNT EXCEEDS 1000 MAX FOR DPMSM, DPMSS, "
                f"or DEIS. INTERNAL STEPS WILL BE REDUCED TO 1000."
            )
            print()
        else:
            print()
            print(
                f"Adjusting steps to account for denoise. From {steps_old} "
                f"to {steps} steps internally."
            )
            print(
                f"Without adjustment the actual step count would be "
                f"~{ceil(steps_old * denoise)} steps."
            )
            print()
    # adjust steps to account for legacy inpaint only using ~80% of set steps
    elif pipeline == "inpaint":
        steps_old = unadjusted_steps
        if unadjusted_steps < 5:
            steps = unadjusted_steps + 1
        elif unadjusted_steps >= 5:
            steps = int((unadjusted_steps / 0.7989) + 1)
        print()
        print(
            f"Adjusting steps for legacy inpaint. From {steps_old} "
            f"to {steps} internally."
        )
        print(
            f"Without adjustment the actual step count would be "
            f"~{int(steps_old * 0.8)} steps."
        )
        print()

    return steps


# set txt2img's pipe to use vae or textenc on cpu
def txt2img_use_cpu(
        model_path,
        provider,
        scheduler,
        textenc_on_cpu,
        vae_on_cpu
):
    if textenc_on_cpu and vae_on_cpu:
        print("Using CPU Text Encoder")
        print("Using CPU VAE")
        cputextenc = OnnxRuntimeModel.from_pretrained(
            model_path + "/text_encoder"
        )
        cpuvaedec = OnnxRuntimeModel.from_pretrained(
            model_path + "/vae_decoder"
        )
        txt2img = OnnxStableDiffusionPipeline.from_pretrained(
            model_path,
            provider=provider,
            scheduler=scheduler,
            text_encoder=cputextenc,
            vae_decoder=cpuvaedec,
            vae_encoder=None,
        )
    elif textenc_on_cpu:
        print("Using CPU Text Encoder")
        cputextenc = OnnxRuntimeModel.from_pretrained(
            model_path + "/text_encoder"
        )
        txt2img = OnnxStableDiffusionPipeline.from_pretrained(
            model_path,
            provider=provider,
            scheduler=scheduler,
            text_encoder=cputextenc,
        )
    elif vae_on_cpu:
        print("Using CPU VAE")
        cpuvaedec = OnnxRuntimeModel.from_pretrained(
            model_path + "/vae_decoder"
        )
        txt2img = OnnxStableDiffusionPipeline.from_pretrained(
            model_path,
            provider=provider,
            scheduler=scheduler,
            vae_decoder=cpuvaedec,
            vae_encoder=None,
        )
    else:
        txt2img = OnnxStableDiffusionPipeline.from_pretrained(
            model_path, provider=provider, scheduler=scheduler
        )

    return txt2img


def run_txt2img(
        prompt,
        neg_prompt,
        steps,
        guidance,
        sched_name,
        seed,
        model_name,
):
    global pipe
    global scheduler

    provider = "DmlExecutionProvider"
    model_path = os.path.join("model", model_name)

    if sched_name == "PNDM" and type(scheduler) is not PNDMScheduler:
        scheduler = PNDMScheduler.from_pretrained(
            model_path, subfolder="scheduler"
        )
    elif sched_name == "LMS" and type(scheduler) is not LMSDiscreteScheduler:
        scheduler = LMSDiscreteScheduler.from_pretrained(
            model_path, subfolder="scheduler"
        )
    elif sched_name == "DDIM" and type(scheduler) is not DDIMScheduler:
        scheduler = DDIMScheduler.from_pretrained(
            model_path, subfolder="scheduler"
        )
    elif sched_name == "DDPM" and type(scheduler) is not DDPMScheduler:
        scheduler = DDPMScheduler.from_pretrained(
            model_path, subfolder="scheduler"
        )
    elif sched_name == "Euler" and type(scheduler) is not EulerDiscreteScheduler:
        scheduler = EulerDiscreteScheduler.from_pretrained(
            model_path, subfolder="scheduler"
        )
    elif sched_name == "EulerA" and type(scheduler) is not EulerAncestralDiscreteScheduler:
        scheduler = EulerAncestralDiscreteScheduler.from_pretrained(
            model_path, subfolder="scheduler"
        )
    elif sched_name == "DPMSM" and type(scheduler) is not DPMSolverMultistepScheduler:
        scheduler = DPMSolverMultistepScheduler.from_pretrained(
            model_path, subfolder="scheduler"
        )
    elif sched_name == "DPMSS" and type(scheduler) is not DPMSolverSinglestepScheduler:
        scheduler = DPMSolverSinglestepScheduler.from_pretrained(
            model_path, subfolder="scheduler"
        )
    elif sched_name == "DEIS" and type(scheduler) is not DEISMultistepScheduler:
        scheduler = DEISMultistepScheduler.from_pretrained(
            model_path, subfolder="scheduler"
        )
    elif sched_name == "KDPM2" and type(scheduler) is not KDPM2DiscreteScheduler:
        scheduler = KDPM2DiscreteScheduler.from_pretrained(
            model_path, subfolder="scheduler"
        )
    elif sched_name == "KDPM2A" and type(scheduler) is not KDPM2AncestralDiscreteScheduler:
        scheduler = KDPM2AncestralDiscreteScheduler.from_pretrained(
            model_path, subfolder="scheduler"
        )
    elif sched_name == "Heun" and type(scheduler) is not HeunDiscreteScheduler:
        scheduler = HeunDiscreteScheduler.from_pretrained(
            model_path, subfolder="scheduler"
        )

    if pipe is None:
        pipe = txt2img_use_cpu(
            model_path,
            provider,
            scheduler,
            True,
            True
        )

    # modifying the methods in the pipeline object
    if type(pipe.scheduler) is not type(scheduler):
        pipe.scheduler = scheduler
    if version.parse(_df_version) >= version.parse("0.8.0"):
        safety_checker = None
    else:
        safety_checker = lambda images, **kwargs: (
            images,
            [False] * len(images),
        )
    pipe.safety_checker = safety_checker
    pipe._encode_prompt = functools.partial(lpw_pipe._encode_prompt, pipe)

    prompt.strip("\n")
    neg_prompt.strip("\n")

    # generate seeds for iterations
    if seed == "":
        rng = np.random.default_rng()
        seed = rng.integers(np.iinfo(np.uint32).max)
    else:
        try:
            seed = int(seed) & np.iinfo(np.uint32).max
        except ValueError:
            seed = hash(seed) & np.iinfo(np.uint32).max

    # use given seed for the first iteration
    seeds = np.array([seed], dtype=np.uint32)

    # create and parse output directory
    output_path = "output"
    os.makedirs(output_path, exist_ok=True)
    dir_list = os.listdir(output_path)
    if len(dir_list):
        pattern = re.compile(
            r"([0-9][0-9][0-9][0-9][0-9][0-9])-([0-9][0-9])\..*"
        )
        match_list = [pattern.match(f) for f in dir_list]
        next_index = max([int(m[1]) if m else -1 for m in match_list]) + 1
    else:
        next_index = 0

    neg_prompt = None if neg_prompt == "" else neg_prompt
    images = []
    time_taken = 0

    rng = np.random.RandomState(seeds[0])

    generated_image = pipe(
        prompt,
        negative_prompt=neg_prompt,
        height=512,
        width=512,
        num_inference_steps=int(steps),
        guidance_scale=int(guidance),
        eta=0,
        num_images_per_prompt=1,
        generator=rng,
    ).images

    generated_image[0].save(
        os.path.join(
            output_path,
            "testimage.png"
        ),
        optimize=True,
    )

    gc.collect()


# window
window = ctk.CTk()
window.title("OnnxDiffusersUI")
window.geometry("1280x720")

# widgets

# txt2img prompt
txt2img_prompt_label = ctk.CTkLabel(
    window,
    text="prompt",
)
txt2img_prompt = ctk.CTkTextbox(window)
txt2img_prompt_label.pack()
txt2img_prompt.pack()

# txt2img negative prompt
txt2img_neg_prompt_label = ctk.CTkLabel(
    window,
    text="negative prompt",
)
txt2img_neg_prompt_label.pack()
txt2img_neg_prompt = ctk.CTkTextbox(window)
txt2img_neg_prompt.pack()

# txt2img step count slider
txt2img_step_slider_label = ctk.CTkLabel(
    window,
    text=f"step count",
)
txt2img_step_slider = ctk.CTkSlider(
    window,
    from_=1,
    to=8,
    number_of_steps=8,
)
txt2img_step_slider_label.pack()
txt2img_step_slider.pack()

# txt2img guidance scale slider
txt2img_guidance_slider_label = ctk.CTkLabel(
    window,
    text=f"guidance scale",
)
txt2img_guidance_slider = ctk.CTkSlider(
    window,
    from_=2,
    to=50,
    number_of_steps=49,
)
txt2img_guidance_slider_label.pack()
txt2img_guidance_slider.pack()

button = ctk.CTkButton(
    window,
    text="Generate",
    fg_color="blue",
    text_color="white",
    command=lambda: run_txt2img(
        txt2img_prompt.get("1.0", "end-1c"),
        txt2img_neg_prompt.get("1.0", "end-1c"),
        txt2img_step_slider.get(),
        txt2img_guidance_slider.get(),
        "DEIS",
        "",
        "1_PhotoMerge_v1-2_MaxSlicing_Optimized_ft_mse_onnx-fp16"
    ),
)
button.pack()


# variables to set before running
scheduler = None
pipe = None

# run
window.mainloop()
