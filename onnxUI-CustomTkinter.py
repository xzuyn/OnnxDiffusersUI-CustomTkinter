import customtkinter as ctk
import PIL
from PIL import Image
from math import ceil

from diffusers import (
    OnnxRuntimeModel,
    OnnxStableDiffusionPipeline,
    OnnxStableDiffusionImg2ImgPipeline,
    DDIMScheduler,
    PNDMScheduler,
    LMSDiscreteScheduler,
)


def resize_and_crop(input_image: PIL.Image.Image, height: int, width: int):
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


def step_adjustment(unadjusted_steps, denoise, pipeline, scheduler):
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
def txt2img_cpu(model_path, provider, scheduler):
    global textenc_on_cpu
    global vae_on_cpu

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


# window
window = ctk.CTk()
window.title("OnnxDiffusersUI")
window.geometry("640x360")

# widgets
label = ctk.CTkLabel(
    window,
    text="txt2img",
)
label.pack()

button = ctk.CTkButton(
    window,
    text="Generate",
    fg_color="blue",
    text_color="white",
    command=lambda: print("a button was pressed"),
)
button.pack()

# run
window.mainloop()
