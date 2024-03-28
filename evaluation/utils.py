import wandb
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel, DDIMScheduler, EulerDiscreteScheduler
from torchvision import transforms
import torch
import random


class Text2Image:
    def __init__(self, weight_dtype, resolution):
        self.pipe = StableDiffusionPipeline.from_pretrained('stabilityai/stable-diffusion-2', safety_checker=None, torch_dtype=weight_dtype)
        self.pipe = self.pipe.to("cuda")
        self.resize_transform = transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR)
    def inference(self, batch):
        out_latents = self.pipe([random.choice(batch["captions"])], output_type="latent", num_inference_steps=50, guidance_scale=7.5).images
        return out_latents
    def postprocess(self, image):
        # return image # for when we are using our custom-trained model
        image = self.resize_transform(image)
        return image # for when we are using our any pre-trained model i.e., stabilityai/stable-diffusion-2

class ImageSuperResolution:
    def __init__(self, weight_dtype, resolution):
        from extras.pipeline_stable_diffusion_upscale import StableDiffusionUpscalePipeline
        self.pipe = StableDiffusionUpscalePipeline.from_pretrained('stabilityai/stable-diffusion-x4-upscaler', safety_checker=None, torch_dtype=weight_dtype)
        self.pipe = self.pipe.to("cuda")
        self.resize_transform_infer = transforms.Resize(resolution//4, interpolation=transforms.InterpolationMode.BILINEAR)
        self.resize_transform = transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR)
    def inference(self, batch):
        out_latents = self.pipe([random.choice(batch["captions"])], image=self.resize_transform_infer(batch['pixel_values']), output_type="latent", num_inference_steps=50, guidance_scale=7.5).images
        return out_latents
    def postprocess(self, image):
        # return image # for when we are using our custom-trained model
        image = self.resize_transform(image)
        return image # for when we are using our any pre-trained model i.e., stabilityai/stable-diffusion-2

class ImageInPaint:
    def __init__(self, weight_dtype, resolution):
        from extras.pipeline_stable_diffusion_inpaint import StableDiffusionInpaintPipeline
        from transformers import OwlViTProcessor, OwlViTForObjectDetection
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-inpainting",
            torch_dtype=weight_dtype,
        )
        self.pipe = self.pipe.to("cuda")
        self.processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
        self.model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32").cuda()
        self.resize_transform = transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR)
        self.resolution = resolution

    def get_mask(self, batch):
        flag=False
        mask_image = torch.zeros((self.resolution, self.resolution))* 0.0
        for i in range(len(batch["captions"])):
            try:
                caption = random.choice(batch["captions"])
                doc = nlp(caption)
                sub_toks = [str(tok) for tok in doc if (tok.dep_ == "nsubj")]
                texts = [f"a photo of {str(sub_toks[0])}"]
                flag=True
                break
            except:
                continue

        if not flag:
            return mask_image, random.choice(batch["captions"])

        transform = transforms.ToPILImage()
        assert batch["pixel_values"].shape[0]==1
        image = transform(batch["pixel_values"][0])
        inputs = self.processor(text=texts, images=image, return_tensors="pt")
        inputs = {k:v.cuda() for k,v in inputs.items()}
        outputs = self.model(**inputs)
        results = self.processor.post_process(outputs=outputs, target_sizes=torch.Tensor([image.size[::-1]]).cuda())
        i = 0  # Retrieve predictions for the first image for the corresponding text queries
        text = texts[i]
        boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]

        score_threshold = 0.1
        bbox = None
        area = -1
        for box, score, label in zip(boxes, scores, labels):
            box = [round(i, 2) for i in box.tolist()]
            if abs(box[0]-box[2])*abs(box[1]-box[3])>area:
                area = abs(box[0]-box[2])*abs(box[1]-box[3])
                bbox = box
        if bbox:
            mask_image[int(bbox[0]):int(bbox[2]), int(bbox[1]):int(bbox[3])] = 1.0

        return mask_image, caption

    def inference(self, batch):
        mask_image, caption = self.get_mask(batch)
        out_latents = self.pipe([caption], image=batch['pixel_values'], mask_image=mask_image, output_type="latent", num_inference_steps=50, guidance_scale=7.5).images
        return out_latents
    def postprocess(self, image):
        image = self.resize_transform(image)
        return image # for when we are using our any pre-trained model i.e., stabilityai/stable-diffusion-2


class MetricModel:
    def __init__(self, _name):
        self._name = _name
        self.data = None
        self._args = None

    def get_name(self):
        return self._name

    def setup(self, accelerator, _args):
        pass

    def infer_batch(self, accelerator=None, decoding_network=None, vae=None, batch=None, outputs=None, phis=None, bsz=None):
        raise NotImplementedError

    def get_results(self):
        raise NotImplementedError

    def print_results(self, accelerator, epoch, step):
        results = self.get_results(accelerator)
        for result, name in zip(results, self._name):
            if not self.data:
                print(f"Test on {name.upper()} is Epoch {epoch} Step {step}: {result}")
                wandb.log({f'validation {name.upper()}': result})
            else:
                assert len(self.data[name])==len(result)
                for k, val in zip(self.data[name], result):
                    print(f"Test on {name.upper()} with value {k} is Epoch {epoch} Step {step}: {val}")
                    wandb.log({f'validation {name.upper()} with {k}': val})
