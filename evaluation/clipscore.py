import torch
import clip
from PIL import Image

import logging
import sklearn
import warnings
from packaging import version
import numpy as np
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

from evaluation.utils import MetricModel

class CLIPScore(MetricModel):
    def __init__(self, _name):
        super().__init__(_name)
        self.initialize()

    def initialize(self):
        self.results = []

    def setup(self, accelerator, _args):
        self.model = CLIPModel(device = accelerator.device)
        self._args = _args

    def infer_batch(self, accelerator=None, decoding_network=None, vae=None, batch=None, outputs=None, phis=None, bsz=None):
        generated_image = outputs['images']

        for idx in range(generated_image.shape[0]):
            self.results += list(self.model.inference(image = generated_image[idx].unsqueeze(0), texts = batch["captions"][idx]))

    def get_results(self, accelerator):
        results = (np.mean(self.results),)
        self.initialize()
        return results

class CLIPModel:
    def __init__(self, device=None, model="ViT-B/32"):
        self.device=device
        self.model, self.preprocess = clip.load(model, device=self.device)
        self.preprocess = Compose([
            Resize(224, interpolation=BICUBIC),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        self.model.eval()

    def inference(self, image, texts):
        with torch.no_grad():
            image = self.preprocess(image[0]).unsqueeze(0).to(self.device)
            texts = clip.tokenize(texts[0]).to(self.device)
            image_feats = self.model.encode_image(image).detach().cpu().numpy()
            text_feats = self.model.encode_text(texts).detach().cpu().numpy()

            ## This snippet is taken from reference free clip score paper.
            if version.parse(np.__version__) < version.parse('1.21'):
                images = sklearn.preprocessing.normalize(image_feats, axis=1)
                candidates = sklearn.preprocessing.normalize(text_feats, axis=1)
            else:
                images = image_feats / np.sqrt(np.sum(image_feats**2, axis=1, keepdims=True))
                candidates = text_feats / np.sqrt(np.sum(text_feats**2, axis=1, keepdims=True))

        w=2.5
        per = w*np.clip(np.sum(images * candidates, axis=1), 0, None)

        return per


if __name__=="__main__":
    logging.info("You called the main script itself, we will not run a single test.")
    image = Image.open("/data_5/data/matt/counterfactual_augmentations/main2/SCD-VG/outputs/example/end2end_dogs_1.jpg")
    texts = ["a photo of a cat", "a photo of a dog"]

    logging.info("Initializing the CLIPModel model.")
    model = CLIPModel()

    logging.info("Running the inference on sample image/texts on cats.")
    results = model.inference(image=image, texts=texts)

    for en,txt in enumerate(texts):
        if results[0][en]>0.5:
            print(f"Input text: {txt} \n Prediction: {results[0][en]>0.5} \n Confidence: {results[0][en]}")
        else:
            print(f"Input text: {txt} \n Prediction: {results[0][en]>0.5} \n Confidence: {1-results[0][en]}")
