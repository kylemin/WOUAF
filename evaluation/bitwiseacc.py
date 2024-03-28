from evaluation.utils import MetricModel
import torch
from torchvision import transforms

def acc_calculation(args, phis, decoding_network, generated_image, bsz = None,vae = None):
    reconstructed_keys = decoding_network(generated_image)
    gt_phi = (phis > 0.5).int()
    reconstructed_keys = (torch.sigmoid(reconstructed_keys) > 0.5).int()
    bit_acc = ((gt_phi == reconstructed_keys).sum(dim=1)) / args.phi_dimension

    return bit_acc

class BitAcc(MetricModel):
    def __init__(self, _name):
        super().__init__(_name)
        self.initialize()
        self.transforms = transforms.Compose([
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

    def initialize(self):
        self.results_ngc = []
        self.results_trainval = []

    def setup(self, accelerator, _args):
        self._args = _args

    def infer_batch(self, accelerator=None, decoding_network=None, vae=None, batch=None, outputs=None, phis=None, bsz=None):
        self.results_ngc.extend(
            acc_calculation(
                self._args,
                phis,
                decoding_network,
                self.transforms(outputs['images']),
                bsz,
                vae
            )
        )

        self.results_trainval.extend(
            acc_calculation(
                self._args,
                phis,
                decoding_network,
                self.transforms(outputs['images_trainval']),
                bsz,
                vae
            )
        )

    def get_results(self, accelerator):
        results = (torch.mean(torch.tensor(self.results_ngc)), torch.mean(torch.tensor(self.results_trainval)))
        self.initialize()
        return results
