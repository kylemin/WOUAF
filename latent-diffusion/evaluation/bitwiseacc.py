from evaluation.utils import MetricModel
import torch

def acc_calculation(args, phis, decoding_network, generated_image, bsz = None,vae = None):
    if args.decoding_network_type == "fc":
        reconstructed_keys = decoding_network(vae.encode(generated_image).latent_dist.sample().reshape(bsz, -1))
    elif args.decoding_network_type == "resnet":
        reconstructed_keys = decoding_network(generated_image)
    else:
        raise ValueError("Not suported network")

    gt_phi = (phis > 0.5).int()
    reconstructed_keys = (torch.sigmoid(reconstructed_keys) > 0.5).int()
    bit_acc = ((gt_phi == reconstructed_keys).sum(dim=1)) / args.phi_dimension

    return bit_acc

class BitAcc(MetricModel):
    def __init__(self, _name):
        super().__init__(_name)
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
                outputs['images'],
                bsz,
                vae
            )
        )

        self.results_trainval.extend(
            acc_calculation(
                self._args,
                phis,
                decoding_network,
                outputs['images_trainval'],
                bsz,
                vae
            )
        )

    def get_results(self, accelerator):
        return (torch.mean(torch.tensor(self.results_ngc)), torch.mean(torch.tensor(self.results_trainval)))
