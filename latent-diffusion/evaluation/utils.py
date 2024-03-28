import wandb

class MetricModel:
    def __init__(self, _name):        
        self._name = _name
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
            print(f"Test on {name.upper()} is Epoch {epoch} Step {step}: {result}")
            wandb.log({f'validation {name.upper()}': result})