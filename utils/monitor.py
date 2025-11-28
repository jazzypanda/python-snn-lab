import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from modules.neuron import LIFNode

class SNNMonitor:
    def __init__(self, writer: SummaryWriter):
        self.writer = writer
        self.hooks = []
        self.monitored_data = {} # {layer_name: {'v': [], 's': []}}

    def register(self, model: torch.nn.Module):
        """
        Register hooks to all LIFNodes in the model.
        """
        # Clear old hooks if any
        self.remove()
        
        for name, module in model.named_modules():
            if isinstance(module, LIFNode):
                # Define hook function with closure to capture name
                def get_hook(layer_name):
                    def hook(m, input, output):
                        # input is usually tuple, output is spike tensor
                        # m.v is the membrane potential AFTER update
                        # m.last_v would be better if we want potential before reset, 
                        # but m.v is the state at end of step.
                        
                        # We only want to record this during specific evaluation steps
                        # to avoid OOM.
                        
                        if layer_name not in self.monitored_data:
                            self.monitored_data[layer_name] = {'v': [], 's': []}
                        
                        # Capture firing rate from output spikes
                        self.monitored_data[layer_name]['s'].append(output.detach().cpu())
                        
                        # Capture membrane potential trace if available
                        if m.v_seq is not None:
                            self.monitored_data[layer_name]['v'].append(m.v_seq.clone()) 
                        else:
                            # Fallback to last step v if not in monitor mode (should not happen if logic is correct)
                            self.monitored_data[layer_name]['v'].append(m.v.detach().cpu().unsqueeze(1)) 
                        
                    return hook
                
                handle = module.register_forward_hook(get_hook(name))
                self.hooks.append(handle)
                
    def set_monitor_mode(self, model, enable=True):
        """
        Recursively set monitor_mode for all LIFNodes
        """
        for m in model.modules():
            if isinstance(m, LIFNode):
                m.monitor_mode = enable
    
    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []
        self.monitored_data = {}

    def flush(self, epoch):
        """
        Process collected data and write to TensorBoard.
        Then clear buffers.
        """
        for layer_name, data in self.monitored_data.items():
            # data['s'] is a list of tensors (Batch, T, ...)
            if not data['s']:
                continue
                
            # Concatenate all batches collected
            spikes = torch.cat(data['s'], dim=0) # (Total_N, T, ...)
            v_trace = torch.cat(data['v'], dim=0) # (Total_N, T, ...)
            
            # 1. Firing Rate (Spikes / T)
            # Calculate mean firing rate per neuron: sum(T) / T
            # Then flatten to see distribution across all neurons in this layer
            firing_rate = spikes.mean(dim=1) # (Total_N, C, H, W)
            self.writer.add_histogram(f'{layer_name}/firing_rate_dist', firing_rate.flatten(), epoch)
            self.writer.add_scalar(f'{layer_name}/mean_firing_rate', firing_rate.mean(), epoch)
            
            # 2. Membrane Potential Distribution (All Timesteps)
            self.writer.add_histogram(f'{layer_name}/v_dist', v_trace.flatten(), epoch)
            
        # Clear
        self.monitored_data = {}
