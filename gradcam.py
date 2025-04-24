import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from utils import scale_accross_batch_and_channels, scale_cam_image

class FullGrad:
    def __init__(self, model_dict):
        self.model = model_dict['arch']
        self.model.eval()
        self.device = next(self.model.parameters()).device
        self.input_size = model_dict['input_size']

        self.gradients = []
        self.target_layers = [module for module in self.model.modules() if isinstance(module, (nn.Conv2d, nn.BatchNorm2d))]

        self.bias_data = [self._extract_bias(module).cpu().numpy() for module in self.target_layers if module.bias is not None]
        self._register_hook()

    def _register_hook(self):
        for layer in self.target_layers:
            layer.register_forward_hook(self.save_gradient)

    def save_gradient(self, module, input, output):
        if not hasattr(output, "requires_grad") or not output.requires_grad:
            # You can only register hooks on tensor requires grad.
            return

        # Gradients are computed in reverse order
        def _store_grad(grad):
            self.gradients = [grad.cpu().detach()] + self.gradients

        output.register_hook(_store_grad)

    def _extract_bias(self, layer):
        if isinstance(layer, nn.BatchNorm2d):
            bias = - (layer.running_mean * layer.weight / torch.sqrt(layer.running_var + layer.eps)) + layer.bias
            return bias.data
        return layer.bias.data

    def __call__(self, input_tensor, class_idx=None):
        self.gradients.clear()
        input_tensor = input_tensor.to(self.device).requires_grad_(True)
        output = self.model(input_tensor)
        class_idx = output.argmax(dim=1).item() if class_idx is None else class_idx
        score = output[:, class_idx].sum()
        self.model.zero_grad()
        
        cam_per_target_layer = []
        # Standard first-order version
        score.backward(retain_graph=True)
        grad_input = input_tensor.grad.data.cpu().numpy()
        grads_list = [g.cpu().data.numpy() for g in self.gradients]
        gradient_multiplied_input = np.abs(grad_input * input_tensor.data.cpu().numpy())
        
        gradient_multiplied_input = scale_accross_batch_and_channels(
            gradient_multiplied_input,
            self.input_size)
        cam_per_target_layer.append(gradient_multiplied_input)

        assert(len(self.bias_data) == len(grads_list))
        for bias, grads in zip(self.bias_data, grads_list):
            bias = bias[None, :, None, None]
            
            bias_grad = np.abs(bias * grads)
            result = scale_accross_batch_and_channels(
                bias_grad, self.input_size)
            result = np.sum(result, axis=1)
            cam_per_target_layer.append(result[:, None, :])
        cam_per_target_layer = np.concatenate(cam_per_target_layer, axis=1)

        cam_per_target_layer = np.sum(
                cam_per_target_layer, axis=1)[:, None, :]
        result = np.sum(cam_per_target_layer, axis=1)
        result = scale_cam_image(result)

        return result

class FullGradpp:
    """
    Combines FullGrad's multi-layer aggregation with GradCAM++'s second-order weighting.

    Args:
        model_dict (dict): must contain:
            - 'arch': a torch.nn.Module (evaluation mode)
            - 'input_size': tuple (H_in, W_in) for upsampling
        verbose (bool): if True, prints the number of target layers
    """
    def __init__(self, model_dict, verbose=False):
        self.model = model_dict['arch']
        self.device = next(self.model.parameters()).device
        self.input_size = model_dict['input_size']

        self.target_layers = [m for m in self.model.modules()
                              if isinstance(m, (torch.nn.Conv2d, torch.nn.BatchNorm2d))]
        if verbose:
            print(f"[*] {len(self.target_layers)} target layers registered.")

        self.activations = []
        self.gradients = []
        for layer in self.target_layers:
            layer.register_forward_hook(self.save_activation)
            layer.register_forward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        activation = output
        self.activations.append(activation.cpu().detach().data.numpy())

    def save_gradient(self, module, input, output):
        if not hasattr(output, "requires_grad") or not output.requires_grad:
            return

        def _store_grad(grad):
            self.gradients = [grad.cpu().detach().data.numpy()] + self.gradients

        output.register_hook(_store_grad)

    def _resize(self, cam: torch.Tensor) -> torch.Tensor:
        return F.interpolate(cam, size=self.input_size,
                             mode='bilinear', align_corners=False)

    def forward(self, input_tensor: torch.Tensor, class_idx=None, retain_graph=False):
        self.activations.clear()
        self.gradients.clear()
        x = input_tensor.to(self.device)

        logits = self.model(x)
        if class_idx is None:
            class_idx = torch.argmax(logits, dim=1).item()
        score = logits[:, class_idx].sum()

        self.model.zero_grad()
        score.backward(retain_graph=True)

        cam_per_target_layer = []
        weight_per_target_layer = []
        eps = 1e-7
        for i in range(len(self.target_layers)):
            layer_activations = None
            layer_grads = None
            if i < len(self.activations):
                layer_activations = self.activations[i]
            if i < len(self.gradients):
                layer_grads = self.gradients[i]

            A = layer_activations
            g1 = layer_grads

            g2 = g1 ** 2
            g3 = g2 * g1

            numerator = g2
            sum_activations = np.sum(A, axis=(2, 3))
            denominator = 2 * g2 + sum_activations[:, :, None, None] * g3
            alpha = numerator / (denominator + eps)

            w_k = np.where(g1 != 0, alpha, 0)

            w_k = np.maximum(g1, 0) * w_k
            w_k = np.sum(w_k, axis=(2, 3))

            layer_cam = np.sum(w_k[:, :, None, None] * A, axis=1)
            layer_cam = np.maximum(layer_cam, 0)
            resized = scale_cam_image(layer_cam, self.input_size)
            cam_per_target_layer.append(resized[:, None, :])
            weight_per_target_layer.append(sum_activations.mean())

        cam_per_target_layer = np.concatenate(cam_per_target_layer, axis=1)
        cam_per_target_layer = np.maximum(cam_per_target_layer, 0)
        result = np.average(cam_per_target_layer, weights=weight_per_target_layer, axis=1)
        
        return scale_cam_image(result, self.input_size)

    def __call__(self, input_tensor, class_idx=None, retain_graph=False):
        return self.forward(input_tensor, class_idx, retain_graph)
