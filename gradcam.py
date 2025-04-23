import torch
from torch import nn
import torch.nn.functional as F


class FullGradBase:
    def __init__(self, model_dict, smoothing=False):
        self.model = model_dict['arch']
        self.model.eval()
        self.device = next(self.model.parameters()).device
        self.input_size = model_dict['input_size']

        self.activations = []
        self.gradients = []
        self.biases = []
        self.use_second_order = False  # Set True in subclass for FullGrad++
        self.smoothing = smoothing

        self._register_hooks()

    def _register_hooks(self):
        def has_bias(layer):
            return isinstance(layer, (nn.Conv2d, nn.BatchNorm2d)) and layer.bias is not None

        for layer in self.model.modules():
            if has_bias(layer):
                layer_id = id(layer)
                self.biases.append((layer_id, self._extract_bias(layer).to(self.device)))

                def fwd_hook(module, inp, outp):
                    self.activations.append((id(module), outp))

                def bwd_hook(module, grad_in, grad_out):
                    self.gradients.append((id(module), grad_out[0]))


                layer.register_forward_hook(fwd_hook)
                layer.register_backward_hook(bwd_hook)

    def _extract_bias(self, layer):
        if isinstance(layer, nn.BatchNorm2d):
            return - (layer.running_mean * layer.weight / torch.sqrt(layer.running_var + layer.eps)) + layer.bias
        return layer.bias
    
    def _apply_smoothing(self, saliency, kernel_size=3, sigma=2):
        """Apply 2D Gaussian smoothing per image"""
        if kernel_size % 2 == 0:
            kernel_size += 1
        channels = saliency.shape[1]
        coords = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        grid = coords.repeat(kernel_size).view(kernel_size, kernel_size)
        gauss = torch.exp(-(grid**2 + grid.T**2) / (2 * sigma**2))
        gauss /= gauss.sum()

        kernel = gauss.view(1, 1, kernel_size, kernel_size).repeat(channels, 1, 1, 1)
        kernel = kernel.to(saliency.device)

        return F.conv2d(saliency, kernel, padding=kernel_size // 2, groups=channels)

    def __call__(self, input_tensor, class_idx=None):
        self.activations.clear()
        self.gradients.clear()

        input_tensor = input_tensor.to(self.device)
        input_tensor.requires_grad = True

        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        score = output[:, class_idx].squeeze()
        self.model.zero_grad()

        if self.use_second_order:
            # FullGrad++ — use second-order gradients
            grad = torch.autograd.grad(score, input_tensor, create_graph=True)[0]
        else:
            score.backward(retain_graph=True)
            grad = input_tensor.grad

        cam = torch.abs(grad * input_tensor).sum(dim=1, keepdim=True)

        for (layer_id, bias) in self.biases:
            matching_grad = next((g for (lid, g) in self.gradients if lid == layer_id), None)
            if matching_grad is None:
                continue

            b = bias.view(1, -1, 1, 1)
            g = matching_grad
            g = g * g if self.use_second_order else g

            if g.shape[1] != b.shape[1]:  # size mismatch, skip
                continue

            contrib = torch.abs(b * g).sum(dim=1, keepdim=True)
            contrib = F.interpolate(contrib, size=input_tensor.shape[2:], mode='bilinear', align_corners=False)
            cam += contrib

        cam = F.relu(cam)

        if self.smoothing:
            cam = self._apply_smoothing(cam)

        cam -= cam.min()
        cam /= (cam.max() + 1e-8)
        return cam.detach(), output.detach()


class FullGrad(FullGradBase):
    def __init__(self, model_dict, smoothing=False):
        super().__init__(model_dict, smoothing)
        self.use_second_order = False


class FullGradpp(FullGradBase):
    def __init__(self, model_dict, smoothing=False):
        super().__init__(model_dict, smoothing)
        self.use_second_order = True
