# Convolutional Neural Networks

$$
\boxed{(f * g)[i] = \sum_{k} f[k] \cdot g[i - k]}
$$

**Convolutional neural networks** exploit spatial structure in data. Instead of learning separate weights for every pixel, CNNs learn small filters that slide across the input, detecting patterns regardless of where they appear. This translation invariance made CNNs dominant in computer vision and influenced modern architectures.

Prerequisites: [backpropagation](../neural-networks/backpropagation.md), [activation functions](../neural-networks/activation-functions.md). Code: `numpy`.

---

## The Problem: Images Are Big

### Fully Connected Approach

A 224×224 RGB image has 150,528 input features. With a 1000-neuron hidden layer:

$$
\text{Parameters} = 150,528 \times 1000 = 150\text{M parameters (first layer alone!)}
$$

This is wasteful and ignores spatial structure:
- Adjacent pixels are related
- Patterns appear at different positions
- A cat is a cat regardless of location

### The Convolution Solution

Learn small filters that detect local patterns:

```python
import numpy as np

def conv2d(image, kernel):
    """
    2D convolution: slide kernel across image.

    Args:
        image: [H, W] or [H, W, C]
        kernel: [kH, kW] or [kH, kW, C]
    Returns:
        Feature map [H-kH+1, W-kW+1]
    """
    if len(image.shape) == 2:
        image = image[:, :, np.newaxis]
        kernel = kernel[:, :, np.newaxis]

    H, W, C = image.shape
    kH, kW, _ = kernel.shape

    out_H = H - kH + 1
    out_W = W - kW + 1
    output = np.zeros((out_H, out_W))

    for i in range(out_H):
        for j in range(out_W):
            # Extract patch and compute dot product
            patch = image[i:i+kH, j:j+kW, :]
            output[i, j] = np.sum(patch * kernel)

    return output
```

**What this means:** A 3×3 kernel has 9 parameters regardless of image size. The same kernel applied everywhere enables pattern detection at any position.

## Convolution Operation

### Filter/Kernel

Small learnable weight matrix:

```python
# Edge detection kernel (horizontal)
horizontal_edge = np.array([
    [-1, -1, -1],
    [ 0,  0,  0],
    [ 1,  1,  1]
])

# Edge detection kernel (vertical)
vertical_edge = np.array([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1]
])

# Sharpen kernel
sharpen = np.array([
    [ 0, -1,  0],
    [-1,  5, -1],
    [ 0, -1,  0]
])
```

### Stride and Padding

Control output size:

```python
def conv2d_strided(image, kernel, stride=1, padding=0):
    """
    Convolution with stride and padding.

    Args:
        stride: Step size for sliding kernel
        padding: Zero-padding around input
    """
    # Add padding
    if padding > 0:
        image = np.pad(image, ((padding, padding), (padding, padding), (0, 0)))

    H, W, C = image.shape
    kH, kW, _ = kernel.shape

    out_H = (H - kH) // stride + 1
    out_W = (W - kW) // stride + 1
    output = np.zeros((out_H, out_W))

    for i in range(out_H):
        for j in range(out_W):
            h_start = i * stride
            w_start = j * stride
            patch = image[h_start:h_start+kH, w_start:w_start+kW, :]
            output[i, j] = np.sum(patch * kernel)

    return output


# Output size formula:
# out_size = (in_size + 2*padding - kernel_size) // stride + 1
```

### Multiple Channels and Filters

```python
class Conv2DLayer:
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        """
        Full convolutional layer.

        Args:
            in_channels: Number of input channels (e.g., 3 for RGB)
            out_channels: Number of filters (output channels)
            kernel_size: Size of each filter
        """
        self.stride = stride
        self.padding = padding

        # Filters: [out_channels, in_channels, kH, kW]
        k = kernel_size
        scale = np.sqrt(2.0 / (in_channels * k * k))
        self.filters = np.random.randn(out_channels, in_channels, k, k) * scale
        self.bias = np.zeros(out_channels)

    def forward(self, x):
        """
        Args:
            x: [batch, in_channels, H, W]
        Returns:
            [batch, out_channels, out_H, out_W]
        """
        batch_size, in_c, H, W = x.shape
        out_c, _, kH, kW = self.filters.shape

        # Padding
        if self.padding > 0:
            x = np.pad(x, ((0, 0), (0, 0),
                          (self.padding, self.padding),
                          (self.padding, self.padding)))

        _, _, H_pad, W_pad = x.shape
        out_H = (H_pad - kH) // self.stride + 1
        out_W = (W_pad - kW) // self.stride + 1

        output = np.zeros((batch_size, out_c, out_H, out_W))

        for b in range(batch_size):
            for oc in range(out_c):
                for i in range(out_H):
                    for j in range(out_W):
                        h_start = i * self.stride
                        w_start = j * self.stride
                        patch = x[b, :, h_start:h_start+kH, w_start:w_start+kW]
                        output[b, oc, i, j] = np.sum(patch * self.filters[oc]) + self.bias[oc]

        return output
```

## Pooling

### Reduce Spatial Dimensions

```python
def max_pool2d(x, pool_size=2, stride=2):
    """
    Max pooling: take maximum in each window.

    Provides:
    - Translation invariance
    - Dimensionality reduction
    - Some noise robustness
    """
    batch_size, channels, H, W = x.shape
    out_H = H // stride
    out_W = W // stride

    output = np.zeros((batch_size, channels, out_H, out_W))

    for i in range(out_H):
        for j in range(out_W):
            h_start = i * stride
            w_start = j * stride
            window = x[:, :, h_start:h_start+pool_size, w_start:w_start+pool_size]
            output[:, :, i, j] = np.max(window, axis=(2, 3))

    return output


def avg_pool2d(x, pool_size=2, stride=2):
    """Average pooling: take mean in each window."""
    batch_size, channels, H, W = x.shape
    out_H = H // stride
    out_W = W // stride

    output = np.zeros((batch_size, channels, out_H, out_W))

    for i in range(out_H):
        for j in range(out_W):
            h_start = i * stride
            w_start = j * stride
            window = x[:, :, h_start:h_start+pool_size, w_start:w_start+pool_size]
            output[:, :, i, j] = np.mean(window, axis=(2, 3))

    return output
```

### Global Average Pooling

Replace fully connected layers:

```python
def global_avg_pool(x):
    """
    Global average pooling: [batch, channels, H, W] → [batch, channels]

    Used to connect conv features to classification head.
    """
    return np.mean(x, axis=(2, 3))
```

**What this means:** Pooling reduces spatial dimensions while preserving important features. A cat detected at (10, 20) or (11, 21) produces similar pooled output—translation invariance.

## CNN Architectures

### Simple CNN

```python
class SimpleCNN:
    def __init__(self, num_classes=10):
        """
        Simple CNN for image classification.
        Input: 32x32x3 (like CIFAR-10)
        """
        # Conv layers: progressively more filters, smaller spatial size
        self.conv1 = Conv2DLayer(3, 32, kernel_size=3, padding=1)    # 32x32x32
        self.conv2 = Conv2DLayer(32, 64, kernel_size=3, padding=1)   # 16x16x64 (after pool)
        self.conv3 = Conv2DLayer(64, 128, kernel_size=3, padding=1)  # 8x8x128 (after pool)

        # FC layer after global pooling: 128 → num_classes
        self.fc = np.random.randn(128, num_classes) * 0.01
        self.fc_bias = np.zeros(num_classes)

    def forward(self, x):
        # Conv block 1
        x = self.conv1.forward(x)
        x = relu(x)
        x = max_pool2d(x)  # 32 → 16

        # Conv block 2
        x = self.conv2.forward(x)
        x = relu(x)
        x = max_pool2d(x)  # 16 → 8

        # Conv block 3
        x = self.conv3.forward(x)
        x = relu(x)
        x = max_pool2d(x)  # 8 → 4

        # Global average pool and classify
        x = global_avg_pool(x)  # [batch, 128]
        logits = x @ self.fc + self.fc_bias

        return logits


def relu(x):
    return np.maximum(0, x)
```

### Historical Architectures

| Architecture | Year | Key Innovation | ImageNet Top-5 |
|--------------|------|----------------|----------------|
| AlexNet | 2012 | ReLU, dropout, GPU training | 15.3% |
| VGGNet | 2014 | Small 3×3 filters, depth | 7.3% |
| GoogLeNet | 2014 | Inception modules | 6.7% |
| ResNet | 2015 | Residual connections | 3.6% |

### ResNet: Residual Connections

```python
class ResidualBlock:
    def __init__(self, channels):
        """
        Residual block: learn f(x) where output = x + f(x)

        Key insight: easier to learn residual f(x) = desired - x
        than to learn the full mapping directly.
        """
        self.conv1 = Conv2DLayer(channels, channels, kernel_size=3, padding=1)
        self.conv2 = Conv2DLayer(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x

        out = self.conv1.forward(x)
        out = relu(out)
        out = self.conv2.forward(out)

        # Skip connection
        out = out + residual
        out = relu(out)

        return out
```

**What this means:** Residual connections solved the vanishing gradient problem for very deep networks. By adding skip connections, gradients can flow directly through the network.

## Backpropagation Through Convolutions

### Gradient Computation

```python
def conv2d_backward(d_out, x, kernel, stride=1, padding=0):
    """
    Backprop through convolution.

    Args:
        d_out: Gradient of loss w.r.t. conv output
        x: Original input
        kernel: Conv weights
    Returns:
        d_x: Gradient w.r.t. input
        d_kernel: Gradient w.r.t. kernel
    """
    # Simplified for single channel
    if padding > 0:
        x_pad = np.pad(x, padding)
    else:
        x_pad = x

    kH, kW = kernel.shape
    d_kernel = np.zeros_like(kernel)
    d_x_pad = np.zeros_like(x_pad)

    out_H, out_W = d_out.shape

    for i in range(out_H):
        for j in range(out_W):
            h_start = i * stride
            w_start = j * stride

            # Gradient w.r.t. kernel
            patch = x_pad[h_start:h_start+kH, w_start:w_start+kW]
            d_kernel += d_out[i, j] * patch

            # Gradient w.r.t. input
            d_x_pad[h_start:h_start+kH, w_start:w_start+kW] += d_out[i, j] * kernel

    # Remove padding from gradient
    if padding > 0:
        d_x = d_x_pad[padding:-padding, padding:-padding]
    else:
        d_x = d_x_pad

    return d_x, d_kernel
```

### Transposed Convolution

For upsampling (used in segmentation, GANs):

```python
def transposed_conv2d(x, kernel, stride=2):
    """
    Transposed convolution: upsample by reversing downsampling.

    Inserts zeros between input elements, then convolves.
    """
    H, W = x.shape
    kH, kW = kernel.shape

    # Upsample by inserting zeros
    up_H = H + (H - 1) * (stride - 1)
    up_W = W + (W - 1) * (stride - 1)
    x_up = np.zeros((up_H, up_W))
    x_up[::stride, ::stride] = x

    # Convolve with flipped kernel
    kernel_flipped = kernel[::-1, ::-1]

    # Pad and convolve
    pad = kH - 1
    x_pad = np.pad(x_up, pad)

    out_H = x_pad.shape[0] - kH + 1
    out_W = x_pad.shape[1] - kW + 1
    output = np.zeros((out_H, out_W))

    for i in range(out_H):
        for j in range(out_W):
            patch = x_pad[i:i+kH, j:j+kW]
            output[i, j] = np.sum(patch * kernel_flipped)

    return output
```

## Connection to Transformers

### From CNNs to Vision Transformers

CNNs dominated computer vision until Vision Transformers (ViT) showed attention works for images too:

| Aspect | CNNs | Vision Transformers |
|--------|------|---------------------|
| Inductive bias | Translation invariance | Minimal (learned from data) |
| Local patterns | 3×3 kernels | Attention over patches |
| Global context | Deep stacking | Direct attention |
| Data efficiency | Better with less data | Needs more data |

```python
def image_to_patches(image, patch_size=16):
    """
    Convert image to patches for Vision Transformer.

    Image [H, W, C] → Patches [num_patches, patch_size^2 * C]
    """
    H, W, C = image.shape
    assert H % patch_size == 0 and W % patch_size == 0

    patches = []
    for i in range(0, H, patch_size):
        for j in range(0, W, patch_size):
            patch = image[i:i+patch_size, j:j+patch_size, :]
            patches.append(patch.flatten())

    return np.array(patches)  # [num_patches, patch_dim]
```

### Modern Hybrid Approaches

Many modern architectures combine:
- **ConvNeXt:** CNN with transformer-inspired designs
- **Swin Transformer:** Shifted window attention (local like CNNs)
- **CoAtNet:** Conv stem + transformer body

**What this means:** CNNs taught us that local patterns and translation invariance matter. Transformers taught us that global attention is powerful. Modern architectures blend both insights.

## Summary

| Concept | Description |
|---------|-------------|
| Convolution | Sliding filter for local pattern detection |
| Pooling | Downsample while preserving features |
| Translation invariance | Same pattern detected anywhere |
| Residual connections | Skip connections for deep networks |
| Feature hierarchy | Low-level → high-level features |

**The essential insight:** CNNs exploit the structure of images—local patterns, translation invariance, and hierarchy. A 3×3 filter can detect an edge regardless of image size. Stacking layers builds from edges to textures to parts to objects. This efficiency made deep learning practical for vision.

**Historical context:** AlexNet (2012) kicked off the deep learning revolution by winning ImageNet with CNNs on GPUs. ResNet (2015) enabled 150+ layer networks with residual connections. Though Vision Transformers now rival CNNs, the lessons of local processing and hierarchical features remain influential in all architectures.

**Relevance to LLMs:** While transformers replaced CNNs for sequences, the concept of efficient weight sharing (same filter everywhere = same attention everywhere) and hierarchical feature learning (layers building abstractions) carried forward.
