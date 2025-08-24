from torchvision import transforms

class SamplewiseNormalize:
    def __call__(self, tensor):
        mean = tensor.mean()
        std = tensor.std()
        return (tensor - mean) / std

class InvertColors:
    def __call__(self, tensor):
        return 1 - tensor

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Resize(512),
     transforms.Grayscale(),
     InvertColors(),
     SamplewiseNormalize()
    ])