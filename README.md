# Train CIFAR10 with PyTorch

I'm playing with [PyTorch](http://pytorch.org/) on the CIFAR10 dataset.

## Pros & cons
Pros:
- Built-in data loading and augmentation, very nice!
- Training is fast, maybe even a little bit faster.
- Very memory efficient!

Cons:
- No progress bar, sad :(
- No built-in log.

## Accuracy
| Model            | Acc.        |
| ------------     | ----------- |
| VGG16            |       |
| VGG19            | 91.36%      |
| ResNet18         | 94.37%      |
| ResNet34         | 94.77%      |
| ResNet50         | 94.24%      |
| ResNet101        |       |
| ResNeXt29(32x4d) |       |
| ResNeXt29(2x64d) |       |
| DenseNet121      |       |

## Learning rate adjustment
I manually change the `lr` during training:
- `0.1` for epoch `[0,150)`
- `0.01` for epoch `[150,250)`
- `0.001` for epoch `[250,350)`

Resume the training with `python main.py --resume  saved_model_path`


