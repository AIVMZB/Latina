# Help with building CNN model on PyTorch

## Layers documentations
- Firstly you can find useful information about Conv layers in torch by this [link](E:\Dyploma\Latina\LatinaProject\datasets). 
- MaxPool2d documentation [here](https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html).
- ReLU activation function [here](https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html).
- GeLU activation function [here](https://pytorch.org/docs/stable/generated/torch.nn.GELU.html).
- Linear transformation layer documentation [here](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html).

## Steps to create trained CNN model
1. Define the model itself. Create a class, which inherits from `torch.nn.Module`,  set layers and implement `forward` method.
2. Create custom `Dataset` class. Create also `Dataloader` from it with specific batch size.
3. Provide data augmentation for better training.
4. Setup training loop and run it.

## Where can you find needed info

### Model and training loop
Overall tutorial of building and training CNN on `torch` you can find [here](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html). 
The model itself is defined [here](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html).

### Dataset and dataloader
The complete tutorial of creating custom dataset and dataloader you can find [here](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html).
About augmentation you can read [here](https://pytorch.org/vision/stable/transforms.html).

### Train
The example of training loop you can find [here](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html). It also be useful to add validation loop in it.

### ML Flow (Optional)
To track your training process you can use ML FLow. The tutorial of it you can find by following the [link](https://mlflow.org/docs/latest/python_api/mlflow.pytorch.html).

# P.S.
I am sure you will be victorious in this task. I am very very glad and happy about my ability to build this project with <b>you</b>. Never lose hope in always yourself and keep going!!!
