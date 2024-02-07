import torch.nn as nn


class ClassificationHead(nn.Module):

    def __init__(self, input_dimensions, output_dimensions):

        super(ClassificationHead, self).__init__()

        self.classifier = nn.Linear(input_dimensions, output_dimensions, bias=True)

    def forward(self, x):

        output = self.classifier(x)

        return output
