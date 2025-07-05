import math
from Util.DataStructures import *
from typing import Union

# ADD IN WHAT THE PURPOSE OF EACH ACTIVATION FUNCTION IS
# TEST ALL ACTIVATION FUNCTIONS

class Linear:
    def __call__(self, x: Union[Array]):
        return x
    
    # BASED OFF SHAPE OF X, CONSTRUCT NEW ARRAY WITH ONLY 1
    def gradient(self, x: Union[Array]):
        return x.shape()
    
    def __str__(self) -> str:
        return "Linear"
    
class BinaryStep:
    def __call__(self, x: Union[Array]):
        return 1 if x >= 0 else 0
    
    def gradient(self, x: Union[Array]):
        return 0
    
    def __str__(self) -> str:
        return "BinaryStep"
    
class Sigmoid:
    def __call__(self, x: Union[Array]):
        return 1 / 1 + (math.e ** -x)
    
    def gradient(self, x: Union[Array]):
        return self(x) * (1 - self(x))
    
    def __str__(self) -> str:
        return "Sigmoid"
    
class Tanh:
    def __call__(self, x: Union[Array]):
        return (math.e ** x - math.e ** -x) / (math.e ** x + math.e ** -x)

    def gradient(self, x: Union[Array]):
        return 1 - self(x) ** 2
    
    def __str__(self) -> str:
        return "Tanh"
    
class SoftSign:
    def __call__(self, x: Union[Array]):
        return x / (1 + abs(x))
    
    def gradient(self, x: Union[Array]):
        return 1 / (1 + abs(x)) ** 2
    
    def __str__(self) -> str:
        return "SoftSign"
    
class ReLU:
    def __call__(self, x: Union[Array]):
        return max(0, x)

    def gradient(self, x: Union[Array]):
        return 0 if x < 0 else 1
    
    def __str__(self) -> str:
        return "ReLU"
    
class SoftPlus:
    def __call__(self, x: Union[Array]):
        return math.log(1 + math.e ** x)
    
    def gradient(self, x: Union[Array]):
        return 1 / (1 + math.e ** -x)
    
    def __str__(self) -> str:
        return "SoftPlus"

class ELU:
    def __call__(self, x: Union[Array], alpha=0.2) -> float:
        return alpha * (math.e ** x - 1) if x <= 0 else x
    
    def gradient(self, x: Union[Array], alpha=0.2) -> float:
        return alpha * (math.e ** x) if x <= 0 else 1
    
    def __str__(self) -> str:
        return "ELU"
    
class SELU:
    def __init__(self) -> None:
        self.scale = 1.0507009873554804934193349852946 
        self.alpha = 1.6732632423543772848170429916717
    
    def __call__(self, x: Union[Array]):
        return self.scale * ELU(x)
    
    def gradient(self, x: Union[Array]):
        return self.scale * ELU().gradient(x)
    
    def __str__(self) -> str:
        return "SELU"
    
class LeakyReLU:
    def __init__(self, alpha=0.01) -> None:
        self.alpha = alpha
    
    def __call__(self, x: Union[Array]):
        return self.alpha * x if x <= 0 else x
    
    def gradient(self, x: Union[Array]):
        return self.alpha if x < 0 else 1
    
    def __str__(self) -> str:
        return "Leaky ReLU"
    
class PReLU:
    def __init__(self, alpha) -> None:
        self.alpha = alpha
        
    def __call__(self, x: Union[Array]):
        return self.alpha * x if x < 0 else x
    
    def gradient(self, x: Union[Array]):
        return self.alpha if x < 0 else 1
    
    def __str__(self) -> str:
        return "PReLU"
    
class SiLU:
    def __call__(self, x: Union[Array]):
        return x / (1 + math.e ** -x)
    
    def gradient(self, x: Union[Array]):
        return (1 + math.e ** -x + x * math.e ** -x) / (1 + math.e ** -x) ** 2
    
    def __str__(self) -> str:
        return "SiLU"

class Sinusoid:
    def __call__(self, x: Union[Array]):
        return math.sin(x)
    
    def gradient(self, x: Union[Array]):
        return math.cos(x)
    
    def __str__(self) -> str:
        return "Sinusoid"
    
# TODO
# class Softmax:
#     def __call__(self, x: Union[Array]):
        