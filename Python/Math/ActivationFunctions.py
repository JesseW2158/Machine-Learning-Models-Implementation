import math
from Util.DataStructures import *
from typing import Union
import builtins

class Linear:
    def __call__(self, x: Union[Array]):
        return x
    
    def gradient(self, x: Union[Array]):
        match x:
            case Array():
                return Array(self.__gradient(x))
            case int():
                return 0
        
    def __gradient(self, x: Union[Array]):
        gradient_list = []
        
        if len(x.shape()) == 1:
            return [1 for _ in range(x.shape()[0])]
        
        for element in x.array:
            gradient_list.append(self.__gradient(Array(element)))
        
        return gradient_list
    
    def __str__(self) -> str:
        return "Linear Activation"
    
class BinaryStep:
    def __call__(self, x: Union[Array]):
        match x:
            case Array():
                return Array(self.__call(x))
            case int():
                return 1 if x >= 0 else 0
    
    def __call(self, x: Union[Array]):
        call_list = []
        
        if len(x.shape()) == 1:
            return [1 if element >= 0 else 0 for element in x.array]
        
        for element in x.array:
            call_list.append(self.__call(Array(element)))
        
        return call_list
    
    def gradient(self, x: Union[Array]):
        match x:
            case Array():
                return Array(self.__gradient(x))
            case int():
                return 0
        
    
    def __gradient(self, x: Union[Array]):
        gradient_list = []
        
        if len(x.shape()) == 1:
            return [0 for _ in range(x.shape()[0])]
        
        for element in x.array:
            gradient_list.append(self.__gradient(Array(element)))
        
        return gradient_list
    
    def __str__(self) -> str:
        return "BinaryStep Activation"
    
class Sigmoid:
    def __call__(self, x: Union[Array, int]):
        match x:
            case Array():
                return Array(self.__call(x))
            case int():
                return 1 / (1 + math.e ** -x)
    
    def __call(self, x: Union[Array, int]):
        call_list = []
        
        if len(x.shape()) == 1:
            return [1 / (1 + math.e ** -element) for element in x.array]
        
        for element in x.array:
            call_list.append(self.__call(Array(element)))
        
        return call_list
            
    def gradient(self, x: Union[Array]):
        match x:
            case Array():
                return Array(self.__gradient(x))
            case int():
                return self(x) * (1 - self(x))
    
    def __gradient(self, x: Union[Array]):
        gradient_list = []
        
        if len(x.shape()) == 1:
            return [self(element) * (1 - self(element)) for element in x.array]
        
        for element in x.array:
            gradient_list.append(self.__gradient(Array(element)))
        
        return gradient_list
    
    def __str__(self) -> str:
        return "Sigmoid Activation"
    
class Tanh:
    def __call__(self, x: Union[Array, int]):
        match x:
            case Array():
                return Array(self.__call(x))
            case int():
                return (math.e ** x - math.e ** -x) / (math.e ** x + math.e ** -x)
            
    def __call(self, x: Union[Array, int]):
        call_list = []
        
        if len(x.shape()) == 1:
            return [(math.e ** element - math.e ** -element) / (math.e ** element + math.e ** -element) for element in x.array]
        
        for element in x.array:
            call_list.append(self.__call(Array(element)))
        
        return call_list
    
    def gradient(self, x: Union[Array]):
        match x:
            case Array():
                return Array(self.__gradient(x))
            case int():
                return 1 - self(x) ** 2
    
    def __gradient(self, x: Union[Array]):
        gradient_list = []
        
        if len(x.shape()) == 1:
            return [1 - self(element) ** 2 for element in x.array]
        
        for element in x.array:
            gradient_list.append(self.__gradient(Array(element)))
        
        return gradient_list
    
    def __str__(self) -> str:
        return "Tanh Activation"
    
class SoftSign:
    def __call__(self, x: Union[Array, int]):
        match x:
            case Array():
                return Array(self.__call(x))
            case int():
                return x / (1 + abs(x))
    
    def __call(self, x: Union[Array, int]):
        call_list = []
        
        if len(x.shape()) == 1:
            return [element / (1 + abs(element)) for element in x.array]
        
        for element in x.array:
            call_list.append(self.__call(Array(element)))
        
        return call_list
    
    def gradient(self, x: Union[Array]):
        match x:
            case Array():
                return Array(self.__gradient(x))
            case int():
                return 1 / (1 + abs(x)) ** 2
    
    def __gradient(self, x: Union[Array, int]):
        gradient_list = []
        
        if len(x.shape()) == 1:
            return [1 / (1 + abs(element)) ** 2 for element in x.array]
        
        for element in x.array:
            gradient_list.append(self.__gradient(Array(element)))
        
        return gradient_list
    
    def __str__(self) -> str:
        return "SoftSign Activation"
    
class ReLU:
    def __call__(self, x: Union[Array]):
        return max(0, x)

    def gradient(self, x: Union[Array]):
        return 0 if x < 0 else 1
    
    def __str__(self) -> str:
        return "ReLU Activation"
    
class SoftPlus:
    def __call__(self, x: Union[Array]):
        return math.log(1 + math.e ** x)
    
    def gradient(self, x: Union[Array]):
        return 1 / (1 + math.e ** -x)
    
    def __str__(self) -> str:
        return "SoftPlus Activation"

class ELU:
    def __call__(self, x: Union[Array], alpha=0.2) -> float:
        return alpha * (math.e ** x - 1) if x <= 0 else x
    
    def gradient(self, x: Union[Array], alpha=0.2) -> float:
        return alpha * (math.e ** x) if x <= 0 else 1
    
    def __str__(self) -> str:
        return "ELU Activation"
    
class SELU:
    def __init__(self) -> None:
        self.scale = 1.0507009873554804934193349852946 
        self.alpha = 1.6732632423543772848170429916717
    
    def __call__(self, x: Union[Array]):
        return self.scale * ELU(x)
    
    def gradient(self, x: Union[Array]):
        return self.scale * ELU().gradient(x)
    
    def __str__(self) -> str:
        return "SELU Activation"
    
class LeakyReLU:
    def __init__(self, alpha=0.01) -> None:
        self.alpha = alpha
    
    def __call__(self, x: Union[Array]):
        return self.alpha * x if x <= 0 else x
    
    def gradient(self, x: Union[Array]):
        return self.alpha if x < 0 else 1
    
    def __str__(self) -> str:
        return "Leaky ReLU Activation"
    
class PReLU:
    def __init__(self, alpha) -> None:
        self.alpha = alpha
        
    def __call__(self, x: Union[Array]):
        return self.alpha * x if x < 0 else x
    
    def gradient(self, x: Union[Array]):
        return self.alpha if x < 0 else 1
    
    def __str__(self) -> str:
        return "PReLU Activation"
    
class SiLU:
    def __call__(self, x: Union[Array]):
        return x / (1 + math.e ** -x)
    
    def gradient(self, x: Union[Array]):
        return (1 + math.e ** -x + x * math.e ** -x) / (1 + math.e ** -x) ** 2
    
    def __str__(self) -> str:
        return "SiLU Activation"

class Sinusoid:
    def __call__(self, x: Union[Array]):
        return math.sin(x)
    
    def gradient(self, x: Union[Array]):
        return math.cos(x)
    
    def __str__(self) -> str:
        return "Sinusoid Activation"
    
# TODO
# class Softmax:
#     def __call__(self, x: Union[Array]):
        