from typing import Union

# TODO implement __format_string before shape()

class Array:
    def __init__(self, array: Union[list, int]):
        if not isinstance(array, list) and not isinstance(array, int):
            raise Exception("Invalid array passed")
        
        if not self.__assert_equality(array):
            raise Exception("Unequal array passed")
        
        if not self.__assert_int_array(array):
            raise Exception("Array contains non-integers")
        # check if all object types are the same in array
        
        self.array = array
        
    def __assert_int_array(self, array):
        if isinstance(array[0], int) or isinstance(array[0], float):
            return all(type(element) == int for element in array) or all(type(element) == float for element in array)
        
        if not isinstance(array[0], list):
            print("here1")
            return False
        
        for element in array:
            if not self.__assert_int_array(element):
                print("here2")
                return False
            
        return True
        
    def __assert_equality(self, array: Union[list, int]):
        # checks if the array is a int, used for the first pass through
        if isinstance(array, int):
            return True
        
        # again for the first pass through, checks for a empty list
        if len(array) == 0:
            return True
        
        # keeps track of last list to ensure equal sizes between all lists
        size = len(array[0]) if isinstance(array[0], list) else 0
        
        for element in array:
            # checks if the inner array is a int array since if it wasnt a list, its a normal int array. Wont break since the previous recursive loop checked the size of the array already and made sure it was consistent with the rest of them
            if not isinstance(element, list):
                return True
            
            # checks if inner array is the same size as rest of the arrays
            if len(element) != size:
                return False
            
            if not self.__assert_equality(element):
                return False
        
        return True

    def __str__(self) -> str:
        return f"{self.__format_array(self.array, 0)}"

    def __format_array(self, array: list, depth: int):
        return_str = ' ' * depth + '['
        
        if not isinstance(array[0], list):
            return " " * depth + str(array)
        
        for element in array:
            return_str += f"\n{self.__format_array(element, depth + 1)}"
            
        return(return_str + '\n' + ' ' * depth + ']')
        
    def shape(self) -> tuple:
        return self.__shape(self.array)
    
    def __shape(self, array:list) -> int:
        if isinstance(array[0], int):
            return [len(array)]
        
        return [len(array)] + self.__shape(array[0])
    