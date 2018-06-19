from enum import Enum

class DataType(Enum):
    trainData = 0
    valData = 1
    testData = 2

class MaskType(Enum):
    nucleusMask = 0
    spaceBetweenMask = 1
