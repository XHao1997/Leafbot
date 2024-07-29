from enum import Enum


class ARMTASK(Enum):
    MOVE_SINGLE_JOINT = 1
    MOVE_ALL_JOINT = 2
    MOVE_ARM_EYE2HAND = 3
    MOVE_ARM_MOVE_RANDOM = 4
    MOVE_ARM_FOR_GRAP_LEAF = 5
    MOVE_JOINT1_NN = 6
    MOVE_ZERO_POSITION = 7
    READ_SERVO = 8
    PICK_CLOSEST_LEAF = 9
    TRAIN_EYE2HAND = 10
    DONE = 0
    SET_TORQUE = 11
    OFF_TORQUE = 12
    IMITATION_MOVE = 13
    CALI_J1 = 14
    CALI_J1_imitation = 15
class Msg:
    def __init__(self, *args):
        if len(args) == 2:
            self.task = args[0]
            self.cmd = args[1]
        elif len(args) == 1:
            self.task = args[0]
            self.cmd = None
        else:
            self.task = None
            self.cmd = None


