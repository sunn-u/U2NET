
class CfgNode:

    @ staticmethod
    def MODEL():
        return CfgNode.MODEL

    @staticmethod
    def DATA():
        return CfgNode.DATA

    @staticmethod
    def OPTIMIZER():
        return CfgNode.OPTIMIZER

    @staticmethod
    def TRAIN():
        return CfgNode.TRAIN

    @staticmethod
    def LOSS():
        return CfgNode.LOSS


def merge_from_file(self):
    pass

def get_config():
    from sunn_models.config.defaults import _C
    # return _C.clone()
    return _C
