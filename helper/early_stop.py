class EarlyStopperByAUC:
    def __init__(self, patience: int = 1):
        self.patience = patience
        self.counter = 0
        self.max_roc_score = float("-inf")

    def early_stop(self, validation_roc: float) -> bool:
        if validation_roc > self.max_roc_score:
            self.max_roc_score = validation_roc
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

class EarlyStopperByLoss:
    def __init__(self, patience: int=1):
        self.patience = patience
        self.counter = 0
        self.min_validation_loss = float("inf")

    def early_stop(self, validation_loss: float)-> bool:
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss >= self.min_validation_loss:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

