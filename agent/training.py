
class Training:
    def __init__(self, device, batch_size):
        self.batch_size = batch_size
        self.device = device

    def run(self):
        print("Training started")
        self.print_parameters()


    def print_parameters(self):
        print(f"- batch size: {self.batch_size}")
    
