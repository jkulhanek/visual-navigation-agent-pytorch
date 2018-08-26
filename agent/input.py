import numpy as np
import torch
from PIL import Image
import torchvision.transforms as T

resize = T.Compose([T.ToPILImage(),
                    T.Resize(40, interpolation=Image.CUBIC),
                    T.ToTensor()])

def get_screen(env, device):
    # transpose into torch order (CHW)
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))  

    # Strip off the top and bottom of the screen
    screen = screen[:, 160:320]

    # Convert to float, rescare, convert to torch tensor
    # (this doesn't require a copy)
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    # Resize, and add a batch dimension (BCHW)
    return resize(screen).unsqueeze(0).to(device)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from environment import make

    env = make("unity")
    env.start()

    plt.figure()
    plt.imshow(get_screen(env, torch.device("cpu")).cpu().squeeze(0).permute(1, 2, 0).numpy(),
            interpolation='none')
    plt.title('Example extracted screen')
    plt.show()