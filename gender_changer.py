import os
import cv2
import torch
import argparse
from PIL import Image
from torchvision.transforms import transforms

from tqdm import tqdm
from configs import hyperparameters, global_config
from training.coaches.base_coach import BaseCoach

def parse_args():
    parser = argparse.ArgumentParser(description='gender changer')
    parser.add_argument('--i', default='0.jpg', help='input image path')      
    parser.add_argument('--o', default='0_out.png', help='output image path')        
    args = parser.parse_args()
    return args

class SingleIDCoach(BaseCoach):

    def __init__(self, data_loader, use_wandb):
        super().__init__(data_loader, use_wandb)

    def train(self, fname, image):
        use_ball_holder = True

        image_name = fname[0]

        self.restart_training()

        w_pivot = self.calc_inversions(image, image_name)

        w_pivot = w_pivot.to(global_config.device)

        # torch.save(w_pivot, f'{image_name}.pt')
        log_images_counter = 0
        real_images_batch = image.to(global_config.device)

        for i in tqdm(range(hyperparameters.max_pti_steps)):

            generated_images = self.forward(w_pivot)
            loss, l2_loss_val, loss_lpips = self.calc_loss(generated_images[0,], real_images_batch, image_name,
                                                            self.G, use_ball_holder, w_pivot)

            self.optimizer.zero_grad()

            if loss_lpips <= hyperparameters.LPIPS_value_threshold:
                break

            loss.backward()
            self.optimizer.step()

            use_ball_holder = global_config.training_step % hyperparameters.locality_regularization_interval == 0

            global_config.training_step += 1
            log_images_counter += 1

        # torch.save(self.G, f'{image_name}.pt')

        return w_pivot, self.G

def run_PTI(input_path):

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = global_config.cuda_visible_devices

    global_config.pivotal_training_steps = 1
    global_config.training_step = 1
        
    source_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    coach = SingleIDCoach(None, False)

    return coach.train(os.path.splitext(input_path)[0], source_transform(Image.open(input_path).convert('RGB')))

if __name__ == '__main__':

    args = parse_args()
    w_pivot, G = run_PTI(args.i)

    new_image = G.synthesis(w_pivot, noise_mode='const', force_fp32 = True)
    new_image = (new_image.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).detach().cpu().numpy()[0]
    cv2.imwrite(args.o, cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR))
