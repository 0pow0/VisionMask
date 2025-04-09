import torch
import torchvision.transforms as T
import torch.nn.functional as F
from PIL import Image
from pathlib import Path

ATARI_ENVS_INFO = {
    'Enduro': {
        'n_actions': 9,
        'content_height': 104,
        'content_width': 152,
        'content_top': 51,
        'content_left': 8
    },
    'Seaquest': {
        'n_actions': 18,
        'content_height': 132,
        'content_width': 152,
        'content_top': 56,
        'content_left': 8 
    },
    'MsPacman': {
        'n_actions': 9,
        'content_height': 172,
        'content_width': 160,
        'content_top': 0,
        'content_left': 0
    }
}

def get_atari_number_of_actions(env: str):
    return ATARI_ENVS_INFO[env]['n_actions']

# img: (batch, 3, H, W * frame)
def atari_seperate_content_info(img: torch.Tensor, atari_env: str):
    if atari_env == "Seaquest":
        return seaquest_seperate_content_info(img)
    elif atari_env == "Enduro":
        return enduro_seperate_content_info(img)
    elif atari_env == "MsPacman":
        return mspacman_seperate_content_info(img)

# (batch, 3, 132, 152 * frame), (batch, 3, H, W * frame)    
def atari_concat_content_info(content: torch.Tensor, info: torch.Tensor, atari_env: str):
    if atari_env == "Seaquest":
        return seaquest_concat_content_info(content=content, info=info)
    elif atari_env == "Enduro":
        return enduro_concat_content_info(content=content, info=info)
    elif atari_env == "MsPacman":
        return mspacman_concat_content_info(content=content, info=info)

# img: (batch, 3, H, W * frame)
def mspacman_seperate_content_info(img: torch.Tensor):
    W_times_frame = img.shape[3]

    h = ATARI_ENVS_INFO['MsPacman']['content_height']
    w = ATARI_ENVS_INFO['MsPacman']['content_width']
    top = ATARI_ENVS_INFO['MsPacman']['content_top']
    left = ATARI_ENVS_INFO['MsPacman']['content_left']

    # (batch, 3, H-31, W * frame)
    content = T.functional.crop(img, top=top, left=left, height=h, width=W_times_frame)
    # (batch, 3, 31, W * frame)
    info = T.functional.crop(img, top=h, left=0, height=210-h, width=W_times_frame)

    return content, info

# (batch, 3, 172, W * frame), (batch, 3, 38, W * frame)    
def mspacman_concat_content_info(content: torch.Tensor, info: torch.Tensor):
    return torch.cat((content, info), dim=2)

class MsPacmanBackgroundAdapter():
    # img: (batch, 3, H, W * frame)
    def get_content_background(self, img):
        batch_size, _, _, W_times_frame = img.shape
        h = ATARI_ENVS_INFO['MsPacman']['content_height']
        background_color = torch.tensor([  0,  28, 136], dtype=torch.uint8)

        # (bacth, 3, 172, W * frame)
        return background_color.reshape(1, 3, 1, 1).repeat(batch_size, 1, h, W_times_frame).to(img.device)

# img: (batch, 3, H, W * frame)
def seaquest_seperate_content_info(img: torch.Tensor):

    h = ATARI_ENVS_INFO['Seaquest']['content_height']
    w = ATARI_ENVS_INFO['Seaquest']['content_width']
    top = ATARI_ENVS_INFO['Seaquest']['content_top']
    left = ATARI_ENVS_INFO['Seaquest']['content_left']

    # [(batch, 3, H, W) * frame]
    frames = img.split(160, 3)

    # (batch, 3, H, W)
    mask = torch.ones_like((frames[0]))
    mask[:, :, top : top + h, left : left + w] = 0.0

    # [(batch, 3, 132, 152) * frame]
    contents = []
    # [(batch, 3, H, W) * frame]
    infos = []
    for frame in frames:
        # (batch, 3, 132, 152)
        contents.append(T.functional.crop(frame, top=top, left=left, height=h, width=w))
        infos.append(frame * mask)
    
    # (batch, 3, 132, 152 * frame)
    content = torch.cat(contents, dim=3)
    # (batch, 3, H, W * frame)
    info = torch.cat(infos, dim=3)

    return content, info

# (batch, 3, 132, 152 * frame), (batch, 3, H, W * frame)    
def seaquest_concat_content_info(content: torch.Tensor, info: torch.Tensor):
    device = info.device

    # [(batch, 3, 132, 152) * frame]
    sep_content = content.split(152, 3)
    # [(batch, 3, H, W) * frame]
    sep_info = info.split(160, 3)

    batch_size, _, h, w = sep_info[0].size()
    _, _, hc, wc = sep_content[0].size()

    # (batch, 3, H, W)
    masked_image = []

    content_h = ATARI_ENVS_INFO['Seaquest']['content_height']
    content_w = ATARI_ENVS_INFO['Seaquest']['content_width']
    content_top = ATARI_ENVS_INFO['Seaquest']['content_top']
    content_left = ATARI_ENVS_INFO['Seaquest']['content_left']

    # (batch, 3, H, W)
    info_mask = torch.ones_like((sep_info[0]), device=device)
    info_mask[:, :, content_top : content_top + content_h, content_left : content_left + content_w] = 0.0
    inversed_info_mask = torch.ones_like(info_mask, device=device) - info_mask

    pad_left = (w - wc)
    pad_right = 0
    pad_top = content_top 
    pad_bottom = (h - content_top - hc)

    for i in range(4):
        padded_sep_content = F.pad(sep_content[i], (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
        masked_image.append(info_mask * sep_info[i] + inversed_info_mask * padded_sep_content)

    # (batch, 3, H, W * frame) 
    masked_image = torch.cat(masked_image, dim=3)
    return masked_image

class SeaquestBackgroundAdapter():
    def __init__(self) -> None:
        transformer = T.PILToTensor()
        background = Image.open(str(Path(__file__).parent / 'background' / 'seaquest_background.png')).convert("RGB")
        self.background = transformer(background)

    # img: (batch, 3, H, W * frame)
    def get_content_background(self, img):
        batch_size, _, H, W_times_frame = img.size()

        # (batch, 3, 132, 152 * frame) 
        return self.background.repeat(batch_size, 1, 1, 1).to(img.device)

class EnduroBackgroundAdapter():
    def __init__(self) -> None:
        pass

    # img: (3, H, W * frame)
    def get_background_color(self, img):
        top = ATARI_ENVS_INFO['Enduro']['content_top']
        left = ATARI_ENVS_INFO['Enduro']['content_left']
        return img[:, top, left].to(img.device)
   
    # img: (batch, 3, H, W * frame)
    def get_content_background(self, img):
        batch_size, _, H, W_times_frame = img.size()

        # (batch, 3, 104, 152 * frame) 
        backgrounds = torch.empty(batch_size, 3, 104, 152 * 4, device=img.device)

        for i in range(batch_size):
            # (3, H, W * frame)
            frames = img[i]
            # (3)
            color = self.get_background_color(frames)
            # (3, 104, 152 * frame)
            backgrounds[i] = (torch.ones(104, 152 * 4, 3, device=img.device) * color).permute(2, 0, 1)
       
        # (batch, 3, 104, 152 * frame)
        return backgrounds

# (batch, 3, 104, 152 * frame), (batch, 3, H, W * frame)    
def enduro_concat_content_info(content: torch.Tensor, info: torch.Tensor):
    device = info.device

    # [(batch, 3, 104, 152) * frame]
    sep_content = content.split(152, 3)
    # [(batch, 3, H, W) * frame]
    sep_info = info.split(160, 3)

    batch_size, _, h, w = sep_info[0].size()
    _, _, hc, wc = sep_content[0].size()

    # (batch, 3, H, W)
    masked_image = []

    content_h = ATARI_ENVS_INFO['Enduro']['content_height']
    content_w = ATARI_ENVS_INFO['Enduro']['content_width']
    content_top = ATARI_ENVS_INFO['Enduro']['content_top']
    content_left = ATARI_ENVS_INFO['Enduro']['content_left']

    # (batch, 3, H, W)
    info_mask = torch.ones_like((sep_info[0]), device=device)
    info_mask[:, :, content_top : content_top + content_h, content_left : content_left + content_w] = 0.0
    inversed_info_mask = torch.ones_like(info_mask, device=device) - info_mask

    pad_left = (w - wc)
    pad_right = 0
    pad_top = content_top 
    pad_bottom = (h - content_top - hc)

    for i in range(4):
        padded_sep_content = F.pad(sep_content[i], (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
        masked_image.append(info_mask * sep_info[i] + inversed_info_mask * padded_sep_content)

    # (batch, 3, H, W * frame) 
    masked_image = torch.cat(masked_image, dim=3)
    return masked_image

# img: (batch, 3, H, W * frame)
def enduro_seperate_content_info(img: torch.Tensor):

    h = ATARI_ENVS_INFO['Enduro']['content_height']
    w = ATARI_ENVS_INFO['Enduro']['content_width']
    top = ATARI_ENVS_INFO['Enduro']['content_top']
    left = ATARI_ENVS_INFO['Enduro']['content_left']

    # [(batch, 3, H, W) * frame]
    frames = img.split(160, 3)

    # (batch, 3, H, W)
    mask = torch.ones_like((frames[0]))
    mask[:, :, top : top + h, left : left + w] = 0.0

    # [(batch, 3, 104, 152) * frame]
    contents = []
    # [(batch, 3, H, W) * frame]
    infos = []
    for frame in frames:
        # (batch, 3, 104, 152)
        contents.append(T.functional.crop(frame, top=top, left=left, height=h, width=w))
        infos.append(frame * mask)
    
    # (batch, 3, 104, 152 * frame)
    content = torch.cat(contents, dim=3)
    # (batch, 3, H, W * frame)
    info = torch.cat(infos, dim=3)

    return content, info 

# segmentations: (batch, num_actions, H_c, W_c * frame)
def extract_masks_atari(segmentations, target_vectors):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    batch_size, num_classes, h_content, w_times_frames = segmentations.size()

    # (batch, H_c, W_c * frame)
    target_masks = torch.empty(batch_size, h_content, w_times_frames, device=device)
    non_target_masks = torch.empty(batch_size, h_content, w_times_frames, device=device)
    for i in range(batch_size):
        class_indices = target_vectors[i].eq(1.0)
        non_class_indices = target_vectors[i].eq(0.0)
        target_masks[i] = (segmentations[i][class_indices]).amax(dim=0)
        non_target_masks[i] = (segmentations[i][non_class_indices]).amax(dim=0)

    return target_masks.sigmoid(), non_target_masks.sigmoid()