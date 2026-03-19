import os
import random
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as tfs
import re

class RainDataset(data.Dataset):
    def __init__(self, path, mode='train', patch_size=64):
        super(RainDataset, self).__init__()
        self.size = patch_size
        self.mode = mode

        # 🌟 智能识别雷达全开：加入 DID-Data 的识别！
        self.is_spa = 'spa' in path.lower()
        self.is_rain12600 = 'rain1400rain12600' in path.lower()
        self.is_rain100h_custom = 'raintrainh-rain100h' in path.lower()
        self.is_rain100l_custom = 'raintrainl-rain100l' in path.lower()
        self.is_ddn = 'ddn-data' in path.lower()
        self.is_did = 'did-data' in path.lower()  # 🌟 新增：专属 DID-Data 识别雷达

        # ================= 1. 智能适配目录结构 =================
        if self.is_spa:
            actual_mode = 'test' if mode == 'validation' else mode
            self.rain_path = os.path.join(path, actual_mode, 'rain')
            self.clean_path = os.path.join(path, actual_mode, 'norain')
            temp_list = sorted(os.listdir(self.rain_path))

        elif self.is_ddn or self.is_did:
            # 🌟 核心修复：DID-Data 和 DDN 一样，验证集叫 test，且子文件夹为 input/target
            folder_mode = 'test' if mode == 'validation' else mode
            self.rain_path = os.path.join(path, folder_mode, 'input')
            self.clean_path = os.path.join(path, folder_mode, 'target')
            temp_list = sorted(os.listdir(self.rain_path))

        elif self.is_rain100h_custom:
            if mode == 'train':
                self.rain_path = os.path.join(path, 'RainTrainH', 'rain')
                self.clean_path = os.path.join(path, 'RainTrainH', 'norain')
                temp_list = sorted(os.listdir(self.clean_path))
            else:
                self.rain_path = os.path.join(path, 'Rain100H', 'rainy')
                self.clean_path = os.path.join(path, 'Rain100H', 'norainly')
                temp_list = sorted(os.listdir(self.rain_path))

        elif self.is_rain100l_custom:
            if mode == 'train':
                self.rain_path = os.path.join(path, 'RainTrainL', 'rain')
                self.clean_path = os.path.join(path, 'RainTrainL', 'norain')
            else:
                self.rain_path = os.path.join(path, 'Rain100L', 'rainy')
                self.clean_path = os.path.join(path, 'Rain100L', 'norain')
            temp_list = sorted(os.listdir(self.rain_path))

        elif self.is_rain12600:
            if mode == 'train':
                self.rain_path = os.path.join(path, 'Rain12600', 'rainy_image')
                self.clean_path = os.path.join(path, 'Rain12600', 'ground_truth')
            else:
                self.rain_path = os.path.join(path, 'Rain1400', 'Rain1400', 'rainy_image')
                self.clean_path = os.path.join(path, 'Rain1400', 'Rain1400', 'ground_truth')
            temp_list = sorted(os.listdir(self.rain_path))

        else:
            self.rain_path = os.path.join(path, mode, 'input')
            self.clean_path = os.path.join(path, mode, 'target')
            temp_list = sorted(os.listdir(self.rain_path))

        if not os.path.exists(self.rain_path) or not os.path.exists(self.clean_path):
            raise FileNotFoundError(
                f"🚨 路径不存在，请检查: \n雨图: {self.rain_path} \n干净图: {self.clean_path}")

        self.img_list = [x for x in temp_list if x.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif'))]
        print(f"[{mode}] 成功加载了 {len(self.img_list)} 张图片.")

    def __getitem__(self, index):
        base_name = self.img_list[index]
        rain_name = base_name
        clean_name = base_name

        # ================= 2. 核心修复：智能转换文件名 =================
        if self.is_ddn or self.is_did:
            # 🌟 DID-Data 雨图和干净图名字完全一样，什么都不用改！
            pass

        elif self.is_rain100h_custom:
            if self.mode == 'train':
                clean_name = base_name
                rain_name = base_name.replace('norain-', 'rain-')
            else:
                rain_name = base_name
                clean_name = base_name.replace('rain-', 'norain-')

        elif self.is_rain100l_custom:
            rain_name = base_name
            clean_name = base_name.replace('rain-', 'norain-')

        elif self.is_spa and self.mode != 'train':
            match = re.search(r'\d+', base_name)
            if match:
                img_id = match.group()
                if os.path.exists(os.path.join(self.clean_path, f"norain-{img_id}.png")):
                    clean_name = f"norain-{img_id}.png"
                elif os.path.exists(os.path.join(self.clean_path, f"{img_id}.png")):
                    clean_name = f"{img_id}.png"

        elif self.is_rain12600:
            base_name_split = rain_name.split('_')[0]
            ext = rain_name.split('.')[-1]
            clean_name = f"{base_name_split}.{ext}"

        elif rain_name.startswith('rain-') and ('Rain100' in self.rain_path):
            clean_name = rain_name.replace('rain-', 'norain-')

        rain_file = os.path.join(self.rain_path, rain_name)
        clean_file = os.path.join(self.clean_path, clean_name)

        try:
            rain_img = Image.open(rain_file).convert('RGB')
            clean_img = Image.open(clean_file).convert('RGB')
        except OSError:
            print(f"⚠️ 无法读取图片: {rain_name} 或 {clean_name}，跳过该图")
            return self.__getitem__((index + 1) % len(self.img_list))

        if self.mode == 'train':
            w, h = rain_img.size
            if h < self.size or w < self.size:
                rain_img = tfs.Resize((self.size, self.size))(rain_img)
                clean_img = tfs.Resize((self.size, self.size))(clean_img)
            else:
                i = random.randint(0, h - self.size)
                j = random.randint(0, w - self.size)
                rain_img = tfs.functional.crop(rain_img, i, j, self.size, self.size)
                clean_img = tfs.functional.crop(clean_img, i, j, self.size, self.size)

            if random.random() < 0.5:
                rain_img = tfs.functional.hflip(rain_img)
                clean_img = tfs.functional.hflip(clean_img)
            if random.random() < 0.5:
                rain_img = tfs.functional.vflip(rain_img)
                clean_img = tfs.functional.vflip(clean_img)
        else:
            w, h = rain_img.size
            max_dim = 1024
            if w > max_dim or h > max_dim:
                scale = min(max_dim / w, max_dim / h)
                new_w = int(w * scale)
                new_h = int(h * scale)
            else:
                new_w, new_h = w, h
            new_w = new_w - (new_w % 16)
            new_h = new_h - (new_h % 16)
            if new_w != w or new_h != h:
                rain_img = rain_img.resize((new_w, new_h), Image.BICUBIC)
                clean_img = clean_img.resize((new_w, new_h), Image.BICUBIC)

        rain_tensor = tfs.ToTensor()(rain_img)
        clean_tensor = tfs.ToTensor()(clean_img)
        return rain_tensor, clean_tensor

    def __len__(self):
        return len(self.img_list)