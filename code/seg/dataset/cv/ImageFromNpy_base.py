import json
from torch.utils.data import Dataset
import cv2
import os
import numpy as np
from PIL import Image


class ImageFromNpyDataset_V1(Dataset):
    #for training or validation
    def __init__(self, config, mode, *args, **params):
        self.config = config
        self.mode = mode
        self.data_path = config.get("data", "%s_data_path" % mode)
        self.label_path = config.get("data", "%s_label_path" % mode)

        self.target_label = [int(val) for val in config.get("data", "target_label").split(",")]

        self.load_mem = config.getboolean("data", "load_into_mem")
        self.file_list = os.listdir(self.data_path)
        self.data_list = [None] * len(self.file_list)
        self.file_record = [0] * len(self.file_list)

        if self.load_mem:
            for file in self.file_list:
                data_file = os.path.join(self.data_path, file)
                label_file = os.path.join(self.label_path, file)
                self.data_list.append({"data": np.load(data_file),
                                        "label": self.process_label(label_file),
                                        "file": file})

    def process_label(self, label_file):
        raw_label = np.load(label_file)
        real_label = raw_label[self.target_label, :, :]
        if len(real_label.shape) == 2:
            real_label = real_label[np.newaxis, :, :]
        return real_label

    def __getitem__(self, item):
        if self.load_mem:
            return {
                "data": self.data_list[item]["data"],
                "label": self.data_list[item]["label"],
                "file": self.data_list[item]["file"]
            }
        else:
            if self.file_record[item] == 0:
                self.file_record[item] = 1
                data_file = os.path.join(self.data_path, self.file_list[item])
                label_file = os.path.join(self.label_path, self.file_list[item])
                self.data_list[item] = {"data": np.load(data_file),
                                        "label": self.process_label(label_file),
                                        "file": self.file_list[item]}
            return {
                "data": self.data_list[item]["data"],
                "label": self.data_list[item]["label"],
                "file": self.data_list[item]["file"]
            }

    def __len__(self):
        return len(self.file_list)


class ImageFromNpyDataset_test(Dataset):
    # for testing or inferencing
    def __init__(self, config, mode, *args, **params):
        self.config = config
        self.mode = mode
        assert self.mode == "test"
        self.data_path = params["input"]
        self.data_path_length = len(self.data_path)
        self.load_mem = config.getboolean("data", "load_into_mem")
        self.file_list = []
        self.search_file(self.data_path)
        self.data_list = [None] * len(self.file_list)
        self.file_record = [0] * len(self.file_list)

    def search_file(self, file_dir):
        file_list = os.listdir(file_dir)
        for file in file_list:
            fullfile = os.path.join(file_dir, file)
            if os.path.isdir(fullfile):
                self.search_file(fullfile)
                continue
            self.file_list.append(fullfile[self.data_path_length + 1:])

    def __getitem__(self, item):
        if self.load_mem:
            return {
                "data": self.data_list[item]["data"],
                "file": self.data_list[item]["file"]
            }
        else:
            data_file = os.path.join(self.data_path, self.file_list[item])
            if data_file[-4:] == ".npy":
                image = np.load(data_file)
                return {"data": image,
                        "file": self.file_list[item]}
            else:
                flag = 1
                try:
                    image = Image.open(data_file).verify()
                except Exception as e:
                    flag = 0
                if flag == 0:
                    return {"empty": 0}
                else:
                    image = Image.open(data_file)
                    image = cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR)
                    image = cv2.resize(image, (512, 512))
                    return {"data": image,
                            "file": self.file_list[item]}


    def __len__(self):
        return len(self.file_list)

