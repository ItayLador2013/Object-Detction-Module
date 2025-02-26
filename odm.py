import torch
import torchvision.transforms as T
from PIL import Image
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import torchvision
import os
import torch.optim as optim
from torch.utils.data import DataLoader
import xml.etree.ElementTree as ET
from tqdm import tqdm

RED = "\033[31m"
END_COLOR = "\033[0m"

class Label():
    """Label for predictions"""
    def __init__(self, name : str, font_color="white", line_color="red"):
        self.name = name
        self.font_color = font_color
        self.line_color = line_color

class ODM():
    """Object Detection Module"""
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def _check_annotations_folder(root, annotations_dir) -> None:
        anns_path = os.path.join(root, annotations_dir)

        if not os.path.isdir(anns_path):
            raise FileNotFoundError(f"ERROR: Annotations directory does not exist at specified location: {anns_path}")

        files = os.listdir(anns_path)

        if not bool(files) and all(file.endswith('.xml') for file in files):
            raise AttributeError("ERROR: Make sure all files in the attribiutes folder end with .xml")

    def _get_annotations(root, annotations_dir, annotations : list[str]):
        anns = []
        for i in range(len(annotations)):
            ann_path = os.path.join(root, annotations_dir, annotations[i])
            tree = ET.parse(ann_path)
            tree_root = tree.getroot()
            if tree_root.find("filename") is not None and tree_root.find("object") is not None:
                for obj in tree_root.findall("object"):
                    if obj.find("name") is not None and obj.find("bndbox") is not None:
                        bbox = obj.find("bndbox")
                        x_min = bbox.find("xmin")
                        y_min = bbox.find("ymin")
                        x_max = bbox.find("xmax")
                        y_max = bbox.find("ymax")
                        if x_min is not None and y_min is not None and x_max is not None and y_max is not None:
                            try:
                                int(x_min.text)
                                int(y_min.text)
                                int(x_max.text)
                                int(y_max.text)
                                anns.append(annotations[i])
                                break
                            except TypeError:
                                print("error")
                                continue
        return anns

    def _get_images(root, img_dir, annotations_dir, annotations : list[str]):
        imgs = []
        invalid_indexes = []
        img_path = os.path.join(root, img_dir)
        if not os.path.isdir(img_path):
            raise FileNotFoundError("Images directory does not exist at specified location")

        for i in range(len(annotations)):
            ann_path = os.path.join(root, annotations_dir, annotations[i])
            tree = ET.parse(ann_path)
            tree_root = tree.getroot()
            img_filename = tree_root.find("filename").text
            img_path = os.path.join(root, img_dir, img_filename)
            try:
                if not os.path.isfile(img_path):
                    raise FileNotFoundError(f"Image file not found for annotation {RED}{annotations[i]}{END_COLOR}. Discarding the annotation for training.")
            except FileNotFoundError as e:
                print(e)
                invalid_indexes.append(i)
                continue
            imgs.append(img_filename)

        for i in invalid_indexes:
            annotations.pop(i)
        return imgs
    
    def _find_label(self, name):
         for l in self.labels:
             if l == None:
                 continue
             if l.name == name:
                 return l
         raise AttributeError("Label does not exist in plans") 
    
    class _Data(torch.utils.data.Dataset):
        def __init__(self, labels : list[Label], root, IMAGES_DIR, ANNOTATIONS_DIR, transforms=None):
            self.IMAGES_DIR = IMAGES_DIR
            self.ANNOTATIONS_DIR = ANNOTATIONS_DIR
            self.root = root
            self.transforms = transforms
            ODM._check_annotations_folder(root, self.ANNOTATIONS_DIR)
            self.anns = sorted(os.listdir(os.path.join(root, ANNOTATIONS_DIR)))
            self.anns = ODM._get_annotations(root, ANNOTATIONS_DIR, self.anns)
            self.imgs = ODM._get_images(root, IMAGES_DIR, ANNOTATIONS_DIR, self.anns)
            self.labels : list[Label] = labels
        
        def _find_label(self, name):
            for l in self.labels:
                if l == None:
                    continue
                if l.name == name:
                    return l
            raise AttributeError("Label does not exist in plans") 
        
        def __getitem__(self, index):
            img_path = os.path.join(self.root, self.IMAGES_DIR, self.imgs[index])
            anns_path = os.path.join(self.root, self.ANNOTATIONS_DIR, self.anns[index])
            image = Image.open(img_path).convert("RGB")
            tree = ET.parse(anns_path)
            root = tree.getroot()

            boxes = []
            labels = []
            for obj in root.findall("object"):
                label = obj.find("name").text
                try:
                    label = self._find_label(label)
                except AttributeError:
                    continue
                bbox = obj.find("bndbox")
                x_min = int(bbox.find("xmin").text)
                y_min = int(bbox.find("ymin").text)
                x_max = int(bbox.find("xmax").text)
                y_max = int(bbox.find("ymax").text)

                boxes.append([x_min, y_min, x_max, y_max])
                labels.append(self.labels.index(label))

            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)


            target = {"boxes": boxes, "labels": labels}


            if self.transforms:
                image = self.transforms(image)

            return image, target

        def __len__(self):
            return len(self.imgs)

    #ODM public functions   
    def __init__(self, labels : list[Label], filename="object_detection.pth"):
        self.labels = [None] + labels
        self.filename = filename

    def _get_model(self, training : bool): 
        weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.COCO_V1 if training else None
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, len(self.labels))
        if training == False:
            model.load_state_dict(torch.load("object_detection_model.pth", map_location=ODM.device))
        return model

    def train(self, images_folder : str, annotations_folder : str, epochs=10, save_to=None, lr : float = 0.0001, batch_size : int = 4):
        """Train the model using the .xml and images from the annotations and images folder"""
        if save_to == None:
            self.save_to = self.filename
        try:
            transform = T.Compose([T.ToTensor()])
            data : ODM._Data = ODM._Data(self.labels, "", images_folder, annotations_folder, transforms=transform)

            model = self._get_model(training=True)
            model.to(ODM.device)
            optimizer = optim.Adam(model.parameters(), lr=lr)
            train_size = int(0.8 * len(data))
            test_size = len(data) - train_size
            train_dataset, test_dataset = torch.utils.data.random_split(data, [train_size, test_size])

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))
            for epoch in range(epochs):
                model.train()
                epoch_loss = 0
                for images, targets in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}"):
                    images = list(img.to(ODM.device) for img in images)
                    targets = [{k: v.to(ODM.device) for k, v in target.items()} for target in targets]
                    try:
                        loss_dict = model(images, targets)
                    except AssertionError as e:
                        print(e)
                        continue
                    loss = sum(loss for loss in loss_dict.values())
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()

                print(f"Epoch {epoch+1}, Loss: {loss.item()}")
                torch.save(model.state_dict(), save_to)
        except FileNotFoundError as e:
            print(RED + str(e) + END_COLOR)
    
    def predict(self, image_path) -> dict[str, torch.Tensor]:
        """Use the model to detect objects in an image"""
        model = self._get_model(training=False)
        model.to(ODM.device)
        model.eval()
        transform = T.Compose([T.ToTensor()])
        
        try:
            image = Image.open(image_path).convert("RGB")

            img_tensor = transform(image).unsqueeze(0).to(ODM.device)

            with torch.no_grad():
                predictions = model(img_tensor)[0]

            boxes = predictions["boxes"].cpu().numpy()
            labels = predictions["labels"].cpu().numpy()

            scores = predictions["scores"].cpu().numpy()
            valid_indices = scores >= 0.2
            boxes, labels = boxes[valid_indices], labels[valid_indices]
            figure, axes = plt.subplots(1, figsize=(8, 8))
            figure.patch.set_facecolor('black')
            axes.imshow(image)
            plt.axis("off")
            for box, label in zip(boxes, labels):
                label : Label = self.labels[label]
                x_min, y_min, x_max, y_max = box
                rect = patches.Rectangle(
                    (x_min, y_min),
                    x_max - x_min,
                    y_max - y_min,
                    linewidth=2,
                    edgecolor=label.line_color,
                    facecolor="none"
                )
                axes.add_patch(rect)
                axes.text(x_min, y_min - 5, f"{label.name}", color=label.font_color, fontsize=12, backgroundcolor=label.line_color)
            plt.show()
            return predictions
        except FileNotFoundError as e:
            print(RED + str(e) + END_COLOR)