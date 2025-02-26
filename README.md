# Object Detection Module

## Description
Easly train, save and use an object detection model based on .xml annotations.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/ItayLador2013/Object-Detction-Module.git
   ```
2. Navigate to the project directory:
   ```bash
   cd Object-Detction-Module
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
## Example made using module for blood cell detection & classification
![classified blood cell]([https://github.com/ItayLador2013/Object-Detction-Module/blob/main/example-usage.png?raw=true)

## Usage
You need to have two data folders: annotations and images. 
The annotation .xml files should contain the following:

```xml
<annotation>
	<filenamea>CORROSPONDING_IMAGE_FILE_NAME</filenamea>
	<object>
		<name>LABEL</name>
		<bndbox>
			<xmin>X_MIN</xmin>
			<ymin>Y_MIN</ymin>
			<xmax>X_MAX</xmax>
			<ymax>Y_MAX</ymax>
		</bndbox>
	</object>
    <object>
		<name>LABEL</name>
		<bndbox>
			<xmin>X_MIN</xmin>
			<ymin>Y_MIN</ymin>
			<xmax>X_MAX</xmax>
			<ymax>Y_MAX</ymax>
		</bndbox>
	</object>
</annotation>

```

### Importing the Class
```python
from odm import ODM, Label
```

```python
class ODM():
    def __init__(self, labels : list[Label], filename : str = "object_detection.pth")

class Label():
     def __init__(self, name : str, font_color : str ="white", line_color : str ="red")

labels = [Label(name=LABEL1_NAME), Label(name=LABEL2_NAME), Label(name=LABEL3_NAME)]
odm_obj = ODM(labels=labels)
```

### Training
```python
def train(self, images_folder : str, annotations_folder : str, epochs=10, save_to=None, lr : float = 0.0001, batch_size : int = 4)

odm_obj.train(images_folder="images", annotations_folder="annotations")
```

## Predict
```python
def predict(self, image_path) -> dict[str, torch.Tensor]:

predictions = odm_obj.predict(image_path=PATH_TO_IMAGE)

where predictions = {
    "boxes": torch.tensor([[x1, y1, x2, y2], ...]),
    "labels": torch.tensor([class1, class2, ...]),
    "scores": torch.tensor([score1, score2, ...])
}
```


## License
[MIT](LICENSE)

