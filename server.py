from flask import Flask, request, jsonify
from PIL import Image
import torch
import torchvision.transforms as transforms
import io
import torch.nn as nn

# Initialize the Flask application
app = Flask(__name__)

# Load the model
class VGG16(nn.Module):
  def __init__(self,num_classes=10):
    super(VGG16,self).__init__()
    self.layer1=nn.Sequential(
        nn.Conv2d(3,64,kernel_size=3,stride=1,padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU()
    )
    self.layer2=nn.Sequential(
        nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2,stride=2)
    )
    self.layer3=nn.Sequential(
        nn.Conv2d(64,128,kernel_size=3,stride=1,padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU()
    )
    self.layer4=nn.Sequential(
        nn.Conv2d(128,128,kernel_size=3,stride=1,padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2,stride=2)
    )
    self.layer5=nn.Sequential(
        nn.Conv2d(128,256,kernel_size=3,stride=1,padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(),
    )
    self.layer6=nn.Sequential(
        nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU()
    )
    self.layer7=nn.Sequential(
        nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2,stride=2)
    )
    self.layer8=nn.Sequential(
        nn.Conv2d(256,512,kernel_size=3,stride=1,padding=1),
        nn.BatchNorm2d(512),
        nn.ReLU()
    )
    self.layer9=nn.Sequential(
        nn.Conv2d(512,512,kernel_size=3,stride=1,padding=1),
        nn.BatchNorm2d(512),
        nn.ReLU()
    )
    self.layer10=nn.Sequential(
        nn.Conv2d(512,512,kernel_size=3,stride=1,padding=1),
        nn.BatchNorm2d(512),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2,stride=2)
    )
    self.layer11=nn.Sequential(
        nn.Conv2d(512,512,kernel_size=3,stride=1,padding=1),
        nn.BatchNorm2d(512),
        nn.ReLU()
    )
    self.layer12=nn.Sequential(
        nn.Conv2d(512,512,kernel_size=3,stride=1,padding=1),
        nn.BatchNorm2d(512),
        nn.ReLU()
    )
    self.layer13=nn.Sequential(
        nn.Conv2d(512,512,kernel_size=3,stride=1,padding=1),
        nn.BatchNorm2d(512),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2,stride=2)
    )
    self.fc=nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(7*7*512,4096),
        nn.ReLU()
    )
    self.fc1=nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(4096,4096),
        nn.ReLU()
    )
    self.fc2=nn.Sequential(
        nn.Linear(4096,num_classes)
    )
  def forward(self,x):
      out=self.layer1(x)
      out=self.layer2(out)
      out=self.layer3(out)
      out=self.layer4(out)
      out=self.layer5(out)
      out=self.layer6(out)
      out=self.layer7(out)
      out=self.layer8(out)
      out=self.layer9(out)
      out=self.layer10(out)
      out=self.layer11(out)
      out=self.layer12(out)
      out=self.layer13(out)
      out=out.reshape(out.size(0),-1)
      out=self.fc(out)
      out=self.fc1(out)
      out=self.fc2(out)
      return out

# Initialize the model and load state_dict
model = VGG16(32)
model.load_state_dict(torch.load('model.pth'))
model.eval()  # Set the model to evaluation mode

# Define the image preprocessing steps
def transform_image(image_bytes):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize the image if necessary
        transforms.ToTensor(),          # Convert to tensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize based on ImageNet standards
    ])
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    return transform(image).unsqueeze(0)  # Add batch dimension

# Define the route for image classification
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        try:
            # Preprocess the image and make a prediction
            image_bytes = file.read()
            tensor = transform_image(image_bytes)
            with torch.no_grad():
                outputs = model(tensor)
                # Assuming the model output is logits, apply softmax to get probabilities
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                _, predicted_class = torch.max(probabilities, 1)
                predicted_class = predicted_class.item()

            # Return the result as JSON
            return jsonify({'class_id': predicted_class, 'probabilities': probabilities.tolist()}), 200
        except Exception as e:
            return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
