import torch
import torchvision.transforms as T
import pandas as pd
from deep_utils import face_detector_loader
from .resnet50 import FaceRecog


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

size = 224
stats = (0.5,), (0.5,)

model_path = '../models/Face_Recognition_checkpoint_resnet50.pth'

model = FaceRecog(105).to(device)
model.eval()
model.load_state_dict(torch.load(model_path, map_location=device))

face_detector = face_detector_loader('MTCNNTorchFaceDetector')

df = pd.read_csv('../data/labels.csv')
classes = list(df.name)

transformer = T.Compose([
    T.Resize(size),
    T.CenterCrop(size),
    T.ToTensor(),
    T.Normalize(*stats)])

setup = {
    'model': model,
    'face_detector': face_detector,
    'classes': classes,
    'transformer': transformer,
    'stats': stats,
    'device': device,

}