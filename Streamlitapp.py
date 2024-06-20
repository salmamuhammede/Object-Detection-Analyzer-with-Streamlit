import streamlit as st
import torch
import pandas as pd
from torchvision import models, transforms
from PIL import Image,ImageDraw,ImageFont
####################general
st.write("Welcome to our analyser **_app_**")
preprocess = transforms.Compose([
    transforms.ToTensor()
])

model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()
catogries=models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT.meta["categories"]
generalList=[]
image=''
font = ImageFont.truetype("arial.ttf",size=20)
###################################################################################fileuploding
file=st.file_uploader(label="Upload the photo you want",type=["jpeg","jpg","png"]) 
##################################################################################
def process():
 if file is not None:
    if "image" in file.type:
      handlephoto()
      createList()       
 return 
###################################################################################

def handlephoto():
  image = Image.open(file).convert("RGB")
  #width, height = image.size
  #image = image.resize((width // 2, height // 2))
  st.image(image, caption='Uploaded Image')   
  input_tensor = preprocess(image).unsqueeze(0)
  with torch.no_grad():
        detections = model(input_tensor)[0]
  #st.write(detections) 
  draw = ImageDraw.Draw(image)
  for box, label, score in zip(detections['boxes'], detections['labels'], detections['scores']):
     if label < len(catogries) and score > 0.5: 
        object_label = catogries[label]
        #st.write(object_label)
        #if object_label not in generalList:
        box = box.numpy()
        draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline='blue', width=2)
        draw.text((box[0], box[1]), f"{catogries[label]}: {score:.2f}", fill='black',font=font)
        generalList.append(object_label)
  st.image(image, caption='Image with Detections')         
  return
######################################################################################
def createList():
   df = pd.DataFrame(generalList, columns=["List Items"])
   st.table(df)  
   
######################################################################################  
st.button("**_Analyse Image_**",on_click=process)