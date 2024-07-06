import time
from tinderapi import Tinder, TinderProfile, TinderDB, Telegram
import random
import json
import codecs
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from torchvision import models
import os

# Function to load and preprocess the image for prediction
def load_image(image_path):
    image = Image.open(image_path)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = preprocess(image).unsqueeze(0)
    return image

# Function to predict the class label
def predict_image(image_path, model, class_names):
    # Load and preprocess the image
    image = load_image(image_path)
    
    # Set the model to evaluation mode
    model.eval()
    
    # Perform the prediction
    with torch.no_grad():
        outputs = model(image)
        score = float(outputs[0][1])
        print("output = ",outputs)
        _, predicted = torch.max(outputs, 1)
        
        # Convert tensor to numpy array and extract the predicted class index
        predicted_class = predicted.item()
        print('predicted = ',predicted)
        # Map index to class name
        predicted_label = class_names[predicted_class]
        
        return predicted_label,score

# Load the trained ResNet-50 model
model_path = 'resnet50_epoch_8.pth'
checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

# Create a new instance of the ResNet model
model = models.resnet50(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 2)

# Load state_dict into the model
model.load_state_dict(checkpoint)

def print_variables(data, indent=0):
    spacing = ' ' * indent
    
    if isinstance(data, dict):
        for key, value in data.items():
            print(f"{spacing}{key}:")
            print_variables(value, indent + 4)
    elif isinstance(data, list):
        for index, value in enumerate(data):
            print(f"{spacing}[{index}]:")
            print_variables(value, indent + 4)
    elif isinstance(data, tuple):
        for index, value in enumerate(data):
            print(f"{spacing}({index}):")
            print_variables(value, indent + 4)
    elif isinstance(data, set):
        for value in data:
            print(f"{spacing}{value}")
    else:
        print(f"{spacing}{data}")
        
#ACCOUNT = json.loads(open('account.json', 'r').read())

AUTH_TOKEN = '' #Input your auth token. 
DB = TinderDB()
threshold = 1
thresholdOfProfile = 3
def liker():
    while True:
        
        TinderAcc = Tinder(AUTH_TOKEN)
        #telegram = Telegram(telegram_key=ACCOUNT['telegram_bot_access_token'], chat_id=ACCOUNT['group_id'])
        matches = TinderAcc.get_potential_matches(verbose=False)
        if matches == -1:
            break
        elif matches == 2:
            print('Timeout waiting 5 min')
            time.sleep(60 * 5)
        else:
            for potential_match in matches:
                profile = TinderProfile(potential_match, AUTH_TOKEN, save_pics=True)
                print_variables(profile)
                print("Type = ",type(profile))
                print("token = " , profile.x_auth_token)
                print("ID = ",profile.id)
                print("Gender = ",profile.gender)
                print("Name = ",profile.name)
                print("Match = ",profile.match)
                print("Show gender on profile = ",profile.show_gender_on_profile)
                print(profile.getAll())

                #Modify your like and dislike condition.
                photoNumber = 0
                scoreOfThis = 0

                path = 'Photos/' + profile.id + '/'
                #time.sleep(random.randint(1,2))
                for i in os.listdir(path):
                    image_path = path + i
                    class_names = ['not_woman', 'woman']  # Replace with your actual class names
                    predicted_label,score = predict_image(image_path, model, class_names)
                    print('File Name = ', i)
                    print(f'Predicted label: {predicted_label}')
                    if predicted_label == 'woman' and score >= threshold:
                        scoreOfThis = scoreOfThis + score
                        photoNumber = photoNumber + 1
                scoreOfProfile = 0
                if(photoNumber > 1):
                    scoreOfProfile = float(scoreOfThis/photoNumber)
                
                print("The score of this profile = ", scoreOfProfile)
                
                status = ""
                
                if(scoreOfProfile >= thresholdOfProfile):
                    print("I like it!")
                    status = "I like it!"
                    profile.like() # for dislike profile.dislike()
                else:
                    print("I dislike it!")
                    status = "I dislike it!"
                    profile.dislike() # for dislike profile.dislike()
                
                #telegram.sendPhoto(f'{profile.name} - {profile.birth_date.split("-")[0]} - {profile.distance_km} KM',
                                   #f'Photos/{profile.id}')
                DB.insert_into_table(profile)
                f = codecs.open("allProfiles.txt", "a", encoding="utf-8")
                pro = str(profile.getAll())
                
                
                f.write(pro + '\n' + status + '\n\n')
                
                f.close()

                time.sleep(random.randint(1, 4))

        print('Searching New Matches...')
        #telegram.sendMessage('Searching for new matches...')


if __name__ == '__main__':
    liker()
