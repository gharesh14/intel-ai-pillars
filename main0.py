# importing the required packages
import google.generativeai as genai
import os
import PIL.Image
import speech_recognition as sr
import cv2
import pyttsx3
from transformers import pipeline
from PIL import Image
import requests
from ultralytics import YOLO
from transformers import DPTImageProcessor, DPTForDepthEstimation
import torch
import numpy as np

image_processor = DPTImageProcessor.from_pretrained("Intel/dpt-hybrid-midas")
DE_model = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas")
#function for voice input
def get_voice_input():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening for user input...")
        recognizer.adjust_for_ambient_noise(source, duration=0.5)  # Adjust for ambient noise
        try:
            audio = recognizer.listen(source)
            text = recognizer.recognize_google(audio)
            print(f"User said: {text}")
            return text
        except sr.UnknownValueError:
            print("Could not understand audio. Please try again.")
            return get_voice_input()  # Retry listening
        except sr.RequestError as e:
            print(f"Could not request results from Google Speech Recognition service; {e}")
            return None

#real time input in the form of image (open cv)
def memory_recall(context):

        import time

        # Initialize the camera
        cap = cv2.VideoCapture(0)

        # Check if the camera opened successfully
        if not cap.isOpened():
            print("Error: Could not open the external camera")
        else:
            img_counter = 0
            print("Press 'q' to stop capturing images.")

            while True:
                # Capture each frame
                ret, frame = cap.read()

                if ret:
                    # Save the image
                    img_name = f"captured_image_{img_counter}.jpg"
                    cv2.imwrite(img_name, frame)
                    print(f"Image saved: {img_name}")
                    img_counter += 1

                    # Wait for 1 minute before capturing the next image
                    time.sleep(60)
                    model=genai.GenerativeModel("gemini-1.5-flash")
                    
                    pic = PIL.Image.open(frame)
                    if not context:
                        context="not available"
                        prompt=f"""You are an image describing assistant . You are going to generate document level content based on the image, also if there Is previous context available, fine tune your answer based on it
                        context:{context}"""
                        memory=model.generate_content([pic,prompt])
                        context.append(memory)
                    else:
                        prompt=f"""You are an image describing assistant . You are going to generate document level content based on the image, also if there Is previous context available, fine tune your answer based on it
                        context:{context}"""
                        memory=model.generate_content([pic,prompt])
                        context.append(memory)


                    # Check if 'q' key is pressed to stop the loop
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    print("Error: Could not read from the camera.")
                    break

    # Release the camera and close all windows
            cap.release()
            cv2.destroyAllWindows()

            print("Image capture stopped.")
        return context



    


# assigning the api key
genai.configure(api_key="AIzaSyBg0uCFF_7Fs52m29gHBnRXhe3mw_MFrns")

#creating function for the scene descryption
def scene_desc(image_path):
    model = genai.GenerativeModel("gemini-1.5-flash")
    pic = PIL.Image.open(image_path)
    pic2=PIL.Image.open("G:/INTEL HACKATHON/images/npcroom.jpg")
    prompt= "what am I seeing, please explain as much as possible, make sure to understand the context of the image,respond as if you are describing the situation to a friend who has his eyes closed shut,explian two images one after another"
    response = model.generate_content([pic,pic2,prompt])
    print(response.text)
    return response.text

#creating function for the sensory search
def sense_res(image_path):
    model = genai.GenerativeModel("gemini-1.5-flash")
    pic = PIL.Image.open(image_path)
    prompt= "what is this, respond as if you are describing the situation to a friend who has his eyes closed shut"
    response = model.generate_content([pic,prompt])
    print(response.text)
    return response.text

def nav(image_path,text):

    model = genai.GenerativeModel("gemini-1.5-pro")
    ov_model=YOLO('yolov8n_openvino_model')
    dlmodel=YOLO('best.pt')
    image=Image.open(image_path)
    inputs = image_processor(images=image, return_tensors="pt")
    # Perform inference
    with torch.no_grad():
        outputs = DE_model(**inputs)
        predicted_depth = outputs.predicted_depth

    # Interpolate to the original image size
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=image.size[::-1],
        mode="bicubic",
        align_corners=False,
)
    # Convert the prediction to a NumPy array and scale it
    output = prediction.squeeze().cpu().numpy()
    formatted = (output * 255 / np.max(output)).astype("uint8")
    depth = Image.fromarray(formatted)
    depth.save('depth.jpg') 

    # Predict with the model
    results=dlmodel.predict(image_path,stream=True,conf=0.5)
    for i, r in enumerate(results):
        # Plot results image
        im_bgr = r.plot()  # BGR-order numpy array
        im_rgb = Image.fromarray(im_bgr[..., ::-1])  # RGB-order PIL image
        # Save results to disk
        r.save(filename=f"resultsseg.jpg")

    results2= ov_model.predict(image_path, stream=True,conf=0.5)
    for i, r in enumerate(results2):
        # Plot results image
        im_bgr = r.plot()  # BGR-order numpy array
        im_rgb = Image.fromarray(im_bgr[..., ::-1])  # RGB-order PIL image
        # Save results to disk
        r.save(filename=f"resultsov.jpg")
    pic = PIL.Image.open(image_path)
    pic2=PIL.Image.open('depth.jpg')
    pic3=PIL.Image.open("resultsov.jpg")
    pic4=PIL.Image.open('resultsseg.jpg')

    prompt= f"""You are a visual navigation guide capable of giving step by step route as my desire .My desired destination: {text}
    INSTRUCTIONS: You are provided with 4 images for the same environment the first pic is the actual environment in which you are supposed to show route for
    the second image is the depth estimation from the user, the increase in darkness represents decrease in depth, the third image consists of object detection to clearly identify the obstacles, and the fourth image does segmentation to show you the path which the user can follow if there is no path the user cannot travel to the desired location
    reply with only text and NEVER use ''
    GUIDANCE:
    1.describe the user's destination and their path
    2.provide details for navigating
    3. Identify obstacles in the image and in how many would I reach the said obstacle which might interfere with my path
    4.clearly specify the estimates amount of steps the user should take in order to reach the destination
    """
    
    response = model.generate_content([pic,pic2,pic3,pic4,prompt])
    print(response)
    return response.text




def speak_text(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def voice_input():
  
    user_input = get_voice_input()
    text=user_input

    if user_input:
        if "describe" in user_input.lower():
            image_path ="/images/npcroom.jpg"
            if image_path:
                search_result = scene_desc(image_path)
                speak_text(search_result)
                print(search_result)
        elif "search" in user_input.lower():
            image_path ="/images/npcroom.jpg"

            if image_path:
                search_result = sense_res(image_path)
                speak_text(search_result)
                print(search_result)

        elif "route" in user_input.lower():
            image_path ="images/ethnoverandah.jpg"
            if image_path:
                search_result = nav(image_path,text)
                speak_text(search_result)
                print(search_result)

        else:
            print("Could not recognize the command, please try again.")
            voice_input()
    else:
        voice_input()


voice_input()
