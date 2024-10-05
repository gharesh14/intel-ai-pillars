# Importing the required packages
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
import threading
import time

# Global variable to control the capturing process
capturing = True

# Function for voice input
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

# Memory recall function
def memory_recall():
    global capturing  # Use the global variable to control the loop

    # Initialize the camera
    cap = cv2.VideoCapture(1)

    # Check if the camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open the external camera")
        return

    img_counter = 0
    context = []  # Initialize context as a list to hold multiple reports
    print("Press 'q' to stop capturing images.")

    while capturing:  # Continue capturing while the flag is True
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

            model = genai.GenerativeModel("gemini-1.5-flash")
            pic = PIL.Image.open(img_name)

            # Generate prompt based on previous context
            if not context:
                context_str = "not available"
            else:
                context_str = " | ".join(context)

            prompt = f"""You are an image describing assistant. Generate document-level content based on the image. 
            If there is previous context available, fine-tune your answer based on it.
            Context: {context_str}"""

            # Generate report
            memory = model.generate_content([pic, prompt])
            context.append(memory.text)  # Store the generated report

            # Optionally, you can save this report to a file
            with open("memory_report.txt", "a") as report_file:
                report_file.write(memory.text + "\n")

        else:
            print("Error: Could not read from the camera.")
            break

    # Release the camera and close all windows
    cap.release()
    cv2.destroyAllWindows()

    print("Image capture stopped.")

# Global variable to hold the generated reports
generated_reports = []

# Creating function for the scene description
def scene_desc(image_path):
    model = genai.GenerativeModel("gemini-1.5-flash")
    pic = PIL.Image.open(image_path)
    pic2 = PIL.Image.open("G:/INTEL HACKATHON/images/npcroom.jpg")
    prompt = "What am I seeing? Please explain as much as possible, making sure to understand the context of the image. Respond as if you are describing the situation to a friend who has his eyes closed shut. Explain two images one after another."
    response = model.generate_content([pic, pic2, prompt])
    print(response.text)
    return response.text

# Creating function for the sensory search
def sense_res(image_path):
    model = genai.GenerativeModel("gemini-1.5-flash")
    pic = PIL.Image.open(image_path)
    prompt = "What is this? Respond as if you are describing the situation to a friend who has his eyes closed shut."
    response = model.generate_content([pic, prompt])
    print(response.text)
    return response.text

# Navigation function
def nav(image_path, text):
    model = genai.GenerativeModel("gemini-1.5-flash")
    ov_model = YOLO('yolov8n_openvino_model')
    dlmodel = YOLO('best.pt')
    
    # Predict with the model
    results = dlmodel.predict(image_path, stream=True, conf=0.5)
    for i, r in enumerate(results):
        r.save(filename=f"resultsseg.jpg")

    results2 = ov_model.predict(image_path, stream=True, conf=0.5)
    for i, r in enumerate(results2):
        r.save(filename=f"resultsov.jpg")

    pic = PIL.Image.open(image_path)
    pic2 = PIL.Image.open('./images/depth_image.png')
    pic3 = PIL.Image.open("resultsov.jpg")
    pic4 = PIL.Image.open('resultsseg.jpg')

    prompt = f"""You are a navigation guide capable of indoor navigation according to the user's desire: {text}. You've been provided with 4 images. Each contains a parameter for your information.
    Image1 is the actual unaltered image for human vision. Image2: It contains depth estimation with varying pixel density where intensity decreases with increased distance.
    Image3: It contains any obstacles that might hinder the user from navigating through, so guide accordingly. Image4: Contains paths and walls segmented and provide me with a route if a path is available.
    INSTRUCTIONS:
    Give me step-by-step directions and the number of steps needed to reach the destination."""
    
    response = model.generate_content([pic, pic2, pic3, pic4, prompt])
    return response.text

def speak_text(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def voice_input():
    global capturing  # Access the global variable
    user_input = get_voice_input()

    if user_input:
        if "exit" in user_input.lower() or "quit" in user_input.lower() or "stop" in user_input.lower():
            global capturing
            capturing = False  # Set the capturing flag to False
            speak_text("Exiting the program.")
            print("Exiting the program.")
        elif "describe" in user_input.lower():
            image_path = "/images/npcroom.jpg"
            if image_path:
                search_result = scene_desc(image_path)
                speak_text(search_result)
                print(search_result)
        elif "search" in user_input.lower():
            image_path = "/images/npcroom.jpg"
            if image_path:
                search_result = sense_res(image_path)
                speak_text(search_result)
                print(search_result)
        elif "route" in user_input.lower():
            image_path = "images/npcroom.jpg"
            if image_path:
                search_result = nav(image_path, user_input)
                speak_text(search_result)
                print(search_result)
        elif "memory recall" in user_input.lower():
            # Retrieve generated memory content
            if generated_reports:
                # Join all reports into a single string
                all_reports = "\n".join(generated_reports)
                speak_text(all_reports)
                print(all_reports)
            else:
                speak_text("No memory reports available yet.")
                print("No memory reports available yet.")
        else:
            print("Could not recognize the command, please try again.")
            voice_input()
    else:
        voice_input()

# Assigning the API key
genai.configure(api_key="AIzaSyBg0uCFF_7Fs52m29gHBnRXhe3mw_MFrns")

# Start the memory recall in a separate thread
memory_thread = threading.Thread(target=memory_recall)
memory_thread.start()

# Start listening for voice commands
voice_input()
