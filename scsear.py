# Importing the required packages
import google.generativeai as genai
import os
import PIL.Image
import speech_recognition as sr
import cv2
import pyttsx3

# Assigning the API key
genai.configure(api_key="AIzaSyBg0uCFF_7Fs52m29gHBnRXhe3mw_MFrns")

# Function for voice input
def get_voice_input():  
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        speak_text("Listening for user input")  
        print("Listening for user input...")
        recognizer.adjust_for_ambient_noise(source, duration=0.5)  
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio)
            print(f"User said: {text}")
            return text
        except sr.UnknownValueError:
            print("Sorry, I did not understand that.")
            return None
        except sr.RequestError as e:
            print(f"Could not request results from the speech recognition service; {e}")
            return None

#image input
def cam_input():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        speak_text("Error: Could not open the external camera.")
        print("Error: Could not open the external camera.")
        return False  
    try:
        
        ret, frame = cap.read()

        if ret:
           
            cv2.imwrite('captured_image.jpg', frame)
            print("The image is saved as: captured_image.jpg")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            return True  
        else:
            speak_text("Error: Could not read from the camera.")
            print("Error: Could not read from the camera.")
            return False  
    finally:
        cap.release()  

# scene descryption
def scene_desc():
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        pic = PIL.Image.open('captured_image.jpg')
        prompt = "You are a describing assistant. You are going to generate document-level content based on the image."
        response = model.generate_content([pic, prompt])
        print(response.text)
        return response.text
    except FileNotFoundError:
        speak_text("Error: The image 'captured_image.jpg' was not found.")
        print("Error: The image 'captured_image.jpg' was not found.")
        return None
    except Exception as e:
        speak_text(f"An error occurred while generating scene description: {e}")
        print(f"An error occurred while generating scene description: {e}")
        return None

# sensory search
def sense_res():
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        pic = PIL.Image.open('captured_image.jpg')
        prompt = "Tell me about the object I am holding. What is this?"
        response = model.generate_content([pic, prompt])
        print(response.text)
        return response.text
    except FileNotFoundError:
        speak_text("Error: The image 'captured_image.jpg' was not found.")
        print("Error: The image 'captured_image.jpg' was not found.")
        return None
    except Exception as e:
        speak_text(f"An error occurred while generating sensory search: {e}")
        print(f"An error occurred while generating sensory search: {e}")
        return None

# text to speech
def speak_text(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

# 
def voice_input():
    while True:  
        user_input = get_voice_input()

        if user_input:
            user_input = user_input.lower()  

          
            if "describe" in user_input or "seeing" in user_input or "scene" in user_input:
                print("Capturing image for scene description...")
                speak_text("Capturing image for scene description...")
                if cam_input():  
                    search_result = scene_desc()
                    if search_result:
                        speak_text(search_result)
                    break  
            elif "search" in user_input or "holding" in user_input or "buy" in user_input:
                print("Capturing image for sensory search...")
                speak_text("Capturing image for sensory search...")
                if cam_input():  # Capture image before processing
                    search_result = sense_res()
                    if search_result:
                        speak_text(search_result)
                    break  
            else:
                speak_text("Could not recognize the voice command, please try again!")
                print("Could not recognize the voice command, please try again!")
        else:
            speak_text("No input received, please try again!")
            print("No input received, please try again!")



if __name__ == "__main__":
    voice_input()
