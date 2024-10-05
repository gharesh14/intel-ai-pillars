import face_recognition
import cv2
import pickle
import os
import speech_recognition as sr
import pyttsx3

# Function to capture image using webcam (OpenCV)
def capture_image():
    cap = cv2.VideoCapture(1)  # 0 for the default camera
    if not cap.isOpened():
        print("Error: Could not open the camera.")
        return None
    
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        return None
    
    cv2.imwrite("captured_face.jpg", frame)
    cap.release()
    return frame

# Voice input function using speech recognition
def get_voice_input():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening for user input...")
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio)
            print(f"User said: {text}")
            return text.lower()
        except sr.UnknownValueError:
            print("Sorry, I didn't understand that.")
            return None
        except sr.RequestError as e:
            print(f"Request error from speech recognition service; {e}")
            return None

# Text-to-speech conversion
def speak_text(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

# Encode face using face_recognition library
def encode_face(frame):
    face_locations = face_recognition.face_locations(frame)
    if face_locations:
        face_encodings = face_recognition.face_encodings(frame, face_locations)
        return face_encodings[0]  # Return the first face encoding
    else:
        print("No face detected in the image.")
        return None

# Save the face encoding with the person's name
def save_face_encoding(name, encoding):
    with open(f"{name}_face_encoding.pkl", 'wb') as f:
        pickle.dump(encoding, f)
    speak_text(f"Face of {name} saved successfully.")

# Load saved face encodings from the system
def load_face_encodings():
    known_face_encodings = []
    known_face_names = []

    for file in os.listdir("."):
        if file.endswith("_face_encoding.pkl"):
            name = file.split('_')[0]
            with open(file, 'rb') as f:
                encoding = pickle.load(f)
                known_face_encodings.append(encoding)
                known_face_names.append(name)

    return known_face_encodings, known_face_names

# Recognize faces using known encodings
def recognize_face(known_encodings, known_names, frame):
    unknown_encoding = encode_face(frame)

    if unknown_encoding is not None:
        results = face_recognition.compare_faces(known_encodings, unknown_encoding)
        if True in results:
            matched_idx = results.index(True)
            name = known_names[matched_idx]
            speak_text(f"Hello {name}, nice to see you again!")
        else:
            speak_text("I don't recognize this person. Would you like to save them?")
            user_input = get_voice_input()
            if user_input and "yes" in user_input:
                name = input("Enter the name of the person: ")
                save_face_encoding(name, unknown_encoding)
    else:
        speak_text("No face detected for recognition.")

# Main function to handle the voice input and control flow
def voice_input():
    while True:
        user_input = get_voice_input()

        if user_input:
            if "exit" in user_input or "quit" in user_input:
                speak_text("Exiting the program.")
                break
            elif "describe" in user_input or "scene" in user_input:
                # Integrate your existing scene description logic here
                speak_text("Scene description feature activated.")
                # (Add your existing scene description function call here)
            elif "search" in user_input or "holding" in user_input:
                # Integrate your existing sensory search logic here
                speak_text("Sensory search feature activated.")
                # (Add your existing sensory search function call here)
            elif "face" in user_input or "recognize" in user_input:
                # Capture the frame and recognize the face
                frame = capture_image()
                if frame is not None:
                    known_encodings, known_names = load_face_encodings()
                    recognize_face(known_encodings, known_names, frame)
        else:
            print("No valid input received. Please try again.")

# Main execution
if __name__ == "__main__":
    voice_input()
