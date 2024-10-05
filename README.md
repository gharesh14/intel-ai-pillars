GENEYE

GENEYE is an assistive tool for visually challenged individuals. It consists of a camera, speaker, and microphone embedded in glasses, enabling users to navigate their surroundings, identify objects, recall events, and interact with people through advanced AI technologies.

Table of Contents

1.Features

2.Installation

3.Usage

4.Technologies Used

5.Data Security

6.Contributing


Features

Path Detection: Navigate closed spaces without relying on GPS by leveraging depth estimation, segmentation, and object detection. Users are guided to their desired destination via audio commands.

Scene Description: Captures an image and generates a textual description of the surroundings, then reads it out to the user through the speaker when requested.

Memory Recall: Automatically captures an image every 10 seconds and stores both the image and a scene description for the last 30 days. This helps users recall past events while ensuring data security with local storage.

Person Identification & Gaze Detection: Recognizes people in front of the camera and provides their names through the speaker. Gaze detection enables the identification of pre-identified individuals within the user’s line of sight.

Semantic Segmentation: Identifies and labels objects in the user's environment, providing contextual awareness.

Contact Retrieval: Stores and retrieves a person's contact information in cloud storage. The user can retrieve contact details by simply stating the person’s name.


Installation

1.To set up GENEYE on your local machine:

Clone this repository:

git clone https://github.com/username/geneye.git

2.Install the required dependencies:

pip install -r requirements.txt

3.Run the project:

python main.py


Usage

1.Start Path Detection:

python path_detection.py
Follow the audio commands to navigate through closed spaces.

2.Capture Scene Description:

python scene_description.py
Ask the system to describe your current surroundings.

3.Memory Recall:

python memory_recall.py
Retrieve events from the last 30 days by providing a date and time.

4.Person Identification:

python person_identification.py
The system will provide names of recognized individuals in front of the user.

Technologies Used

Deep Learning Models: For depth estimation, semantic segmentation, and object detection.
APIs: For text generation, voice commands, and scene description.
Local & Cloud Storage: Ensures both security and convenience in storing user data.

Data Security
Memory recall and personal information are securely stored locally to protect user privacy.
Only necessary contact details are stored in the cloud, ensuring minimal exposure.

Contributing

We welcome contributions to improve GENEYE. To contribute:

1.Fork the repository.

2.Create a feature branch.

3.Make necessary changes.

4.Submit a pull request.



