# Dynamic Affective State Analysis - Homework @ Applied Informatics 4.0
Project made by:
- Ciprian Constantinescu @ 325CA
- Stefan-George Ghinescu @ 321CA
- Mihnea-Andrei Cazan @ 325CA

Project repository: [here](https://github.com/CiprianC1/IAp4)

## Project Description
This project aims to analyze the affective state of a person based on the facial expressions. The modality of implementation is a web application that uses the built-in camera of the user's device to capture their facial expressions in real-time. The application will then analyze the facial expressions either on the server or on device and determine the affective state of the user. The affective state will be displayed on the screen in real-time as a coloured mask over the user's face. If server-side processing is used, a label with the specific emotion (not just one of the primary emotions) will be displayed on the screen, above the user's face. The application supports at most 10 people in the same frame.

There is also a feature that allows the user to take a screenshot of their face with the mask on and save it to the server's database. Moreover, the users can view all screenshots taken in a dedicated gallery, where they can also delete the screenshots. The screenshots are stored in a ***mongoDB database***, alongside a thumbnail of the image to be loaded faster in the gallery.

## Implementation
The project is implemented using Python and the Flask framework for the server-side processing. The client-side processing is done using JavaScript. The backend has endpoints for the client to send the frames to the server for processing and for requesting one of the models used by the server.

The frontend is iplemented using plain HTML, CSS and JavaScript. The client-side processing is done using the `getUserMedia` API for accessing the user's camera and the `Canvas` API for drawing the mask over the user's face. The client sends the frames to the server for processing using the `fetch` API.

Face masks are infered by the ***MediaPipe Face Mesh*** model, which is a lightweight model that detects facial landmarks in real-time. The landmarks are then used to draw the mask over the user's face and to crop the face from the frame to send it to the server for emotion infering.

The whole infrastructure uses docker containers - one for the server and another for the ***mongoDB database*** used to store screenshots of the users who took one.

The facial expressions are analysed by a model trained on an [IEEE dataset](https://ieee-dataport.org/documents/135-class-emotional-facial-expression-dataset) comprising of upwards of 500k images, with 135 different classes of specific emotions and converted into a format compatible with the ***ONNX runtime*** for protability. The server uses one model that produces specific results, but due to the limited dataset and the high number of classes, is inaccurate (the highest accuracy is 21%, which is still significantly higher than pure chance - 0.74%).

To stabilise the colour prediction of the mask, another model is used, which is trained on the same dataset, only that similar emotions are grouped into the superclass of a primary emotion (happiness, anger, sadness and so on). This grouping is done semantically and predefined (i.e. hardcoded based on psychological considerations). The accuracy of this model is around 55%, which is significantly higher than the previous model, but still not ideal.

## Encountered Problems
Due to the large dataset, that had to be downloaded from several sources, the training was cumbersome and the available computational resources limited us in our experiments with the models.

The canvas API proved to not be ideal for this async application and, at first, we enountered problems with either the mask not being drawn or the mask flickering between frames. We solved this by using a buffered canvas.

The dataset needed to be downloaded from several sources, due to IEEE not having licenses for the images. If we tried to download the dataset single-threaded, it would have taken upwards of 6 hours. We solved this by using a multi-threaded downloader, that also cropps the images to the face, using the landmarks provided by the ***MediaPipe Face Mesh*** model. Since the images are stock photos, there are logos covering some of the faces, which weren't removed by the cropper. This is a limitation of the dataset and could have affected the training of the models.

## Contributions of Each Team Members:
Ciprian Constantinescu (the master of AI):
- Managed the dataset and the training of the models
- Implemented the emotion inference
- Managed onnx runtimes on the client and on the server
- Integrating media pipe face mesh model for face detection

Stefan-George Ghinescu (the lord of infrastructure):
- Managed the server script
- Dockerised the server and the database
- Integrated the database with the server
- Implemented the screenshot feature on the server

Mihnea-Andrei Cazan (the king of frontend):
- Implemented the frontend
- Integrated the data received from the server with the frontend
- Managed (somehow) to display the masks over the users' face in javascript
- Implemented the screenshot feature on frontend and the gallery

## Note on Running the Application
The when building the docker image, it is expected that the onnx models are already present in their respective places (the root of the server and the static folder). Because of the assignment uploading constraints, we were unable to upload the models, so they either need to be computed (with the appropriate computational resources) or downloaded from github. To build the models, download the dataset from [here](https://ieee-dataport.org/documents/135-class-emotional-facial-expression-dataset), run the _build_dataset_multithreading.py_ script to download the dataset and preprocess it. Then, run the _train-resnet101.ipynb_ for the innacurate but specific model (used for labels) and _train-coalescing-augumentation.ipynb_ for the more accurate but general model (used for colours). After the models are generated, use the _model_save.py_ script to convert them to onnx format. To use on-device processing, put a copy of _emotion_model.onnx_ in the static folder.