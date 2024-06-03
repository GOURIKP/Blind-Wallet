
# Blind Wallet

Blind Wallet is an application designed to assist visually impaired individuals in identifying currency notes using image classification technology. The application utilizes machine learning algorithms to recognize different denominations of currency notes commonly used in India.

## Features
- **Currency Recognition:** Blind Wallet uses image classification to identify the denomination of currency notes.
- **User-Friendly Interface:** The application provides a simple and intuitive interface for easy interaction.
- **Voice Output:** Blind Wallet announces the detected currency denomination using speech synthesis, making it accessible to visually impaired users.
- **Web-Based Application:** Access Blind Wallet through a web browser, making it convenient to use on various devices.

## How it Works
1. **Upload Image:** Users can upload an image of a currency note through the application interface.
2. **Image Processing:** The uploaded image is processed using computer vision techniques to extract relevant features.
3. **Classification:** The pre-trained machine learning model analyzes the features extracted from the image to classify the currency denomination.
4. **Voice Output:** The result of the classification is announced using speech synthesis, providing auditory feedback to the user.
5. **Display Result:** The application displays the detected currency denomination on the interface for confirmation.

## Technologies Used
- **Python:** Backend development and machine learning model implementation.
- **Flask:** Web framework for building the web-based application.
- **HTML/CSS:** Frontend design and layout.
- **JavaScript:** Client-side scripting for dynamic interactions.
- **OpenCV:** Image processing and feature extraction.
- **Scikit-learn:** Machine learning library for model training and prediction.
- **Pyttsx3:** Text-to-speech library for voice output.

## Deployment
Blind Wallet can be deployed on a web server to make it accessible to users. It can also be hosted on platforms like Heroku for easy deployment and scaling.

## Contributing
Blind Wallet is an open-source project, and contributions are welcome. If you're interested in contributing, please check out the project repository on GitHub and follow the guidelines for contributing code, reporting issues, or suggesting features.

## License
Blind Wallet is licensed under the MIT License, which means it is free to use, modify, and distribute, subject to the terms and conditions of the license.
