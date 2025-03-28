# Face Authentication System

## Overview
The Face Authentication System is a web application built using Streamlit that allows users to register and log in using facial recognition technology. The application captures users' facial images, processes them to generate embeddings, and stores them in a SQLite database for authentication.

## Features
- User registration with facial recognition
- User login with facial authentication
- Secure storage of user data in a SQLite database
- Intuitive user interface built with Streamlit

## Installation
1. Clone the repository:
   ```bash
   git clone [https://github.com/arslaan5/Face-Authentication-System]
   cd Face-Recognition-for-Login-Authentication-System
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up the environment variables:
   - Create a `.env` file in the root directory and specify the database path:
     ```
     DB_PATH=path/to/your/database.db
     ```

## Usage
1. Run the application:
   ```bash
   streamlit run app.py
   ```

2. Navigate to the application in your web browser.

3. Register a new user by capturing a clear image of your face and entering your name.

4. Log in using the captured image to authenticate.

## Architecture
- **Frontend**: Built with Streamlit, providing a user-friendly interface for registration and login.
- **Backend**: 
  - **Database**: SQLite for storing user data and face embeddings.
  - **Face Recognition**: Utilizes the `face_recognition` library for detecting and recognizing faces.
  - **Utilities**: Functions for image processing, database interactions, and user management.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

## License
This project is licensed under the MIT License.
