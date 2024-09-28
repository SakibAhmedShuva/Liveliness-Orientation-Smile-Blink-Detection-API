# Liveliness Test: Facial Orientation, Smile, and Blink Detection API

This API is a comprehensive solution for real-time face analysis, leveraging advanced computer vision techniques to enhance security in facial recognition systems.

## Features

- **Face Orientation Detection**: Determines if the face is oriented left, right, or front.
- **Smile Detection**: Identifies whether the subject is smiling.
- **Blink Detection**: Detects eye blinks using the Eye Aspect Ratio (EAR) method.
- **Liveliness Assessment**: Combines multiple factors to evaluate the liveliness of the subject.
- **Session Management**: Maintains user sessions for continuous verification.
- **Image Storage**: Saves frontal images for further processing or verification.

## API Endpoint

The main endpoint for the liveliness test is:

```
POST /liveliness_test
```

### Request

- Method: POST
- Content-Type: multipart/form-data
- Parameters:
  - `image`: The image file to be processed
  - `session_id` (optional): A unique identifier for the session

### Response

The API returns a JSON object with the following fields:

```json
{
  "session_id": "unique-session-identifier",
  "orientation": "Front|Left|Right",
  "smile": "Yes|No",
  "blink": "Yes|No",
  "frontal_image_url": "/sessions/session-id/frontal_image.jpg"
}
```

## Setup and Installation

1. Clone the repository:
   ```
   git clone https://github.com/SakibAhmedShuva/liveliness-orientation-smile-blink-detection-api.git
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the Flask application:
   ```
   python app.py
   ```

The server will start on `http://localhost:5000`

## HTML Template

The project includes a basic HTML template (`liveliness_template.html`) for testing the API with realtime video from camera. This template provides a simple interface to interact with the liveliness detection features. To use the template open `liveliness_template.html` in a web browser.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
