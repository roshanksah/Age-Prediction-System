import cv2
import numpy as np
from flask import Flask, render_template, Response

class AgeGenderDetector:
    def __init__(self, models_path='models/'):
        """
        Initialize face, age, and gender detection models
        
        Args:
            models_path (str): Path to the directory containing model files
        """
        # Face Detection Model
        self.face_proto = f"{models_path}opencv_face_detector.pbtxt"
        self.face_model = f"{models_path}opencv_face_detector_uint8.pb"
        
        # Age Detection Model
        self.age_proto = f"{models_path}age_deploy.prototxt"
        self.age_model = f"{models_path}age_net.caffemodel"
        
        # Gender Detection Model
        self.gender_proto = f"{models_path}gender_deploy.prototxt"
        self.gender_model = f"{models_path}gender_net.caffemodel"
        
        # Model configuration
        self.MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
        
        # Age categories (moved to self to make it a class attribute)
        self.AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(33-37)','(38-43)', '(48-53)', '(60-75)', '(80-100)']
        self.GENDER_LIST = ['Male', 'Female']
        
        # Padding for face detection
        self.PADDING = 20
        
        # Load models
        self.face_net = cv2.dnn.readNet(self.face_model, self.face_proto)
        self.age_net = cv2.dnn.readNet(self.age_model, self.age_proto)
        self.gender_net = cv2.dnn.readNet(self.gender_model, self.gender_proto)
    
    def detect_faces(self, frame, conf_threshold=0.7):
        """
        Detect faces in the frame
        
        Args:
            frame (numpy.ndarray): Input image
            conf_threshold (float): Confidence threshold for face detection
        
        Returns:
            tuple: Processed frame and list of face bounding boxes
        """
        frame_opencv_dnn = frame.copy()
        frame_height = frame_opencv_dnn.shape[0]
        frame_width = frame_opencv_dnn.shape[1]
        
        # Create blob from frame
        blob = cv2.dnn.blobFromImage(
            frame_opencv_dnn, 1.0, (300, 300), 
            [104, 117, 123], True, False
        )
        
        # Detect faces
        self.face_net.setInput(blob)
        detections = self.face_net.forward()
        
        face_boxes = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > conf_threshold:
                # Calculate face coordinates
                x1 = int(detections[0, 0, i, 3] * frame_width)
                y1 = int(detections[0, 0, i, 4] * frame_height)
                x2 = int(detections[0, 0, i, 5] * frame_width)
                y2 = int(detections[0, 0, i, 6] * frame_height)
                
                face_boxes.append([x1, y1, x2, y2])
                
                # Draw rectangle around face
                cv2.rectangle(
                    frame_opencv_dnn, (x1, y1), (x2, y2), 
                    (0, 255, 0), int(round(frame_height/150)), 8
                )
        
        return frame_opencv_dnn, face_boxes
    
    def predict_age_gender(self, frame, face_box):
        """
        Predict age and gender for a given face
        
        Args:
            frame (numpy.ndarray): Original frame
            face_box (list): Bounding box coordinates of the face
        
        Returns:
            tuple: Predicted gender and age
        """
        # Extract face with padding
        face = frame[
            max(0, face_box[1] - self.PADDING):
            min(face_box[3] + self.PADDING, frame.shape[0] - 1),
            max(0, face_box[0] - self.PADDING):
            min(face_box[2] + self.PADDING, frame.shape[1] - 1)
        ]
        
        # Prepare blob for prediction
        blob = cv2.dnn.blobFromImage(
            face, 1.0, (227, 227), 
            self.MODEL_MEAN_VALUES, swapRB=False
        )
        
        # Predict gender
        self.gender_net.setInput(blob)
        gender_preds = self.gender_net.forward()
        gender = self.GENDER_LIST[gender_preds[0].argmax()]
        
        # Predict age
        self.age_net.setInput(blob)
        age_preds = self.age_net.forward()
        age = self.AGE_LIST[age_preds[0].argmax()]
        
        return gender, age

class AgeGenderDetectionApp:
    def __init__(self):
        """
        Initialize Flask app and age-gender detector
        """
        self.app = Flask(__name__)
        self.detector = AgeGenderDetector()
        
        # Configure routes
        self.app.route('/')(self.index)
        self.app.route('/video_feed')(self.video_feed)
    
    def index(self):
        """
        Render the main page
        """
        return render_template('index.html')
    
    def generate_frames(self):
        """
        Generate video frames with age and gender detection
        """
        # Open video capture
        video = cv2.VideoCapture(0)
        
        while True:
            # Read frame
            success, frame = video.read()
            if not success:
                break
            
            # Detect faces
            result_img, face_boxes = self.detector.detect_faces(frame)
            
            # Process detected faces
            for face_box in face_boxes:
                try:
                    # Predict age and gender
                    gender, age = self.detector.predict_age_gender(frame, face_box)
                    
                    # Annotate frame
                    cv2.putText(
                        result_img, 
                        f'{gender}, {age}', 
                        (face_box[0], face_box[1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.8, (0, 255, 255), 2, 
                        cv2.LINE_AA
                    )
                except Exception as e:
                    print(f"Error processing face: {e}")
            
            # Encode frame
            ret, buffer = cv2.imencode('.jpg', result_img)
            frame = buffer.tobytes()
            
            # Yield frame for streaming
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
        # Release video capture
        video.release()
    
    def video_feed(self):
        """
        Video streaming route
        """
        return Response(
            self.generate_frames(), 
            mimetype='multipart/x-mixed-replace; boundary=frame'
        )
    
    def run(self, debug=True, host='0.0.0.0', port=5000):
        """
        Run the Flask application
        
        Args:
            debug (bool): Enable debug mode
            host (str): Host IP
            port (int): Port number
        """
        self.app.run(debug=debug, host=host, port=port)

def main():
    """
    Main application entry point
    """
    app = AgeGenderDetectionApp()
    app.run()

if __name__ == '__main__':
    main()
