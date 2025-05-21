import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog
from tkinter import messagebox
from PIL import Image, ImageTk
import os
import time
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

class FacialExpressionRecognition:
    def __init__(self, root):
        self.root = root
        self.root.title("Facial Expression Recognition System")
        self.root.geometry("1200x700")
        self.root.configure(bg="#f0f0f0")
        
        # Set app icon
        try:
            self.root.iconbitmap("icon.ico")
        except:
            pass
        
        # Load the pre-trained models
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Load the emotion recognition model
        try:
            # Path to the model file - update this to your model's location
            model_path = "fer_model.h5"
            if os.path.exists(model_path):
                self.emotion_model = load_model(model_path)
                self.model_loaded = True
                print("Model loaded successfully from", model_path)
            else:
                print(f"Warning: Model file {model_path} not found.")
                self.model_loaded = False
                # We'll download a pre-trained model if needed
                self.download_model()
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model_loaded = False
        
        # Expression labels
        self.expressions = ["Angry", "Disgusted", "Fearful", "Happy", "Sad", "Surprised", "Neutral"]
        
        # Variables
        self.cap = None
        self.is_webcam_active = False
        self.current_frame = None
        self.detection_active = False
        self.current_image_path = None
        
        # Create GUI elements
        self.create_widgets()
        
    def download_model(self):
        """
        Downloads a pre-trained facial expression recognition model if needed.
        For this example, we'll simulate this by informing the user.
        """
        response = messagebox.askyesno(
            "Model Download", 
            "The facial expression model is not found. Would you like to download it now?"
        )
        
        if response:
            try:
                # Here you would implement the actual download logic
                # For now, we'll just show a message
                messagebox.showinfo(
                    "Download", 
                    "This is a demo. In a real app, the model would download here.\n\n"
                    "Please place a trained 'fer_model.h5' file in the application directory."
                )
            except Exception as e:
                messagebox.showerror("Error", f"Failed to download model: {e}")
        
    def create_widgets(self):
        # Create main frames
        top_frame = tk.Frame(self.root, bg="#f0f0f0")
        top_frame.pack(fill=tk.X, pady=10)
        
        main_frame = tk.Frame(self.root, bg="#f0f0f0")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Create title label
        title_label = tk.Label(
            top_frame, 
            text="Facial Expression Recognition", 
            font=("Arial", 24, "bold"),
            bg="#f0f0f0",
            fg="#2c3e50"
        )
        title_label.pack(pady=10)
        
        # Left frame (video/image display)
        self.left_frame = tk.Frame(main_frame, bg="#ffffff", width=800, height=500)
        self.left_frame.pack(side=tk.LEFT, padx=10, fill=tk.BOTH, expand=True)
        self.left_frame.pack_propagate(False)
        
        # Video display
        self.video_label = tk.Label(self.left_frame, bg="black")
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Right frame (controls)
        right_frame = tk.Frame(main_frame, bg="#ffffff", width=300)
        right_frame.pack(side=tk.RIGHT, padx=10, fill=tk.BOTH)
        
        # Status display
        status_frame = tk.LabelFrame(right_frame, text="Status", bg="#ffffff", font=("Arial", 12))
        status_frame.pack(fill=tk.X, padx=10, pady=10, ipady=5)
        
        self.status_label = tk.Label(
            status_frame, 
            text="Idle", 
            font=("Arial", 12),
            bg="#ffffff",
            fg="#2c3e50"
        )
        self.status_label.pack(pady=5)
        
        # Model status
        if hasattr(self, 'model_loaded'):
            model_status = "Model loaded" if self.model_loaded else "Model not loaded"
            model_color = "#28a745" if self.model_loaded else "#dc3545"
        else:
            model_status = "Model status unknown"
            model_color = "#ffc107"
            
        self.model_status_label = tk.Label(
            status_frame, 
            text=model_status, 
            font=("Arial", 10),
            bg="#ffffff",
            fg=model_color
        )
        self.model_status_label.pack(pady=2)
        
        # Controls
        controls_frame = tk.LabelFrame(right_frame, text="Controls", bg="#ffffff", font=("Arial", 12))
        controls_frame.pack(fill=tk.X, padx=10, pady=10, ipady=5)
        
        # Source selection
        source_frame = tk.Frame(controls_frame, bg="#ffffff")
        source_frame.pack(fill=tk.X, pady=5)
        
        self.source_var = tk.StringVar(value="webcam")
        
        webcam_radio = tk.Radiobutton(
            source_frame, 
            text="Webcam", 
            variable=self.source_var, 
            value="webcam",
            bg="#ffffff",
            font=("Arial", 10),
            command=self.update_source
        )
        webcam_radio.pack(side=tk.LEFT, padx=10)
        
        image_radio = tk.Radiobutton(
            source_frame, 
            text="Image", 
            variable=self.source_var, 
            value="image",
            bg="#ffffff",
            font=("Arial", 10),
            command=self.update_source
        )
        image_radio.pack(side=tk.LEFT, padx=10)
        
        # Buttons
        button_frame = tk.Frame(controls_frame, bg="#ffffff")
        button_frame.pack(fill=tk.X, pady=10)
        
        self.start_button = ttk.Button(
            button_frame,
            text="Start Detection",
            command=self.toggle_detection,
            style="TButton"
        )
        self.start_button.pack(fill=tk.X, padx=10, pady=5)
        
        self.browse_button = ttk.Button(
            button_frame,
            text="Browse Image",
            command=self.browse_image,
            state=tk.DISABLED,
            style="TButton"
        )
        self.browse_button.pack(fill=tk.X, padx=10, pady=5)
        
        # Load model button (if model is not loaded)
        if not hasattr(self, 'model_loaded') or not self.model_loaded:
            self.load_model_button = ttk.Button(
                button_frame,
                text="Load Model",
                command=self.load_model,
                style="TButton"
            )
            self.load_model_button.pack(fill=tk.X, padx=10, pady=5)
        
        # Results display
        results_frame = tk.LabelFrame(right_frame, text="Expression Results", bg="#ffffff", font=("Arial", 12))
        results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.result_text = tk.Text(results_frame, bg="#ffffff", height=10, width=30, font=("Arial", 10))
        self.result_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.result_text.config(state=tk.DISABLED)
        
        # Style the buttons
        style = ttk.Style()
        style.configure("TButton", font=("Arial", 12), padding=5)
        
        # Statistics Frame
        stats_frame = tk.LabelFrame(right_frame, text="Statistics", bg="#ffffff", font=("Arial", 12))
        stats_frame.pack(fill=tk.X, padx=10, pady=10, ipady=5)
        
        self.faces_detected_label = tk.Label(
            stats_frame, 
            text="Faces detected: 0", 
            font=("Arial", 10),
            bg="#ffffff",
            fg="#2c3e50",
            anchor=tk.W
        )
        self.faces_detected_label.pack(fill=tk.X, padx=5, pady=2)
        
        self.processing_speed_label = tk.Label(
            stats_frame, 
            text="Processing: 0 ms", 
            font=("Arial", 10),
            bg="#ffffff",
            fg="#2c3e50",
            anchor=tk.W
        )
        self.processing_speed_label.pack(fill=tk.X, padx=5, pady=2)
        
        # Create exit button
        exit_button = ttk.Button(
            right_frame,
            text="Exit",
            command=self.exit_app,
            style="TButton"
        )
        exit_button.pack(fill=tk.X, padx=10, pady=10)
    
    def load_model(self):
        """Load the facial expression recognition model from a file"""
        file_path = filedialog.askopenfilename(
            title="Select Model File",
            filetypes=[("Model files", "*.h5"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                self.emotion_model = load_model(file_path)
                self.model_loaded = True
                self.model_status_label.config(text="Model loaded", fg="#28a745")
                
                # Remove the load model button if it exists
                if hasattr(self, 'load_model_button'):
                    self.load_model_button.destroy()
                    
                messagebox.showinfo("Success", "Model loaded successfully!")
                
                # Print model summary for debugging
                self.emotion_model.summary()
                print(f"Input shape: {self.emotion_model.input_shape}")
                print(f"Output shape: {self.emotion_model.output_shape}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load model: {e}")
                self.model_status_label.config(text="Model load failed", fg="#dc3545")
        
    def update_source(self):
        if self.source_var.get() == "webcam":
            self.browse_button.configure(state=tk.DISABLED)
            if self.is_webcam_active:
                self.stop_webcam()
        else:  # image
            self.browse_button.configure(state=tk.NORMAL)
            if self.is_webcam_active:
                self.stop_webcam()
                
    def browse_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        
        if file_path:
            self.current_image_path = file_path
            self.show_image(file_path)
            self.status_label.config(text="Image loaded")
            
    def show_image(self, image_path):
        # Read the image
        img = cv2.imread(image_path)
        if img is None:
            messagebox.showerror("Error", "Could not open the image file")
            return
            
        # Convert to RGB for display
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.current_frame = img_rgb.copy()
        
        # Resize for display
        img_resized = self.resize_image_for_display(img_rgb)
        
        # Convert to PhotoImage
        img_pil = Image.fromarray(img_resized)
        img_tk = ImageTk.PhotoImage(image=img_pil)
        
        # Update the video label
        self.video_label.configure(image=img_tk)
        self.video_label.image = img_tk  # Keep a reference
        
    def resize_image_for_display(self, image):
        # Get the dimensions of the display area
        width = self.left_frame.winfo_width() - 20  # Adjust for padding
        height = self.left_frame.winfo_height() - 20  # Adjust for padding
        
        if width <= 1 or height <= 1:  # Widget not fully initialized
            width = 780
            height = 480
        
        # Get image dimensions
        img_height, img_width = image.shape[:2]
        
        # Calculate the scaling factor
        scale_width = width / img_width
        scale_height = height / img_height
        scale = min(scale_width, scale_height)
        
        # Calculate new dimensions
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        
        # Resize the image
        resized_image = cv2.resize(image, (new_width, new_height))
        
        return resized_image
        
    def start_webcam(self):
        if self.cap is None or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Could not open webcam")
                return False
        
        self.is_webcam_active = True
        self.status_label.config(text="Webcam active")
        self.update_webcam_feed()
        return True
        
    def update_webcam_feed(self):
        if not self.is_webcam_active:
            return
            
        ret, frame = self.cap.read()
        if ret:
            # Convert to RGB for display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.current_frame = frame_rgb.copy()
            
            # Process the frame if detection is active
            if self.detection_active:
                self.process_frame(frame_rgb)
            else:
                # Just display the frame
                frame_resized = self.resize_image_for_display(frame_rgb)
                img_pil = Image.fromarray(frame_resized)
                img_tk = ImageTk.PhotoImage(image=img_pil)
                self.video_label.configure(image=img_tk)
                self.video_label.image = img_tk  # Keep a reference
            
        # Update again after 10ms
        self.root.after(10, self.update_webcam_feed)
        
    def stop_webcam(self):
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
        self.is_webcam_active = False
        self.status_label.config(text="Webcam stopped")
        
    def toggle_detection(self):
        # Check if model is loaded
        if not hasattr(self, 'model_loaded') or not self.model_loaded:
            messagebox.showinfo("Info", "Please load a facial expression model first")
            return
            
        if not self.detection_active:
            # Start detection
            if self.source_var.get() == "webcam":
                if not self.is_webcam_active:
                    if not self.start_webcam():
                        return
            elif self.source_var.get() == "image":
                if self.current_image_path is None:
                    messagebox.showinfo("Info", "Please select an image first")
                    return
                    
            self.detection_active = True
            self.start_button.config(text="Stop Detection")
            self.status_label.config(text="Detection active")
            
            # If it's an image, process it immediately
            if self.source_var.get() == "image" and self.current_frame is not None:
                self.process_frame(self.current_frame)
                
        else:
            # Stop detection
            self.detection_active = False
            self.start_button.config(text="Start Detection")
            self.status_label.config(text="Detection stopped")
            
            # Clear results
            self.result_text.config(state=tk.NORMAL)
            self.result_text.delete(1.0, tk.END)
            self.result_text.config(state=tk.DISABLED)
            
            # Reset stats
            self.faces_detected_label.config(text="Faces detected: 0")
            self.processing_speed_label.config(text="Processing: 0 ms")
            
    def preprocess_face_for_emotion(self, face_roi):
        """Preprocess the face image for emotion recognition"""
        try:
            # Make a copy to avoid modifying the original
            face = face_roi.copy()
            
            # Check if input is already grayscale or needs conversion
            if len(face.shape) > 2 and face.shape[2] > 1:
                face = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
            
            # Resize to 48x48 for the model
            face = cv2.resize(face, (48, 48))
            
            # Apply histogram equalization to improve contrast
            face = cv2.equalizeHist(face)
            
            # Convert to float and normalize to [0,1]
            face = face.astype("float") / 255.0
            
            # Debug info
            print(f"Preprocessed face shape: {face.shape}, min: {face.min()}, max: {face.max()}")
            
            # Expand dimensions to fit model input requirements
            # Add batch dimension
            face = np.expand_dims(face, axis=0)
            
            # For grayscale input, add channel dimension if model expects it
            if self.model_loaded:
                expected_dims = len(self.emotion_model.input_shape)
                if expected_dims > 3:  # If model expects channel dimension
                    face = np.expand_dims(face, axis=-1)
                print(f"Input shape after expansion: {face.shape}")
            
            return face
        except Exception as e:
            print(f"Error preprocessing face: {e}")
            return None
            
    def predict_emotion(self, face_roi):
        """Predict emotion from face ROI"""
        try:
            if not hasattr(self, 'model_loaded') or not self.model_loaded:
                print("Model not loaded, returning random prediction")
                # If model is not loaded, return neutral prediction rather than random
                return "Neutral", 0.5
                
            # Preprocess the face
            processed_face = self.preprocess_face_for_emotion(face_roi)
            if processed_face is None:
                print("Face preprocessing failed")
                return "Unknown", 0.0
                
            # Get prediction from model
            print("Making prediction...")
            emotion_preds = self.emotion_model.predict(processed_face)[0]
            print(f"Raw predictions: {emotion_preds}")
            
            # Get the index of the highest prediction
            emotion_idx = np.argmax(emotion_preds)
            emotion_label = self.expressions[emotion_idx]
            confidence = emotion_preds[emotion_idx]
            
            print(f"Predicted {emotion_label} with confidence {confidence:.2f}")
            return emotion_label, float(confidence)
        except Exception as e:
            print(f"Error predicting emotion: {e}")
            import traceback
            traceback.print_exc()
            return "Error", 0.0
        
    def process_frame(self, frame):
        start_time = time.time()
        
        # Create a copy for drawing
        frame_copy = frame.copy()
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        # Detect faces - try multiple parameters if faces aren't detected
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        # If no faces detected, try with more lenient parameters
        if len(faces) == 0:
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.05,
                minNeighbors=3,
                minSize=(20, 20)
            )
            print(f"Retried face detection, found {len(faces)} faces")
        
        # Update faces detected count
        self.faces_detected_label.config(text=f"Faces detected: {len(faces)}")
        
        # Process each face
        results = []
        
        for (x, y, w, h) in faces:
            # Draw rectangle around the face
            cv2.rectangle(frame_copy, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Extract the face region - use the original frame for better color info
            face_roi = frame[y:y+h, x:x+w]
            
            # Get emotion prediction
            emotion_label, confidence = self.predict_emotion(face_roi)
            
            # Select color based on emotion
            if emotion_label == "Happy":
                color = (0, 255, 0)  # Green
            elif emotion_label in ["Angry", "Disgusted"]:
                color = (255, 0, 0)  # Red
            elif emotion_label in ["Sad", "Fearful"]:
                color = (0, 0, 255)  # Blue
            elif emotion_label == "Surprised":
                color = (255, 255, 0)  # Yellow
            else:
                color = (255, 255, 255)  # White
            
            # Add text above the face
            label = f"{emotion_label} ({confidence:.2f})"
            cv2.putText(
                frame_copy, 
                label, 
                (x, y-10), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                color, 
                2
            )
            
            # Add to results
            results.append({
                "emotion": emotion_label,
                "confidence": confidence,
                "position": (x, y, w, h)
            })
        
        # Calculate processing time
        end_time = time.time()
        processing_time = (end_time - start_time) * 1000  # Convert to ms
        self.processing_speed_label.config(text=f"Processing: {processing_time:.2f} ms")
        
        # Display the processed frame
        frame_resized = self.resize_image_for_display(frame_copy)
        img_pil = Image.fromarray(frame_resized)
        img_tk = ImageTk.PhotoImage(image=img_pil)
        self.video_label.configure(image=img_tk)
        self.video_label.image = img_tk  # Keep a reference
        
        # Update results text
        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete(1.0, tk.END)
        
        if not results:
            self.result_text.insert(tk.END, "No faces detected.")
        else:
            for i, result in enumerate(results):
                self.result_text.insert(
                    tk.END, 
                    f"Face {i+1}:\n"
                    f"Expression: {result['emotion']}\n"
                    f"Confidence: {result['confidence']:.2f}\n\n"
                )
                
        self.result_text.config(state=tk.DISABLED)
        
    def exit_app(self):
        if messagebox.askokcancel("Exit", "Are you sure you want to exit?"):
            if self.is_webcam_active:
                self.stop_webcam()
            self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = FacialExpressionRecognition(root)
    root.protocol("WM_DELETE_WINDOW", app.exit_app)
    root.mainloop()