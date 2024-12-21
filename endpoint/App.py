import sys
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QPushButton, QLabel, QLineEdit, 
                               QComboBox, QFileDialog, QRadioButton, QGroupBox, 
                               QFormLayout, QMessageBox)
from PySide6.QtGui import QPixmap
from PySide6.QtCore import Qt
from PIL import ImageQt
import torch
from PIL import Image
import numpy as np
from funcs import load_data, build_model, train, inference, generate_saliency

class ModelTrainingApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Model Training and Inference")
        self.setGeometry(100, 100, 800, 600)
        self.init_ui()

    def init_ui(self):
        main_layout = QHBoxLayout()

        # Left Pane
        left_pane = QWidget()
        left_layout = QVBoxLayout()
        left_pane.setLayout(left_layout)

        self.model_combobox = QComboBox()
        self.model_combobox.addItems(["resnet", "mobilenet"])
        left_layout.addWidget(self.model_combobox)
        self.model_size_combobox = QComboBox()
        left_layout.addWidget(self.model_size_combobox)

        # Set initial items based on the default model
        self.update_model_size_options()
        self.model_combobox.currentIndexChanged.connect(self.update_model_size_options)        

        self.epochs_input = QLineEdit()
        self.epochs_input.setPlaceholderText("Enter number of epochs")
        left_layout.addWidget(self.epochs_input)

        self.dataset_input = QLineEdit()
        self.dataset_input.setPlaceholderText("Enter dataset path")
        left_layout.addWidget(self.dataset_input)
        self.browse_button = QPushButton("Browse", self)
        left_layout.addWidget(self.browse_button)
        self.browse_button.clicked.connect(self.show_dialog)

        self.load_model_groupbox = QGroupBox("Model Loading Option")
        load_model_layout = QFormLayout()
        self.load_pretrained_radio = QRadioButton("Load Pretrained Model")
        self.load_trained_radio = QRadioButton("Load Model")
        self.load_trained_radio.setChecked(True)  
        load_model_layout.addRow(self.load_pretrained_radio)
        load_model_layout.addRow(self.load_trained_radio)
        self.load_model_groupbox.setLayout(load_model_layout)
        left_layout.addWidget(self.load_model_groupbox)

        self.train_button = QPushButton("Start Training")
        self.train_button.clicked.connect(self.start_training)
        left_layout.addWidget(self.train_button)

        # Right Pane
        right_pane = QWidget()
        right_layout = QVBoxLayout()
        right_pane.setLayout(right_layout)

        # Button for loading an image and running inference
        self.inference_button = QPushButton("Load Image and Process")
        self.inference_button.clicked.connect(self.load_image_and_process)
        right_layout.addWidget(self.inference_button)

        # Image display
        self.image_label = QLabel("No image loaded")
        self.image_label.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(self.image_label)

        # Predicted label display
        self.predicted_label = QLabel("Predicted label: None")
        self.predicted_label.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(self.predicted_label)

        # Saliency map display
        self.saliency_map_label = QLabel("No saliency map generated")
        self.saliency_map_label.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(self.saliency_map_label)


        # Combine Panes
        main_layout.addWidget(left_pane)
        main_layout.addWidget(right_pane)

        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

    def update_model_size_options(self):
            selected_model = self.model_combobox.currentText()
            
            if selected_model == "resnet":
                self.model_size_combobox.clear()  # Clear existing items
                self.model_size_combobox.addItems(["small", "medium", "large", "xlarge", "xxlarge"])
            elif selected_model == "mobilenet":
                self.model_size_combobox.clear()  # Clear existing items
                self.model_size_combobox.addItems(["small", "large"])
                
    def show_dialog(self):
        file_path = QFileDialog.getExistingDirectory(self, 'Select Folder')
        if file_path:
            self.dataset_input.setText(file_path)

    def start_training(self):
        try:
            model_name = self.model_combobox.currentText()
            model_size = self.model_size_combobox.currentText()
            epochs = int(self.epochs_input.text())
            data_path = self.dataset_input.text()

            data = load_data(data_path)
            if "error" in data:
                print(data["error"])
                return

            train_loader = data['train_loader']
            val_loader = data.get('val_loader')

            model = build_model(model_name, model_size, num_classes=1000)
            trained_model = train(model, train_loader, val_loader, epochs=epochs)

            model_path, _ = QFileDialog.getSaveFileName(self, "Save Trained Model", "", "Model Files (*.pth)")
            if model_path:
                torch.save(trained_model.state_dict(), model_path)
                print(f"Model saved to {model_path}")

        except Exception as e:
            print(f"Error during training: {str(e)}")

    def load_image_and_process(self):
        """
        Allows the user to load an image, run inference, and optionally generate a saliency map.
        """
        # Step 1: Load the image
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.bmp)")
        if not file_path:
            return

        # Step 2: Load the model
        model_name = self.model_combobox.currentText()
        model_size = self.model_size_combobox.currentText()

        if self.load_pretrained_radio.isChecked():
            model = build_model(model_name, model_size, num_classes=1000)
        else:
            model_path, _ = QFileDialog.getOpenFileName(self, "Open Trained Model", "", "Model Files (*.pth)")
            if not model_path:
                print("No model file selected")
                return
            model = build_model(model_name, model_size, num_classes=1000)
            model.load_state_dict(torch.load(model_path))

        # Step 3: Run inference
        image, pred = inference(file_path, model)
        pixmap = QPixmap.fromImage(ImageQt.ImageQt(image))
        self.image_label.setPixmap(pixmap.scaled(400, 400, Qt.KeepAspectRatio))
        self.predicted_label.setText(f"Predicted label: {pred}")

        # Step 4: Ask user if they want to generate a saliency map
        saliency_choice = QMessageBox.question(
            self, "Generate Saliency Map", "Would you like to generate a saliency map?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )

        if saliency_choice == QMessageBox.Yes:
            # Generate saliency map
            saliency_map = generate_saliency(file_path, model)
            if saliency_map.dtype != np.uint8:
                saliency_map = (255 * (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())).astype(np.uint8)
            saliency_image = Image.fromarray(saliency_map)
            pixmap = QPixmap.fromImage(ImageQt.ImageQt(saliency_image))
            self.saliency_map_label.setPixmap(pixmap.scaled(400, 400, Qt.KeepAspectRatio))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ModelTrainingApp()
    window.show()
    sys.exit(app.exec())
