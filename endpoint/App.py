import sys
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QPushButton, QLabel, QLineEdit, 
                               QComboBox, QFileDialog, QRadioButton, QGroupBox, 
                               QFormLayout,
                               )
from PySide6.QtGui import QPixmap
from PySide6.QtCore import Qt
from PIL import ImageQt
from matplotlib import pyplot as plt
import torch
from funcs import load_data, build_model, train, inference, generate_saliency


class ModelTrainingApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Model Training and Inference")
        self.setGeometry(100, 100, 800, 600)

        
        self.init_ui()

    def init_ui(self):
        
        main_layout = QHBoxLayout()

        
        left_pane = QWidget()
        left_layout = QVBoxLayout()
        left_pane.setLayout(left_layout)

        
        self.model_combobox = QComboBox()
        self.model_combobox.addItems(["resnet", "mobilenet"])
        left_layout.addWidget(self.model_combobox)

        
        self.model_size_combobox = QComboBox()
        self.model_size_combobox.addItems(["small", "medium", "large", "xlarge", "xxlarge"])
        left_layout.addWidget(self.model_size_combobox)

        
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
        self.load_trained_radio = QRadioButton("Load Trained Model")
        self.load_trained_radio.setChecked(True)  
        load_model_layout.addRow(self.load_pretrained_radio)
        load_model_layout.addRow(self.load_trained_radio)
        self.load_model_groupbox.setLayout(load_model_layout)
        left_layout.addWidget(self.load_model_groupbox)

        
        self.train_button = QPushButton("Start Training")
        self.train_button.clicked.connect(self.start_training)
        left_layout.addWidget(self.train_button)

        
        right_pane = QWidget()
        right_layout = QVBoxLayout()
        right_pane.setLayout(right_layout)

        
        self.inference_button = QPushButton("Run Inference")
        self.inference_button.clicked.connect(self.run_inference)
        right_layout.addWidget(self.inference_button)

        
        self.saliency_button = QPushButton("Generate Saliency")
        self.saliency_button.clicked.connect(self.generate_saliency_map)
        right_layout.addWidget(self.saliency_button)

        
        self.image_label = QLabel("No image loaded")
        right_layout.addWidget(self.image_label)

        
        main_layout.addWidget(left_pane)
        main_layout.addWidget(right_pane)

        
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

    def show_dialog(self):
        file_path = QFileDialog.getExistingDirectory(self, 'Select Folder')
        # trim path to root folder
        # file_path = str(Path(file_path).parents[1])   # can raise error if path does not have 2 parents
         
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

    def run_inference(self):
        
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.bmp)")
        if not file_path:
            return

        
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

        
        image,pred = inference(file_path, model)
        
        pixmap = QPixmap.fromImage(ImageQt.ImageQt(image))
        self.image_label.setPixmap(pixmap.scaled(400, 400, Qt.KeepAspectRatio))
        print(f"Predicted class: {pred}")

    def generate_saliency_map(self):
        
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.bmp)")
        if not file_path:
            return

        
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

        
        result = generate_saliency(file_path, model)     
            
        plt.imshow(result, cmap="hot")
        plt.show()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ModelTrainingApp()
    window.show()
    sys.exit(app.exec())