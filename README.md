# 🧠 Alz-InsightNet

<div align="center">

![Alz-InsightNet Logo](https://img.shields.io/badge/Alz--InsightNet-AI%20Powered-blue?style=for-the-badge&logo=brain&logoColor=white)

**An Explainable Attention-Based Multimodal and Multimodel System for Early Alzheimer's Detection**

[![React](https://img.shields.io/badge/React-19.0.0-61DAFB?style=flat-square&logo=react&logoColor=black)](https://reactjs.org/)
[![Material-UI](https://img.shields.io/badge/Material--UI-6.4.7-0081CB?style=flat-square&logo=mui&logoColor=white)](https://mui.com/)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://tensorflow.org/)
[![Flask](https://img.shields.io/badge/Flask-2.x-000000?style=flat-square&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)

[Demo](#demo) • [Features](#features) • [Installation](#installation) • [Usage](#usage) • [API](#api-documentation) • [Contributing](#contributing)

</div>

---

## 📖 Overview

Alz-InsightNet is a cutting-edge AI-powered system designed for early detection of Alzheimer's Disease using multimodal neuroimaging data. The system combines MRI and PET scan analysis with explainable AI techniques to provide clinicians with interpretable predictions and visual explanations.

### 🎯 Key Highlights

- **Multimodal Analysis**: Processes both MRI and PET scans for comprehensive brain analysis
- **Explainable AI**: Implements LIME, Grad-CAM, and Integrated Gradients for transparent predictions
- **High Accuracy**: Achieves 99.63% accuracy on MRI data and 88% on PET data
- **Clinical Ready**: User-friendly interface designed for healthcare professionals
- **Real-time Processing**: Fast prediction and explanation generation

## 🌟 Features

### 🔬 **AI Models**
- **Ensemble MRI Model**: Modified ResNet50 + DenseNet201 with CBAM attention
- **PET Model**: Modified VGG19 with CBAM attention mechanism
- **Decision-Level Fusion**: Combines multimodal predictions for enhanced accuracy

### 🎨 **Frontend Capabilities**
- **Interactive Dashboard**: Clean, intuitive Material-UI interface
- **Multi-Scan Support**: Upload MRI, PET, or both scan types
- **Real-time Visualization**: Progress bars, charts, and heatmaps
- **PDF Reports**: Generate downloadable diagnostic reports
- **Responsive Design**: Works seamlessly across devices

### 🧮 **Explainable AI**
- **Grad-CAM**: Gradient-weighted Class Activation Mapping
- **LIME**: Local Interpretable Model-agnostic Explanations
- **Integrated Gradients**: Attribution-based explanations
- **Visual Heatmaps**: Brain region highlighting for predictions

### 📊 **Analytics & Reporting**
- **Confidence Scores**: Detailed prediction probabilities
- **Stage Classification**: AD, CN, EMCI, LMCI detection
- **Comparative Analysis**: Side-by-side model comparisons
- **Export Functionality**: PDF and image exports

## 🛠️ Technology Stack

### Frontend
```json
{
  "framework": "React 19.0.0",
  "ui_library": "Material-UI 6.4.7",
  "routing": "React Router DOM 7.3.0",
  "charts": ["Chart.js 4.4.8", "Recharts 2.15.2"],
  "pdf_generation": "jsPDF 3.0.1",
  "styling": "CSS3 + Emotion"
}
```

### Backend
```json
{
  "framework": "Flask 2.x",
  "ml_framework": "TensorFlow 2.x",
  "image_processing": "OpenCV",
  "explainability": ["LIME", "Captum", "tf-explain"],
  "data_processing": "NumPy, Pandas"
}
```

### AI/ML Stack
```json
{
  "models": ["ResNet50", "DenseNet201", "VGG19"],
  "attention": "CBAM (Convolutional Block Attention Module)",
  "techniques": ["Transfer Learning", "Ensemble Learning"],
  "explainability": ["Grad-CAM", "LIME", "Integrated Gradients"]
}
```

## 🚀 Quick Start

### Prerequisites

- Node.js 16+ and npm/yarn
- Python 3.8+
- CUDA-compatible GPU (recommended)

### Frontend Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/alz-insightnet.git
cd alz-insightnet

# Install dependencies
npm install

# Start development server
npm start
```

### Backend Setup

```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start Flask server
python app.py
```

### 🔧 Environment Variables

Create a `.env` file in the root directory:

```env
REACT_APP_API_URL=http://localhost:5000
REACT_APP_MODEL_VERSION=v1.0
REACT_APP_MAX_FILE_SIZE=50MB
```

## 📱 Usage

### 1. **Single Modality Prediction**
```bash
# MRI-only analysis
POST /predict_mri
Content-Type: multipart/form-data
file: [MRI_IMAGE]

# PET-only analysis  
POST /predict_pet
Content-Type: multipart/form-data
file: [PET_IMAGE]
```

### 2. **Multimodal Analysis**
```bash
# Combined MRI + PET analysis
POST /predict_both
Content-Type: multipart/form-data
mri: [MRI_IMAGE]
pet: [PET_IMAGE]
```

### 3. **Explainability**
```bash
# Get visual explanations
GET /explain_mri
GET /explain_pet
GET /explain_both
```

## 🎯 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|---------|----------|
| MRI Ensemble | **99.63%** | 99% | 99% | 99% |
| PET VGG19 | **88%** | 88% | 88% | 88% |
| Fused Model | **94%** | 93% | 94% | 93% |

### 📈 Classification Stages
- **AD**: Alzheimer's Disease
- **CN**: Cognitively Normal
- **EMCI**: Early Mild Cognitive Impairment
- **LMCI**: Late Mild Cognitive Impairment

## 🔍 API Documentation

<details>
<summary>📋 Click to expand API endpoints</summary>

### Prediction Endpoints

#### MRI Prediction
```http
POST /predict_mri
Content-Type: multipart/form-data

Response:
{
  "predicted_label": "AD",
  "confidence": 95.67,
  "actual_label": "Unknown",
  "all_confidences": {
    "AD": 95.67,
    "CN": 2.15,
    "EMCI": 1.23,
    "LMCI": 0.95
  }
}
```

#### Explanation Endpoints
```http
GET /explain_mri

Response:
{
  "gradcam": {
    "densenet": "base64_image_string",
    "resnet": "base64_image_string"
  },
  "ig": {
    "densenet": "base64_image_string",
    "resnet": "base64_image_string"
  },
  "lime": {
    "densenet": "base64_image_string",
    "resnet": "base64_image_string"
  }
}
```

</details>

## 🗂️ Project Structure

```
alz-insightnet/
├── 📁 public/
│   ├── index.html
│   └── manifest.json
├── 📁 src/
│   ├── 📁 components/
│   │   ├── Dashboard/
│   │   ├── Prediction/
│   │   └── Reports/
│   ├── 📁 services/
│   │   └── api.js
│   ├── 📁 utils/
│   └── App.js
├── 📁 backend/
│   ├── app.py
│   ├── 📁 models/
│   ├── 📁 utils/
│   └── requirements.txt
├── 📁 docs/
├── package.json
└── README.md
```

## 🧪 Testing

```bash
# Run frontend tests
npm test

# Run backend tests
cd backend
python -m pytest tests/

# Run integration tests
npm run test:integration
```

## 🔬 Research & Publications

This project is based on the dissertation:
> **"Alz-InsightNet: An Explainable Attention-Based Multimodal and Multimodel System for Early Alzheimer's Detection"**  
> *Vithyabavan Sunthareswaran, University of Westminster, April 2024*

### 📚 Key Contributions
1. Novel ensemble approach combining ResNet50 and DenseNet201
2. Integration of CBAM attention mechanisms
3. Comprehensive explainability framework
4. Clinical-ready interface design

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md).

### Development Workflow
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **University of Westminster** - Academic support and guidance
- **ADNI Dataset** - Neuroimaging data for model training
- **Medical Experts** - Clinical validation and feedback
- **Open Source Community** - Tools and frameworks

## 📞 Contact & Support

<div align="center">

**Vithyabavan Sunthareswaran**  
*Software Engineer & AI Researcher*

[![Email](https://img.shields.io/badge/Email-Contact-red?style=flat-square&logo=gmail)](mailto:your.email@example.com)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat-square&logo=linkedin)](https://linkedin.com/in/yourprofile)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?style=flat-square&logo=github)](https://github.com/yourusername)

</div>

---

<div align="center">

**⭐ Star this repository if you found it helpful!**

*Made with ❤️ for advancing AI in healthcare*

</div>
