
// import React, { useState, useRef } from 'react';
// import './PredictPage_PET.css';
// import Navbar from '../Navbar/Navbar';
// import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
// import jsPDF from 'jspdf';
// import html2canvas from 'html2canvas';

// function PredictPagePET() {
//   const backendBaseUrl = 'http://127.0.0.1:5000';
//   const [step, setStep] = useState(1);
//   const [file, setFile] = useState(null);
//   const [fileName, setFileName] = useState('');
//   const [uploadedImage, setUploadedImage] = useState(null);
//   const [uploadProgress, setUploadProgress] = useState(0);
//   const [isUploaded, setIsUploaded] = useState(false);
//   const [fileType, setFileType] = useState('');
//   const [fileSize, setFileSize] = useState(0);
//   const [fileTypeError, setFileTypeError] = useState(false);
//   const [loadingExplanation, setLoadingExplanation] = useState(false);
//   const [explanations, setExplanations] = useState({ lime: null, gradcam: null, ig: null });
//   const [isLoading, setIsLoading] = useState(false);
//   const interpretationsRef = useRef(null);

//   const [predictionResult, setPredictionResult] = useState({
//     predicted_label: '',
//     confidence: 0,
//     actual_label: 'Unknown',
//     all_confidences: {},
//   });

//   const handleFileSelect = (event) => {
//     const selectedFile = event.target.files[0];
//     if (selectedFile) {
//       const isValidFileType = /\.(jpg|jpeg|png)$/i.test(selectedFile.name);
//       if (!isValidFileType) {
//         setFileTypeError(true);
//         alert('Only .jpg, .jpeg, and .png files are allowed.');
//         return;
//       }
//       setFileTypeError(false);
//       setFile(selectedFile);
//       setFileName(selectedFile.name);
//       setFileType(selectedFile.type);
//       setFileSize(selectedFile.size);
//       setUploadProgress(0);
//       setIsUploaded(false);
//       setUploadedImage(URL.createObjectURL(selectedFile));
//     }
//   };

//   const handleUpload = () => {
//     if (file) {
//       setUploadProgress(0);
//       const interval = setInterval(() => {
//         setUploadProgress(oldProgress => {
//           const newProgress = Math.min(oldProgress + 10, 100);
//           if (newProgress === 100) {
//             clearInterval(interval);
//             setIsUploaded(true);
//           }
//           return newProgress;
//         });
//       }, 100);
//     }
//   };

//   const handlePreview = () => {
//     if (file) setStep(2);
//     else alert("Please upload a file.");
//   };

//   const handlePredict = async () => {
//     if (!file) return;
//     setIsLoading(true);
//     try {
//       const formData = new FormData();
//       formData.append('file', file);

//       const response = await fetch(`${backendBaseUrl}/predict_pet`, {
//         method: 'POST',
//         body: formData,
//       });
//       const data = await response.json();
//       setPredictionResult({
//         predicted_label: data.predicted_label || 'N/A',
//         confidence: data.confidence || 0,
//         actual_label: data.actual_label || 'Unknown',
//         all_confidences: data.all_confidences || {},
//       });
//       setStep(3);
//     } catch (error) {
//       alert('Failed to get prediction.');
//     } finally {
//       setIsLoading(false);
//     }
//   };

//   const handleVisualInterpretations = async () => {
//     setIsLoading(true);
//     try {
//       const response = await fetch(`${backendBaseUrl}/explain_pet`);
//       const data = await response.json();
//       setExplanations({
//         lime: data.lime || null,
//         gradcam: data.gradcam || null,
//         ig: data.ig || null,
//       });
//       setStep(4);
//     } catch (error) {
//       alert('Failed to load explanations.');
//       setExplanations({ lime: null, gradcam: null, ig: null });
//     } finally {
//       setIsLoading(false);
//     }
//   };

//   const handleDownloadPDF = async () => {
//     if (!interpretationsRef.current) {
//       alert("No visual interpretations found.");
//       return;
//     }

//     try {
//       const canvas = await html2canvas(interpretationsRef.current, {
//         useCORS: true,
//         scale: 2,
//         scrollX: 0,
//         scrollY: -window.scrollY,
//         windowWidth: interpretationsRef.current.scrollWidth,
//         windowHeight: interpretationsRef.current.scrollHeight,
//       });

//       const imgData = canvas.toDataURL('image/png');
//       const imgWidth = canvas.width;
//       const imgHeight = canvas.height;

//       const pdf = new jsPDF({
//         orientation: 'portrait',
//         unit: 'pt',
//         format: 'a4',
//       });

//       const pdfWidth = 595.28;
//       const pdfHeight = 841.89;
//       const margin = 40;
//       let currentY = margin;

//       pdf.setFontSize(16);
//       pdf.setFont(undefined, 'bold');
//       pdf.text('Visual Interpretations Report', pdfWidth / 2, currentY, { align: 'center' });
//       currentY += 25;

//       pdf.setFontSize(11);
//       pdf.setFont(undefined, 'normal');
//       pdf.text(`Source File: ${fileName || 'N/A'}`, pdfWidth / 2, currentY, { align: 'center' });
//       currentY += 25;

//       pdf.setFont(undefined, 'bold');
//       pdf.text('Prediction Result:', margin, currentY);
//       currentY += 15;

//       pdf.setFont(undefined, 'normal');
//       pdf.text(`- Predicted Stage: ${predictionResult.predicted_label || 'N/A'}`, margin + 15, currentY);
//       currentY += 15;
//       pdf.text(`- Confidence Score: ${predictionResult.confidence?.toFixed(2) ?? '0.00'}%`, margin + 15, currentY);
//       currentY += 25;

//       pdf.setFontSize(14);
//       pdf.setFont(undefined, 'bold');
//       pdf.text('Explanation Visualizations:', pdfWidth / 2, currentY, { align: 'center' });
//       currentY += 20;

//       const availableWidth = pdfWidth - margin * 2;
//       const availableHeight = pdfHeight - currentY - margin;
//       const scale = Math.min(availableWidth / imgWidth, availableHeight / imgHeight);
//       const finalImageWidth = imgWidth * scale;
//       const finalImageHeight = imgHeight * scale;
//       const xOffset = (pdfWidth - finalImageWidth) / 2;

//       pdf.addImage(imgData, 'PNG', xOffset, currentY, finalImageWidth, finalImageHeight);
//       pdf.save(`visual_interpretations_${fileName.split('.')[0] || 'report'}.pdf`);
//     } catch (err) {
//       console.error("PDF generation failed", err);
//       alert("Failed to generate PDF.");
//     }
//   };

//   const resetState = () => {
//     setStep(1);
//     setFile(null);
//     setFileName('');
//     setUploadedImage(null);
//     setUploadProgress(0);
//     setIsUploaded(false);
//     setFileType('');
//     setFileSize(0);
//     setFileTypeError(false);
//     setExplanations({ lime: null, gradcam: null, ig: null });
//     setLoadingExplanation(false);
//     setIsLoading(false);
//     setPredictionResult({ predicted_label: '', confidence: 0, actual_label: 'Unknown', all_confidences: {} });
//   };

//   const renderBarChart = (data) => {
//     if (!data || !data.all_confidences) {
//       return <p>No confidence data available.</p>;
//     }

//     const chartData = Object.entries(data.all_confidences).map(([stage, confidence]) => ({
//       stage,
//       confidence: Number(confidence) || 0,
//     }));

//     return (
//       <ResponsiveContainer width="100%" height={300}>
//         <BarChart data={chartData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
//           <CartesianGrid strokeDasharray="3 3" />
//           <XAxis dataKey="stage" />
//           <YAxis tickFormatter={(value) => `${value}%`} domain={[0, 100]} />
//           <Tooltip formatter={(value) => `${value.toFixed(2)}%`} />
//           <Legend />
//           <Bar dataKey="confidence" fill="#1e90ff" />
//         </BarChart>
//       </ResponsiveContainer>
//     );
//   };

//   const renderStepContent = () => {
//     switch (step) {
//       case 1:
//         return (
//           <>
//             <div className="file-upload-container">
//               <div className="file-input-container">
//                 <input type="file" onChange={handleFileSelect} className="file-input" />
//                 {fileTypeError && <p className="error-message">Invalid file type.</p>}
//               </div>
//               <button className="btn" onClick={handleUpload} disabled={!file}>Upload</button>
//             </div>
//             {file && (
//               <div className="upload-info-container">
//                 <div className="upload-info">
//                   <span className="file-name">{fileName}</span>
//                   <progress value={uploadProgress} max="100"></progress>
//                   <span className="upload-percentage">{uploadProgress}%</span>
//                 </div>
//               </div>
//             )}
//             <div className="step-actions">
//               {uploadProgress === 100 && <button className="btn btn-preview" onClick={handlePreview}>Preview</button>}
//             </div>
//           </>
//         );
//       case 2:
//         return (
//           <>
//             <div className="preview-container">
//               {uploadedImage ? <img src={uploadedImage} alt="Preview" className="preview-image" /> : <p>No image uploaded.</p>}
//               <div className="file-details">
//                 <p><strong>File:</strong> {fileName}</p>
//                 <p><strong>Type:</strong> {fileType}</p>
//                 <p><strong>Size:</strong> {fileSize.toLocaleString()} bytes</p>
//               </div>
//             </div>
//             <div className="step-actions">
//               <button className="btn btn-back" onClick={() => setStep(1)}>Back</button>
//               <button className="btn btn-predict" onClick={handlePredict}>Predict</button>
//             </div>
//           </>
//         );
//       case 3:
//         return (
//           <>
//             <div className="result-container">
//               <div className="result-top-section">
//                 <div className="result-details">
//                   <div className="detail-item"><span className="detail-label">File</span>: {fileName}</div>
//                   <div className="detail-item"><span className="detail-label">Stage</span>: {predictionResult.predicted_label}</div>
//                   <div className="detail-item"><span className="detail-label">Confidence</span>: {predictionResult.confidence.toFixed(2)}%</div>
//                 </div>
//                 {uploadedImage && <img src={uploadedImage} alt="Result" className="result-image" />}
//               </div>
//               <div className="confidence-bars">
//                 <h3 className="confidence-bar-heading">Results Summary</h3>
//                 {renderBarChart(predictionResult)}
//               </div>
//             </div>
//             <div className="step-actions">
//               <button className="btn btn-back" onClick={resetState}>New Scan</button>
//               <button className="btn btn-interpretations" onClick={handleVisualInterpretations}>Visual Interpretations</button>
//             </div>
//           </>
//         );
//       case 4:
//         return (
//           <>
//             <div className="interpretations-container" ref={interpretationsRef}>
//               {loadingExplanation ? (
//                 <div className="loading-container">
//                   <div className="spinner"></div>
//                   <p>Loading...</p>
//                 </div>
//               ) : (
//                 <div className="model-column single">
//                   <h2>Model 1 Interpretation</h2>
//                   {explanations.lime && <img src={`data:image/png;base64,${explanations.lime}`} alt="LIME" className="interpretation-image" />}
//                   {explanations.gradcam && <img src={`data:image/png;base64,${explanations.gradcam}`} alt="GradCAM" className="interpretation-image" />}
//                   {explanations.ig && <img src={`data:image/png;base64,${explanations.ig}`} alt="IG" className="interpretation-image" />}
//                 </div>
//               )}
//             </div>
//             <div className="step-actions">
//               <button className="btn btn-back" onClick={resetState}>New Scan</button>
//               <button className="btn btn-secondary btn-download" onClick={handleDownloadPDF}>Download Interpretations</button>
//               <button className="btn btn-preview" onClick={() => setStep(3)}>Go Back</button>
//             </div>
//           </>
//         );
//       default:
//         return null;
//     }
//   };

//   return (
//     <div className="predict-page-pet-container">
//       <Navbar />
//       <div className="steps-container">
//         <div className="step-indicator-container">
//           <div className={`step ${step === 1 ? 'active' : ''}`}>1. Upload PET</div>
//           <div className={`step ${step === 2 ? 'active' : ''}`}>2. Preview PET</div>
//           <div className={`step ${step === 3 ? 'active' : ''}`}>3. Detection Result</div>
//           <div className={`step ${step === 4 ? 'active' : ''}`}>4. Visual Interpretations</div>
//         </div>
//         <div className="step-content">
//           {renderStepContent()}
//         </div>
//       </div>

//       {isLoading && (
//         <div className="loading-screen">
//           <div className="loading-content">
//             <div className="spinner"></div>
//             <p>Loading...</p>
//           </div>
//         </div>
//       )}
//     </div>
//   );
// }

// export default PredictPagePET;

import React, { useState, useRef, useEffect } from 'react';
import './PredictPage_PET.css'; // Ensure this CSS file exists and is styled accordingly
import Navbar from '../Navbar/Navbar';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import jsPDF from 'jspdf';
import html2canvas from 'html2canvas';

function PredictPagePET() {
  const backendBaseUrl = 'http://127.0.0.1:5000';
  const [step, setStep] = useState(1);
  const [file, setFile] = useState(null);
  const [fileName, setFileName] = useState('');
  const [uploadedImage, setUploadedImage] = useState(null);
  const [uploadProgress, setUploadProgress] = useState(0);
  // Removed isUploaded as uploadProgress === 100 serves the same purpose
  const [fileType, setFileType] = useState('');
  const [fileSize, setFileSize] = useState(0);
  const [fileTypeError, setFileTypeError] = useState(false);
  const [loadingExplanation, setLoadingExplanation] = useState(false); // Added this state
  const [explanations, setExplanations] = useState({ lime: null, gradcam: null, ig: null });
  const [isLoading, setIsLoading] = useState(false);
  const interpretationsRef = useRef(null);
  const fileInputRef = useRef(null); // Ref for the file input element

  const [predictionResult, setPredictionResult] = useState({
    predicted_label: '',
    confidence: 0,
    actual_label: 'Unknown',
    all_confidences: {},
  });

  // --- START: Updated handlers ---
  const handleFileSelect  = async (event) => {
    const selectedFile = event.target.files[0];
    if (selectedFile) {
        const isValidFileType = /\.(jpg|jpeg|png)$/i.test(selectedFile.name);
        if (!isValidFileType) {
            setFileTypeError(true);
            alert('Only .jpg, .jpeg, and .png files are allowed.');
            event.target.value = null;
            resetFileStates();
            return;
        }

        // ---  Start: Validate PET Image using Backend Validator ---
        try {
            const formData = new FormData();
            formData.append('file', selectedFile);

            const response = await fetch('http://127.0.0.1:5000/validate_pet', { // << Change endpoint to /validate_pet
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }

            const result = await response.json();

            if (!result.is_valid) {
                alert(' Uploaded image is not recognised as a valid PET scan.');
                event.target.value = null;
                resetFileStates();
                return;
            }
        } catch (error) {
            console.error("Validation error:", error);
            alert('Failed to validate the PET scan. Please try again.');
            event.target.value = null;
            resetFileStates();
            return;
        }
        // ---  End PET Validation ---

        if (uploadedImage) {
            URL.revokeObjectURL(uploadedImage);
        }
        setFileTypeError(false);
        setFile(selectedFile);
        setFileName(selectedFile.name);
        setFileType(selectedFile.type);
        setFileSize(selectedFile.size);
        setUploadProgress(0);
        setUploadedImage(URL.createObjectURL(selectedFile));
    } else {
        handleCancelUpload();
    }
};


   // --- NEW: Handler for the Cancel button in Step 1 ---
   const handleCancelUpload = () => {
    setFile(null);
    setFileName('');
    if (uploadedImage) {
        URL.revokeObjectURL(uploadedImage);
    }
    setUploadedImage(null);
    setUploadProgress(0);
    setFileType('');
    setFileSize(0);
    setFileTypeError(false);
    // Reset the actual file input element
    if (fileInputRef.current) {
        fileInputRef.current.value = null;
    }
};

const resetFileStates = () => {
  setFile(null);
  setFileName('');
  setFileType('');
  setFileSize(0);
  setUploadProgress(0);
  setFileTypeError(false);
  if (uploadedImage) {
      URL.revokeObjectURL(uploadedImage);
  }
  setUploadedImage(null);
  if (fileInputRef.current) {
      fileInputRef.current.value = null;
  }
};

// --- END: New Cancel Handler ---

  const handleUpload = () => {
    if (file) {
      setUploadProgress(0); // Start progress from 0
      const interval = setInterval(() => {
        setUploadProgress(oldProgress => {
           // Prevent progress update if file was cancelled during upload
           if (!file) {
            clearInterval(interval);
            return 0; // Reset progress if file is gone
           }
          const newProgress = Math.min(oldProgress + 10, 100);
          if (newProgress === 100) {
            clearInterval(interval);
            // No need for setIsUploaded, just check progress === 100
          }
          return newProgress;
        });
      }, 100);
    }
  };

  const handlePreview = () => {
    // Proceed only if file exists and upload simulation reached 100%
    if (file && uploadProgress === 100) setStep(2);
    else if (!file) alert("Please upload a file.");
    else alert("Please complete the upload indication first.");
  };

  const handlePredict = async () => {
    if (!file) return;
    setIsLoading(true);
    try {
      const formData = new FormData();
      formData.append('file', file); // Use 'file' key for backend

      const response = await fetch(`${backendBaseUrl}/predict_pet`, { // Ensure correct endpoint
        method: 'POST',
        body: formData,
      });
       if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      setPredictionResult({
        predicted_label: data.predicted_label || 'N/A',
        confidence: data.confidence || 0,
        actual_label: data.actual_label || 'Unknown',
        all_confidences: data.all_confidences || {},
      });
      setStep(3);
    } catch (error) {
      console.error("Prediction failed:", error);
      alert(`Failed to get prediction: ${error.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  const handleVisualInterpretations = async () => {
    if (!file) { // Need file context
      alert("Cannot request interpretations without an uploaded image context.");
      return;
    }
    // Add check for prediction result before proceeding
    if (!predictionResult.predicted_label) {
        alert("Please run the prediction first to get context for interpretations.");
        return;
    }
    setIsLoading(true);
    setLoadingExplanation(true); // Use specific loader
    try {
      // Assuming GET is ok if backend remembers the last file from /predict_pet
      const response = await fetch(`${backendBaseUrl}/explain_pet`); // Ensure correct endpoint
       if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();

      // Basic validation - check if *any* data was received
      const hasContent = data && (data.lime || data.gradcam || data.ig);

      if (!hasContent) {
          console.warn("Received empty or invalid explanation data:", data);
          alert("Explanations received from the server were empty or unreadable.");
          setExplanations({ lime: null, gradcam: null, ig: null });
          // Don't automatically proceed to step 4 if data is bad
      } else {
           setExplanations({
              lime: data.lime || null,
              gradcam: data.gradcam || null,
              ig: data.ig || null,
          });
          setStep(4); // Proceed only if data looks okay
      }

    } catch (error) {
        console.error("Explanation fetch failed:", error);
        alert(`Failed to load explanations: ${error.message}`);
        setExplanations({ lime: null, gradcam: null, ig: null }); // Reset on error
    } finally {
      setIsLoading(false);
      setLoadingExplanation(false);
    }
  };

  // --- PDF Download Function (Kept similar, check logic against needs) ---
  const handleDownloadPDF = async () => {
    const elementToCapture = interpretationsRef.current;
    if (!elementToCapture) {
      alert("Could not find the interpretations content to download.");
      console.error("interpretationsRef is not attached or the component is not rendered.");
      return;
    }

    const hasExplanations = explanations.lime || explanations.gradcam || explanations.ig;
    if (!hasExplanations) {
        alert("No visual interpretations are available to download.");
        return;
    }

    if (!predictionResult.predicted_label) {
       alert("Prediction results are not available to include in the PDF.");
       // Optionally proceed without prediction info or stop here
       // return;
    }

    setIsLoading(true); // Indicate activity
    try {
      const canvas = await html2canvas(elementToCapture, {
        useCORS: true,
        scale: 2,
        logging: false,
        scrollX: 0,
        scrollY: -window.scrollY,
        windowWidth: elementToCapture.scrollWidth,
        windowHeight: elementToCapture.scrollHeight,
      });

      const imgData = canvas.toDataURL('image/png');
      const imgWidth = canvas.width;
      const imgHeight = canvas.height;

      const pdf = new jsPDF({
        orientation: imgWidth > imgHeight ? 'landscape' : 'portrait', // Auto orientation
        unit: 'pt',
        format: 'a4',
      });

      const pdfWidth = pdf.internal.pageSize.getWidth();
      const pdfHeight = pdf.internal.pageSize.getHeight();
      const margin = 40;
      let currentY = margin;

      // --- Add Text Content ---
      pdf.setFontSize(16);
      pdf.setFont(undefined, 'bold');
      pdf.text('Visual Interpretations Report (PET)', pdfWidth / 2, currentY, { align: 'center' });
      currentY += 20;

      pdf.setFontSize(11);
      pdf.setFont(undefined, 'normal');
      pdf.text(`Source File: ${fileName || 'N/A'}`, pdfWidth / 2, currentY, { align: 'center' });
      currentY += 25;

      if (predictionResult.predicted_label) {
          pdf.setFont(undefined, 'bold');
          pdf.text('Prediction Result:', margin, currentY);
          currentY += 15;

          pdf.setFont(undefined, 'normal');
          pdf.text(`- Predicted Stage: ${predictionResult.predicted_label}`, margin + 15, currentY);
          currentY += 15;
          pdf.text(`- Confidence Score: ${predictionResult.confidence?.toFixed(2) ?? 'N/A'}%`, margin + 15, currentY);
          currentY += 25;
      } else {
          pdf.setFont(undefined, 'italic');
          pdf.text('Prediction results were not available for this report.', margin, currentY);
          currentY += 25;
      }

      pdf.setFontSize(14);
      pdf.setFont(undefined, 'bold');
      pdf.text('Explanation Visualizations:', pdfWidth / 2, currentY, { align: 'center' });
      currentY += 20;
      // --- End Text Content ---

      // --- Add Image Content ---
      const availableWidth = pdfWidth - margin * 2;
      const availableHeight = pdfHeight - currentY - margin; // Adjust available height
      const scale = Math.min(availableWidth / imgWidth, availableHeight / imgHeight);
      const finalImageWidth = imgWidth * scale;
      const finalImageHeight = imgHeight * scale;
      const xOffset = (pdfWidth - finalImageWidth) / 2; // Center horizontally
      const yOffset = currentY; // Place image below text

      if (yOffset + finalImageHeight > pdfHeight - margin) {
         console.warn("Calculated image height might exceed page limits after adding text.");
      }

      pdf.addImage(imgData, 'PNG', xOffset, yOffset, finalImageWidth, finalImageHeight);
      pdf.save(`visual_interpretations_pet_${fileName.split('.')[0] || 'report'}.pdf`);

    } catch (err) {
      console.error("PDF generation failed", err);
      alert("Failed to generate PDF for visual interpretations. See console for details.");
    } finally {
      setIsLoading(false); // Stop loading indicator
    }
  };

  // --- Reset State (uses handleCancelUpload) ---
  const resetState = () => {
    setStep(1);
    // Use the cancel handler logic to reset file-related state
    handleCancelUpload();
    // Reset states not covered by handleCancelUpload
    setExplanations({ lime: null, gradcam: null, ig: null });
    setLoadingExplanation(false);
    setIsLoading(false);
    setPredictionResult({ predicted_label: '', confidence: 0, actual_label: 'Unknown', all_confidences: {} });
  };

  // --- useEffect for cleanup ---
  useEffect(() => {
    // Cleanup function for Object URL
    return () => {
        if (uploadedImage) {
            URL.revokeObjectURL(uploadedImage);
        }
    };
}, [uploadedImage]); // Dependency ensures cleanup on change/unmount

// --- END: Updated handlers ---

  const renderBarChart = (data) => {
    // Keep original logic, ensure it handles potentially missing data gracefully
    if (!data || !data.all_confidences || Object.keys(data.all_confidences).length === 0) {
        return <p>No confidence data available.</p>;
    }

    const chartData = Object.entries(data.all_confidences).map(([stage, confidence]) => ({
      stage,
      confidence: Number(confidence) || 0,
    }));

    // Basic check if chartData is empty after processing
    if (chartData.length === 0) {
        return <p>Confidence data could not be processed for chart.</p>;
    }

    return (
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={chartData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="stage" />
          <YAxis tickFormatter={(value) => `${value}%`} domain={[0, 100]} />
          <Tooltip formatter={(value) => `${value?.toFixed(2) ?? 'N/A'}%`} />
          <Legend />
          <Bar dataKey="confidence" fill="#1e90ff" />
        </BarChart>
      </ResponsiveContainer>
    );
  };

  // --- Updated Render Step Content ---
  const renderStepContent = () => {
    switch (step) {
      case 1:
        return (
          <>
            <div className="file-upload-container">
              <div className="file-input-container">
                 {/* Added ref and accept attribute */}
                <input type="file" onChange={handleFileSelect} className="file-input" accept=".jpg,.jpeg,.png" ref={fileInputRef}/>
                {fileTypeError && <p className="error-message">Invalid file type.</p>}
              </div>
               {/* Conditionally render Upload button only if file is selected and valid */}
               {file && !fileTypeError && (
                    <button className="btn" onClick={handleUpload} disabled={uploadProgress > 0 && uploadProgress < 100}>
                        {uploadProgress === 100 ? "Uploaded ✓" : "Upload"}
                    </button>
                )}
            </div>
            {file && !fileTypeError && (
              <div className="upload-info-container">
                <div className="upload-info">
                  <span className="file-name">{fileName}</span>
                  <progress value={uploadProgress} max="100"></progress>
                  <span className="upload-percentage">{uploadProgress}%</span>
                </div>
              </div>
            )}
             {/* --- Updated Step 1 Actions --- */}
             {/* Show actions only when a file is selected and valid */}
             {file && !fileTypeError && (
                <div className="step-actions step-actions-step1"> {/* Added specific class */}
                    {/* Cancel Button (Left) */}
                    <button className="btn btn-cancel" onClick={handleCancelUpload} disabled={isLoading}>
                        Cancel
                    </button>
                    {/* Preview Button (Right) */}
                    <button className="btn btn-preview" onClick={handlePreview} disabled={uploadProgress !== 100 || isLoading}>
                        Preview
                    </button>
                </div>
             )}
             {/* --- End Updated Step 1 Actions --- */}
          </>
        );
      case 2:
        return (
          <>
            <div className="preview-container">
              {uploadedImage ? <img src={uploadedImage} alt="Preview" className="preview-image" /> : <p>No image uploaded.</p>}
              <div className="file-details">
                <p><strong>File:</strong> {fileName}</p>
                <p><strong>Type:</strong> {fileType}</p>
                {/* Corrected file size display */}
                <p><strong>Size:</strong> {fileSize > 0 ? `${(fileSize / 1024).toFixed(2)} KB` : 'N/A'}</p>
              </div>
            </div>
            <div className="step-actions">
              <button className="btn btn-back" onClick={() => setStep(1)}>Back</button>
              <button className="btn btn-predict" onClick={handlePredict} disabled={isLoading}>Predict</button>
            </div>
          </>
        );
      case 3:
        return (
          <>
            <div className="result-container">
              <div className="result-top-section">
                <div className="result-details">
                  <div className="detail-item"><span className="detail-label">File</span>: {fileName}</div>
                  {/* Use nullish coalescing for safety */}
                  <div className="detail-item"><span className="detail-label">Stage</span>: {predictionResult.predicted_label || 'N/A'}</div>
                  <div className="detail-item"><span className="detail-label">Confidence</span>: {predictionResult.confidence?.toFixed(2) ?? '0.00'}%</div>
                </div>
                {uploadedImage && <img src={uploadedImage} alt="Result" className="result-image" />}
              </div>
              <div className="confidence-bars">
                <h3 className="confidence-bar-heading">Results Summary</h3>
                {renderBarChart(predictionResult)}
              </div>
            </div>
            <div className="step-actions">
              <button className="btn btn-back" onClick={resetState}>New Scan</button>
              {/* Disable button if prediction hasn't run or is loading */}
              <button className="btn btn-interpretations" onClick={handleVisualInterpretations} disabled={isLoading || !predictionResult.predicted_label}>Visual Interpretations</button>
            </div>
          </>
        );
      case 4:
        const hasAnyExplanation = explanations.lime || explanations.gradcam || explanations.ig;
        return (
          <>
            <div className="interpretations-container" ref={interpretationsRef}>
              {loadingExplanation ? (
                <div className="loading-container">
                  <div className="spinner"></div>
                  <p>Loading Explanations...</p> {/* Updated text */}
                </div>
              ) : !hasAnyExplanation ? (
                 <p className="placeholder-text">No explanation data available.</p> // Placeholder
              ) : (
                 // Assuming PET backend returns single images per method, not per model key
                <div className="model-column single"> {/* Use appropriate class */}
                  <h2>PET Scan Interpretations</h2>
                  {explanations.lime && (
                    <div className="explanation-item">
                        <h4>LIME</h4>
                        <img src={`data:image/png;base64,${explanations.lime}`} alt="LIME" className="interpretation-image" />
                    </div>
                  )}
                  {explanations.gradcam && (
                    <div className="explanation-item">
                        <h4>Grad-CAM</h4>
                        <img src={`data:image/png;base64,${explanations.gradcam}`} alt="GradCAM" className="interpretation-image" />
                    </div>
                    )}
                  {explanations.ig && (
                     <div className="explanation-item">
                        <h4>Integrated Gradients</h4>
                        <img src={`data:image/png;base64,${explanations.ig}`} alt="IG" className="interpretation-image" />
                    </div>
                    )}
                </div>
              )}
            </div>
            {/* --- Step 4 Actions --- */}
            <div className="step-actions step-actions-step4"> {/* Added specific class */}
              <button className="btn btn-back" onClick={resetState}>New Scan</button>
              <button
                className="btn btn-secondary btn-download"
                onClick={handleDownloadPDF}
                disabled={isLoading || loadingExplanation || !hasAnyExplanation}
                >
                Download Interpretations
              </button>
              <button className="btn btn-preview" onClick={() => setStep(3)}>Go Back</button>
            </div>
             {/* --- END: Step 4 Actions --- */}
          </>
        );
      default:
        return <p>An error occurred. Please refresh.</p>; // Default case
    }
  };

  // --- Main Component Return ---
  return (
    // Use a specific class for the PET page container
    <div className="predict-page-pet-container">
      <Navbar />
      <div className="steps-container">
        <div className="step-indicator-container">
          {/* Updated step labels for PET */}
          <div className={`step ${step === 1 ? 'active' : ''}`}>1. Upload PET</div>
          <div className={`step ${step === 2 ? 'active' : ''}`}>2. Preview PET</div>
          <div className={`step ${step === 3 ? 'active' : ''}`}>3. Detection Result</div>
          <div className={`step ${step === 4 ? 'active' : ''}`}>4. Visual Interpretations</div>
        </div>
        <div className="step-content">
          {renderStepContent()}
        </div>
      </div>

      {/* Use general isLoading OR specific loadingExplanation for overlay */}
      {(isLoading || loadingExplanation) && (
        <div className="loading-screen">
          <div className="loading-content">
            <div className="spinner"></div>
             {/* Show more specific message if loading explanations */}
            <p>{loadingExplanation ? 'Loading Explanations...' : 'Loading...'}</p>
          </div>
        </div>
      )}
    </div>
  );
}

export default PredictPagePET;
