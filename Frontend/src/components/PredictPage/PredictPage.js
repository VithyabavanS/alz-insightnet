// import React, { useState, useRef, useEffect } from 'react';
// import './PredictPage.css'; // Ensure this CSS file exists and is styled
// import Navbar from '../Navbar/Navbar'; // Ensure this path is correct
// import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
// import jsPDF from 'jspdf';
// import html2canvas from 'html2canvas';

// function PredictPage() {
//     const backendBaseUrl = 'http://127.0.0.1:5000';
//     const [step, setStep] = useState(1);
//     const [file, setFile] = useState(null);
//     const [fileName, setFileName] = useState('');
//     const [uploadedImage, setUploadedImage] = useState(null);
//     const [uploadProgress, setUploadProgress] = useState(0);
//     const [fileType, setFileType] = useState('');
//     const [fileSize, setFileSize] = useState(0);
//     const [fileTypeError, setFileTypeError] = useState(false);
//     const [loadingExplanation, setLoadingExplanation] = useState(false);
//     const [explanations, setExplanations] = useState({ lime: null, gradcam: null, ig: null });
//     const [isLoading, setIsLoading] = useState(false);
//     const [predictionResult, setPredictionResult] = useState({
//         predicted_label: '',
//         confidence: 0,
//         actual_label: 'Unknown',
//         all_confidences: {},
//     });

//     const resultRef = useRef(null);
//     const interpretationsRef = useRef(null);

//     // --- START: Updated handleDownloadVisualInterpretations (from second block, integrated into first block's structure) ---
//     const handleDownloadVisualInterpretations = async () => {
//         const elementToCapture = interpretationsRef.current;
//         if (!elementToCapture) {
//             alert("Could not find the interpretations content to download.");
//             console.error("interpretationsRef is not attached or the component is not rendered.");
//             return;
//         }

//         const hasExplanations = explanations.lime || explanations.gradcam || explanations.ig;
//          // Check if explanations have actual data (e.g., at least one model has one type)
//          const hasExplanationData = hasExplanations && Object.values(hasExplanations).some(method => method && Object.keys(method).length > 0);
//          if (!hasExplanationData) {
//              alert("No visual interpretations are available to download.");
//              return;
//          }

//          // Check if prediction result is available
//          if (!predictionResult.predicted_label) {
//             alert("Prediction results are not available to include in the PDF.");
//             // Optionally proceed without prediction info or stop here
//             // return; // You might uncomment this if prediction data is mandatory
//          }

//         setIsLoading(true); // Use general loading state to indicate activity
//         try {
//             const canvas = await html2canvas(elementToCapture, {
//                  useCORS: true, // Handle base64 images correctly
//                  scale: 2, // Improve resolution
//                  logging: false,
//                  scrollX: 0,
//                  scrollY: -window.scrollY, // Account for page scroll if element isn't fully visible
//                  windowWidth: elementToCapture.scrollWidth,
//                  windowHeight: elementToCapture.scrollHeight
//             });

//             const imgData = canvas.toDataURL('image/png');
//             const imgWidth = canvas.width;
//             const imgHeight = canvas.height;

//             // A4 dimensions in points: 595.28 x 841.89 pt
//             const pdfWidth = 595.28;
//             const pdfHeight = 841.89;

//             const pdf = new jsPDF({
//                  orientation: imgWidth > imgHeight ? 'landscape' : 'portrait',
//                  unit: 'pt',
//                  format: 'a4'
//             });

//             // --- START: Add Prediction Info Text to PDF ---
//             let currentY = 30; // Initial Y position for text

//             pdf.setFontSize(16);
//             pdf.setFont(undefined, 'bold'); // Use bold for title
//             pdf.text('Visual Interpretations Report', pdfWidth / 2, currentY, { align: 'center' });
//             currentY += 20; // Increase Y position

//             pdf.setFontSize(11);
//             pdf.setFont(undefined, 'normal'); // Reset font style
//             pdf.text(`Source File: ${fileName || 'N/A'}`, pdfWidth / 2, currentY, { align: 'center' });
//             currentY += 25; // Increase Y position

//             // Add Predicted Case and Confidence
//             if (predictionResult.predicted_label) {
//                 pdf.setFont(undefined, 'bold');
//                 pdf.text('Prediction Result:', 40, currentY); // Use a fixed X margin (e.g., 40pt)
//                 currentY += 15;

//                 pdf.setFont(undefined, 'normal');
//                 pdf.text(`- Predicted Stage: ${predictionResult.predicted_label}`, 55, currentY); // Indent slightly
//                 currentY += 15;
//                 pdf.text(`- Confidence Score: ${predictionResult.confidence?.toFixed(2) ?? 'N/A'}%`, 55, currentY);
//                 currentY += 25; // Add more space before the image section title
//             } else {
//                  // Optionally add a placeholder if prediction data isn't available but you still proceed
//                  pdf.setFont(undefined, 'italic');
//                  pdf.text('Prediction results were not available for this report.', 40, currentY);
//                  currentY += 25;
//             }

//             pdf.setFontSize(14);
//             pdf.setFont(undefined, 'bold');
//             pdf.text('Explanation Visualizations:', pdfWidth / 2, currentY, { align: 'center' });
//             currentY += 20; // Space after the section title, before the image
//             // --- END: Add Prediction Info Text to PDF ---


//             // Calculate the aspect ratio to fit the image on the page below the text
//             // Adjust available height for the image
//             const availableHeight = pdfHeight - currentY - 20; // Subtract current Y and bottom margin
//             // Adjust available width slightly for margins (e.g., 40pt each side)
//             const ratio = Math.min((pdfWidth - 80) / imgWidth, availableHeight / imgHeight);
//             const pdfImageWidth = imgWidth * ratio;
//             const pdfImageHeight = imgHeight * ratio;

//             // Center the image (optional) below the text
//             const xOffset = (pdfWidth - pdfImageWidth) / 2;
//             const yOffset = currentY; // Start image after the text

//             // Ensure image doesn't overflow page (redundant check if ratio is calculated correctly)
//             if (yOffset + pdfImageHeight > pdfHeight - 20) {
//                 console.warn("Calculated image height might exceed page limits after adding text.");
//                 // Optional: further reduce scale or handle multi-page
//             }

//             pdf.addImage(imgData, 'PNG', xOffset, yOffset, pdfImageWidth, pdfImageHeight);

//             pdf.save(`visual_interpretations_${fileName.split('.')[0] || 'report'}.pdf`);

//         } catch (error) {
//             console.error("Error generating PDF:", error);
//             alert("Failed to generate PDF for visual interpretations. See console for details.");
//         } finally {
//             setIsLoading(false); // Stop loading indicator
//         }
//     };
//     // --- END: Updated handleDownloadVisualInterpretations ---


//     const handleFileSelect = (event) => {
//         const selectedFile = event.target.files[0];
//         if (selectedFile) {
//             const isValidFileType = /\.(jpg|jpeg|png)$/i.test(selectedFile.name);
//             if (!isValidFileType) {
//                 setFileTypeError(true);
//                 alert('Only .jpg, .jpeg, and .png files are allowed.');
//                 event.target.value = null; // Clear invalid input
//                 return;
//             }
//             if (uploadedImage) {
//                 URL.revokeObjectURL(uploadedImage); // Clean previous URL
//             }
//             setFileTypeError(false);
//             setFile(selectedFile);
//             setFileName(selectedFile.name);
//             setFileType(selectedFile.type);
//             setFileSize(selectedFile.size);
//             setUploadProgress(0);
//             setUploadedImage(URL.createObjectURL(selectedFile));
//         } else {
//              // Handle cancellation
//              setFile(null);
//              setFileName('');
//              if (uploadedImage) {
//                  URL.revokeObjectURL(uploadedImage);
//              }
//              setUploadedImage(null);
//              setUploadProgress(0);
//              setFileType('');
//              setFileSize(0);
//              setFileTypeError(false);
//         }
//     };

//     const handleUpload = () => { // Simulate upload progress
//         if (file) {
//             setUploadProgress(0);
//             const interval = setInterval(() => {
//                 setUploadProgress(oldProgress => {
//                     const newProgress = Math.min(oldProgress + 10, 100);
//                     if (newProgress === 100) {
//                         clearInterval(interval);
//                     }
//                     return newProgress;
//                 });
//             }, 100); // Keep original interval
//         }
//     };

//     const handlePreview = () => {
//         // Proceed only if file exists and upload simulation reached 100%
//         if (file && uploadProgress === 100) setStep(2);
//         else if (!file) alert("Please upload a file.");
//         else alert("Please complete the upload indication first."); // Keep original message
//     };

//     const handlePredict = async () => { // Predict using single file
//         if (!file) return;
//         setIsLoading(true);
//         try {
//             const formData = new FormData();
//             formData.append('file', file); // Use 'file' key for backend

//             const response = await fetch(`${backendBaseUrl}/predict_mri`, { // Use single MRI endpoint
//                 method: 'POST',
//                 body: formData,
//             });
//              if (!response.ok) {
//                 // Keep original error handling
//                 throw new Error(`HTTP error! status: ${response.status}`);
//             }
//             const data = await response.json();
//             setPredictionResult({
//                 predicted_label: data.predicted_label || 'N/A',
//                 confidence: data.confidence || 0,
//                 actual_label: data.actual_label || 'Unknown',
//                 all_confidences: data.all_confidences || {},
//             });
//             setStep(3);
//         } catch (error) {
//             console.error("Prediction failed:", error);
//             alert(`Failed to get prediction: ${error.message}`); // Keep original alert
//         } finally {
//             setIsLoading(false);
//         }
//     };

//     const handleVisualInterpretations = async () => { // Explain single file
//          if (!file) { // Need file context
//              alert("Cannot request interpretations without an uploaded image context.");
//              return;
//          }
//          // Add check for prediction result before proceeding
//          if (!predictionResult.predicted_label) {
//              alert("Please run the prediction first to get context for interpretations.");
//              return;
//          }
//         setIsLoading(true);
//         setLoadingExplanation(true);
//         try {
//             // Assuming GET is ok if backend remembers the last file from /predict_mri
//             const response = await fetch(`${backendBaseUrl}/explain_mri`); // Use single MRI explanation endpoint
//              if (!response.ok) {
//                  // Keep original error handling
//                 throw new Error(`HTTP error! status: ${response.status}`);
//             }
//             const data = await response.json();

//              // Basic validation added - check if *any* data was received
//              const hasData = data && (data.lime || data.gradcam || data.ig);
//              const hasContent = hasData && (Object.keys(data.lime || {}).length > 0 || Object.keys(data.gradcam || {}).length > 0 || Object.keys(data.ig || {}).length > 0);

//              if (!hasContent) {
//                  console.warn("Received empty or invalid explanation data:", data);
//                  alert("Explanations received from the server were empty or unreadable.");
//                  setExplanations({ lime: null, gradcam: null, ig: null });
//                  // Don't automatically proceed to step 4 if data is bad
//              } else {
//                   setExplanations({
//                       lime: data.lime || null,
//                       gradcam: data.gradcam || null,
//                       ig: data.ig || null,
//                   });
//                   setStep(4); // Proceed only if data looks okay
//              }
//         } catch (error) {
//             console.error("Explanation fetch failed:", error);
//             alert(`Failed to load explanations: ${error.message}`); // Keep original alert
//             setExplanations({ lime: null, gradcam: null, ig: null }); // Reset on error
//         } finally {
//             setIsLoading(false);
//             setLoadingExplanation(false);
//         }
//     };

//     const resetState = () => { // Reset state for single file
//         setStep(1);
//         setFile(null);
//         setFileName('');
//         if (uploadedImage) {
//             URL.revokeObjectURL(uploadedImage);
//         }
//         setUploadedImage(null);
//         setUploadProgress(0);
//         setFileType('');
//         setFileSize(0);
//         setFileTypeError(false);
//         setExplanations({ lime: null, gradcam: null, ig: null });
//         setLoadingExplanation(false);
//         setIsLoading(false);
//         setPredictionResult({ predicted_label: '', confidence: 0, actual_label: 'Unknown', all_confidences: {} });
//          const fileInput = document.querySelector('.file-input'); // Assumes input has this class
//          if (fileInput) {
//              fileInput.value = null;
//          }
//     };
//     // --- End Handlers ---

//     useEffect(() => {
//         // Cleanup function for Object URL
//         return () => {
//             if (uploadedImage) {
//                 URL.revokeObjectURL(uploadedImage);
//             }
//         };
//     }, [uploadedImage]); // Dependency ensures cleanup on change/unmount


//     const renderBarChart = (data) => {
//         if (!data || !data.all_confidences || Object.keys(data.all_confidences).length === 0) {
//             return <p>No confidence data available.</p>;
//         }

//         // Ensure confidence values are numbers
//         const chartData = Object.entries(data.all_confidences).map(([stage, confidence]) => ({
//             stage,
//             confidence: Number(confidence) || 0,
//         }));

//         return (
//             <ResponsiveContainer width="100%" height={300}>
//                 <BarChart data={chartData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
//                     <CartesianGrid strokeDasharray="3 3" />
//                     <XAxis dataKey="stage" />
//                     <YAxis tickFormatter={(value) => `${value}%`} domain={[0, 100]} />
//                     <Tooltip formatter={(value) => `${value.toFixed(2)}%`} />
//                     <Legend />
//                     <Bar dataKey="confidence" fill="#1e90ff" />
//                 </BarChart>
//             </ResponsiveContainer>
//         );
//     };


//     // --- Render Step Content - Kept exactly as in the first code block ---
//     const renderStepContent = () => {
//         switch (step) {
//             case 1:
//                 return (
//                     <>
//                         <div className="file-upload-container">
//                             <div className="file-input-container">
//                                 <input type="file" onChange={handleFileSelect} className="file-input" accept=".jpg,.jpeg,.png"/>
//                                 {fileTypeError && <p className="error-message">Invalid file type.</p>}
//                             </div>
//                             <button className="btn" onClick={handleUpload} disabled={!file || (uploadProgress > 0 && uploadProgress < 100)}>
//                                {uploadProgress === 100 ? "Uploaded ✓" : "Upload"}
//                             </button>
//                         </div>
//                         {file && !fileTypeError && (
//                             <div className="upload-info-container">
//                                 <div className="upload-info">
//                                     <span className="file-name">{fileName}</span>
//                                     <progress value={uploadProgress} max="100"></progress>
//                                     <span className="upload-percentage">{uploadProgress}%</span>
//                                 </div>
//                             </div>
//                         )}
//                         <div className="step-actions">
//                             <button className="btn btn-preview" onClick={handlePreview} disabled={!file || fileTypeError || uploadProgress !== 100}>Preview</button>
//                         </div>
//                     </>
//                 );
//             case 2:
//                 return (
//                     <>
//                         <div className="preview-container">
//                             {uploadedImage ? <img src={uploadedImage} alt="Preview" className="preview-image" /> : <p>No image uploaded.</p>}
//                             <div className="file-details">
//                                 <p><strong>File:</strong> {fileName}</p>
//                                 <p><strong>Type:</strong> {fileType}</p>
//                                 <p><strong>Size:</strong> {fileSize > 0 ? `${(fileSize / 1024).toFixed(2)} KB` : 'N/A'}</p>
//                             </div>
//                         </div>
//                         <div className="step-actions">
//                             <button className="btn btn-back" onClick={() => setStep(1)}>Back</button>
//                             <button className="btn btn-predict" onClick={handlePredict} disabled={isLoading}>Predict</button>
//                         </div>
//                     </>
//                 );
//             case 3:
//                 return (
//                     <>
//                         <div className="result-container" ref={resultRef}>
//                             <div className="result-top-section">
//                                 <div className="result-details">
//                                     <div className="detail-item"><span className="detail-label">File</span>: {fileName}</div>
//                                     <div className="detail-item"><span className="detail-label">Stage</span>: {predictionResult.predicted_label || 'N/A'}</div>
//                                     <div className="detail-item"><span className="detail-label">Confidence</span>: {predictionResult.confidence?.toFixed(2) ?? '0.00'}%</div>
//                                 </div>
//                                 {uploadedImage && <img src={uploadedImage} alt="Result" className="result-image" />}
//                             </div>
//                             <div className="confidence-bars">
//                                 <h3 className="confidence-bar-heading">Results Summary</h3>
//                                 {renderBarChart(predictionResult)}
//                             </div>
//                         </div>
//                         <div className="step-actions">
//                             <button className="btn btn-back" onClick={resetState}>New Scan</button>
//                             <button className="btn btn-interpretations" onClick={handleVisualInterpretations} disabled={isLoading || !predictionResult.predicted_label}>Visual Interpretations</button>
//                         </div>
//                     </>
//                 );
//             case 4:
//                  const hasAnyExplanation = Object.values(explanations).some(method => method && Object.keys(method).length > 0);
//                 return (
//                     <>
//                         <div className="interpretations-container" ref={interpretationsRef}>
//                             {loadingExplanation ? (
//                                 <div className="loading-container">
//                                     <div className="spinner"></div>
//                                     <p>Loading Explanations...</p>
//                                 </div>
//                              ) : !hasAnyExplanation ? (
//                                  <p className="placeholder-text">No explanation data available.</p>
//                              ) : (
//                                 <div className="models-row">
//                                     {/* Use backend model keys (e.g., densenet, resnet) */}
//                                     {Object.keys(explanations.lime || explanations.gradcam || explanations.ig || {}).map((modelKey) => (
//                                          // Check if data exists for this model before rendering the column
//                                          (explanations.lime?.[modelKey] || explanations.gradcam?.[modelKey] || explanations.ig?.[modelKey]) && (
//                                             <div className="model-column" key={modelKey}>
//                                                 <h2>Model: {modelKey.charAt(0).toUpperCase() + modelKey.slice(1)}</h2>
//                                                 {explanations.lime?.[modelKey] && (
//                                                      <div className="explanation-item">
//                                                         <h4>LIME</h4>
//                                                         <img src={`data:image/png;base64,${explanations.lime[modelKey]}`} alt={`${modelKey} LIME`} className="interpretation-image" />
//                                                     </div>
//                                                 )}
//                                                 {explanations.gradcam?.[modelKey] && (
//                                                      <div className="explanation-item">
//                                                         <h4>Grad-CAM</h4>
//                                                         <img src={`data:image/png;base64,${explanations.gradcam[modelKey]}`} alt={`${modelKey} GradCAM`} className="interpretation-image" />
//                                                     </div>
//                                                 )}
//                                                 {explanations.ig?.[modelKey] && (
//                                                      <div className="explanation-item">
//                                                         <h4>Integrated Gradients</h4>
//                                                         <img src={`data:image/png;base64,${explanations.ig[modelKey]}`} alt={`${modelKey} IG`} className="interpretation-image" />
//                                                     </div>
//                                                 )}
//                                             </div>
//                                         )
//                                     ))}
//                                 </div>
//                             )}
//                         </div>
//                         {/* --- Step 4 Actions - Kept exactly as in the first code block --- */}
//                         <div className="step-actions">
//                              {/* Button 1: New Scan (Aligned Left) */}
//                             <button className="btn btn-back" onClick={resetState}>New Scan</button>

//                              {/* Button 2: Download (Centered) */}
//                             <button
//                                 className="btn btn-secondary btn-download" // Assuming this class exists or you style it
//                                 onClick={handleDownloadVisualInterpretations} // This is the updated function
//                                 disabled={isLoading || loadingExplanation || !hasAnyExplanation}
//                             >
//                                 Download Interpretations
//                             </button>

//                              {/* Button 3: Go Back (Aligned Right) */}
//                             <button className="btn btn-preview" onClick={() => setStep(3)}>Go Back</button>
//                         </div>
//                          {/* --- END: Step 4 Actions --- */}
//                     </>
//                 );
//             default:
//                 return <p>An error occurred. Please refresh.</p>;
//         }
//     };

//     // --- Main Component Return - Kept exactly as in the first code block ---
//     return (
//         <div className="predict-container">
//             <Navbar />
//             <div className="steps-container">
//                 <div className="step-indicator-container">
//                     <div className={`step ${step === 1 ? 'active' : ''}`}>1. Upload MRI</div>
//                     <div className={`step ${step === 2 ? 'active' : ''}`}>2. Preview MRI</div>
//                     <div className={`step ${step === 3 ? 'active' : ''}`}>3. Detection Result</div>
//                     <div className={`step ${step === 4 ? 'active' : ''}`}>4. Visual Interpretations</div>
//                 </div>
//                 <div className="step-content">
//                     {renderStepContent()}
//                 </div>
//             </div>

//             {(isLoading || loadingExplanation) && (
//                 <div className="loading-screen">
//                     <div className="loading-content">
//                         <div className="spinner"></div>
//                         <p>Loading...</p>
//                     </div>
//                 </div>
//             )}
//         </div>
//     );
// }

// export default PredictPage;

// import React, { useState, useRef, useEffect } from 'react';
// import './PredictPage.css'; // Ensure this CSS file exists and is styled
// import Navbar from '../Navbar/Navbar'; // Ensure this path is correct
// import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
// import jsPDF from 'jspdf';
// import html2canvas from 'html2canvas';

// function PredictPage() {
//     const backendBaseUrl = 'http://127.0.0.1:5000';
//     const [step, setStep] = useState(1);
//     const [file, setFile] = useState(null);
//     const [fileName, setFileName] = useState('');
//     const [uploadedImage, setUploadedImage] = useState(null);
//     const [uploadProgress, setUploadProgress] = useState(0);
//     const [fileType, setFileType] = useState('');
//     const [fileSize, setFileSize] = useState(0);
//     const [fileTypeError, setFileTypeError] = useState(false);
//     const [loadingExplanation, setLoadingExplanation] = useState(false);
//     const [explanations, setExplanations] = useState({ lime: null, gradcam: null, ig: null });
//     const [isLoading, setIsLoading] = useState(false);
//     const [predictionResult, setPredictionResult] = useState({
//         predicted_label: '',
//         confidence: 0,
//         actual_label: 'Unknown',
//         all_confidences: {},
//     });

//     const resultRef = useRef(null);
//     const interpretationsRef = useRef(null);
//     const fileInputRef = useRef(null); // Ref for the file input element

//     // --- START: Updated handleDownloadVisualInterpretations ---
//     const handleDownloadVisualInterpretations = async () => {
//         const elementToCapture = interpretationsRef.current;
//         if (!elementToCapture) {
//             alert("Could not find the interpretations content to download.");
//             console.error("interpretationsRef is not attached or the component is not rendered.");
//             return;
//         }

//         const hasExplanations = explanations.lime || explanations.gradcam || explanations.ig;
//          // Check if explanations have actual data (e.g., at least one model has one type)
//          const hasExplanationData = hasExplanations && Object.values(hasExplanations).some(method => method && Object.keys(method).length > 0);
//          if (!hasExplanationData) {
//              alert("No visual interpretations are available to download.");
//              return;
//          }

//          // Check if prediction result is available
//          if (!predictionResult.predicted_label) {
//             alert("Prediction results are not available to include in the PDF.");
//             // Optionally proceed without prediction info or stop here
//             // return; // You might uncomment this if prediction data is mandatory
//          }

//         setIsLoading(true); // Use general loading state to indicate activity
//         try {
//             const canvas = await html2canvas(elementToCapture, {
//                  useCORS: true, // Handle base64 images correctly
//                  scale: 2, // Improve resolution
//                  logging: false,
//                  scrollX: 0,
//                  scrollY: -window.scrollY, // Account for page scroll if element isn't fully visible
//                  windowWidth: elementToCapture.scrollWidth,
//                  windowHeight: elementToCapture.scrollHeight
//             });

//             const imgData = canvas.toDataURL('image/png');
//             const imgWidth = canvas.width;
//             const imgHeight = canvas.height;

//             // A4 dimensions in points: 595.28 x 841.89 pt
//             const pdfWidth = 595.28;
//             const pdfHeight = 841.89;

//             const pdf = new jsPDF({
//                  orientation: imgWidth > imgHeight ? 'landscape' : 'portrait',
//                  unit: 'pt',
//                  format: 'a4'
//             });

//             // --- START: Add Prediction Info Text to PDF ---
//             let currentY = 30; // Initial Y position for text

//             pdf.setFontSize(16);
//             pdf.setFont(undefined, 'bold'); // Use bold for title
//             pdf.text('Visual Interpretations Report', pdfWidth / 2, currentY, { align: 'center' });
//             currentY += 20; // Increase Y position

//             pdf.setFontSize(11);
//             pdf.setFont(undefined, 'normal'); // Reset font style
//             pdf.text(`Source File: ${fileName || 'N/A'}`, pdfWidth / 2, currentY, { align: 'center' });
//             currentY += 25; // Increase Y position

//             // Add Predicted Case and Confidence
//             if (predictionResult.predicted_label) {
//                 pdf.setFont(undefined, 'bold');
//                 pdf.text('Prediction Result:', 40, currentY); // Use a fixed X margin (e.g., 40pt)
//                 currentY += 15;

//                 pdf.setFont(undefined, 'normal');
//                 pdf.text(`- Predicted Stage: ${predictionResult.predicted_label}`, 55, currentY); // Indent slightly
//                 currentY += 15;
//                 pdf.text(`- Confidence Score: ${predictionResult.confidence?.toFixed(2) ?? 'N/A'}%`, 55, currentY);
//                 currentY += 25; // Add more space before the image section title
//             } else {
//                  // Optionally add a placeholder if prediction data isn't available but you still proceed
//                  pdf.setFont(undefined, 'italic');
//                  pdf.text('Prediction results were not available for this report.', 40, currentY);
//                  currentY += 25;
//             }

//             pdf.setFontSize(14);
//             pdf.setFont(undefined, 'bold');
//             pdf.text('Explanation Visualizations:', pdfWidth / 2, currentY, { align: 'center' });
//             currentY += 20; // Space after the section title, before the image
//             // --- END: Add Prediction Info Text to PDF ---


//             // Calculate the aspect ratio to fit the image on the page below the text
//             // Adjust available height for the image
//             const availableHeight = pdfHeight - currentY - 20; // Subtract current Y and bottom margin
//             // Adjust available width slightly for margins (e.g., 40pt each side)
//             const ratio = Math.min((pdfWidth - 80) / imgWidth, availableHeight / imgHeight);
//             const pdfImageWidth = imgWidth * ratio;
//             const pdfImageHeight = imgHeight * ratio;

//             // Center the image (optional) below the text
//             const xOffset = (pdfWidth - pdfImageWidth) / 2;
//             const yOffset = currentY; // Start image after the text

//             // Ensure image doesn't overflow page (redundant check if ratio is calculated correctly)
//             if (yOffset + pdfImageHeight > pdfHeight - 20) {
//                 console.warn("Calculated image height might exceed page limits after adding text.");
//                 // Optional: further reduce scale or handle multi-page
//             }

//             pdf.addImage(imgData, 'PNG', xOffset, yOffset, pdfImageWidth, pdfImageHeight);

//             pdf.save(`visual_interpretations_${fileName.split('.')[0] || 'report'}.pdf`);

//         } catch (error) {
//             console.error("Error generating PDF:", error);
//             alert("Failed to generate PDF for visual interpretations. See console for details.");
//         } finally {
//             setIsLoading(false); // Stop loading indicator
//         }
//     };
//     // --- END: Updated handleDownloadVisualInterpretations ---


//     const handleFileSelect = (event) => {
//         const selectedFile = event.target.files[0];
//         if (selectedFile) {
//             const isValidFileType = /\.(jpg|jpeg|png)$/i.test(selectedFile.name);
//             if (!isValidFileType) {
//                 setFileTypeError(true);
//                 alert('Only .jpg, .jpeg, and .png files are allowed.');
//                 event.target.value = null; // Clear invalid input
//                 // Reset relevant states if an invalid file was attempted
//                 setFile(null);
//                 setFileName('');
//                 if (uploadedImage) URL.revokeObjectURL(uploadedImage);
//                 setUploadedImage(null);
//                 setUploadProgress(0);
//                 setFileType('');
//                 setFileSize(0);
//                 return;
//             }
//             if (uploadedImage) {
//                 URL.revokeObjectURL(uploadedImage); // Clean previous URL
//             }
//             setFileTypeError(false);
//             setFile(selectedFile);
//             setFileName(selectedFile.name);
//             setFileType(selectedFile.type);
//             setFileSize(selectedFile.size);
//             setUploadProgress(0); // Reset progress for new file
//             setUploadedImage(URL.createObjectURL(selectedFile));
//         } else {
//              // Handle case where user cancels the file selection dialog
//              handleCancelUpload(); // Use the cancel handler to reset
//         }
//     };

//     // --- NEW: Handler for the Cancel button in Step 1 ---
//     const handleCancelUpload = () => {
//         setFile(null);
//         setFileName('');
//         if (uploadedImage) {
//             URL.revokeObjectURL(uploadedImage);
//         }
//         setUploadedImage(null);
//         setUploadProgress(0);
//         setFileType('');
//         setFileSize(0);
//         setFileTypeError(false);
//         // Reset the actual file input element
//         if (fileInputRef.current) {
//             fileInputRef.current.value = null;
//         }
//     };
//     // --- END: New Cancel Handler ---

//     const handleUpload = () => { // Simulate upload progress
//         if (file) {
//             setUploadProgress(0); // Start progress from 0
//             const interval = setInterval(() => {
//                 setUploadProgress(oldProgress => {
//                     // Prevent progress update if file was cancelled during upload
//                     if (!file) {
//                         clearInterval(interval);
//                         return 0; // Reset progress if file is gone
//                     }
//                     const newProgress = Math.min(oldProgress + 10, 100);
//                     if (newProgress === 100) {
//                         clearInterval(interval);
//                     }
//                     return newProgress;
//                 });
//             }, 100);
//         }
//     };

//     const handlePreview = () => {
//         // Proceed only if file exists and upload simulation reached 100%
//         if (file && uploadProgress === 100) setStep(2);
//         else if (!file) alert("Please upload a file.");
//         else alert("Please complete the upload indication first.");
//     };

//     const handlePredict = async () => { // Predict using single file
//         if (!file) return;
//         setIsLoading(true);
//         try {
//             const formData = new FormData();
//             formData.append('file', file); // Use 'file' key for backend

//             const response = await fetch(`${backendBaseUrl}/predict_mri`, { // Use single MRI endpoint
//                 method: 'POST',
//                 body: formData,
//             });
//              if (!response.ok) {
//                 throw new Error(`HTTP error! status: ${response.status}`);
//             }
//             const data = await response.json();
//             setPredictionResult({
//                 predicted_label: data.predicted_label || 'N/A',
//                 confidence: data.confidence || 0,
//                 actual_label: data.actual_label || 'Unknown',
//                 all_confidences: data.all_confidences || {},
//             });
//             setStep(3);
//         } catch (error) {
//             console.error("Prediction failed:", error);
//             alert(`Failed to get prediction: ${error.message}`);
//         } finally {
//             setIsLoading(false);
//         }
//     };

//     const handleVisualInterpretations = async () => { // Explain single file
//          if (!file) {
//              alert("Cannot request interpretations without an uploaded image context.");
//              return;
//          }
//          if (!predictionResult.predicted_label) {
//              alert("Please run the prediction first to get context for interpretations.");
//              return;
//          }
//         setIsLoading(true);
//         setLoadingExplanation(true);
//         try {
//             const response = await fetch(`${backendBaseUrl}/explain_mri`);
//              if (!response.ok) {
//                 throw new Error(`HTTP error! status: ${response.status}`);
//             }
//             const data = await response.json();

//              const hasData = data && (data.lime || data.gradcam || data.ig);
//              const hasContent = hasData && (Object.keys(data.lime || {}).length > 0 || Object.keys(data.gradcam || {}).length > 0 || Object.keys(data.ig || {}).length > 0);

//              if (!hasContent) {
//                  console.warn("Received empty or invalid explanation data:", data);
//                  alert("Explanations received from the server were empty or unreadable.");
//                  setExplanations({ lime: null, gradcam: null, ig: null });
//              } else {
//                   setExplanations({
//                       lime: data.lime || null,
//                       gradcam: data.gradcam || null,
//                       ig: data.ig || null,
//                   });
//                   setStep(4);
//              }
//         } catch (error) {
//             console.error("Explanation fetch failed:", error);
//             alert(`Failed to load explanations: ${error.message}`);
//             setExplanations({ lime: null, gradcam: null, ig: null });
//         } finally {
//             setIsLoading(false);
//             setLoadingExplanation(false);
//         }
//     };

//     const resetState = () => { // Reset state for single file
//         setStep(1);
//         // Use the cancel handler logic to reset file-related state
//         handleCancelUpload();
//         // Reset states not covered by handleCancelUpload
//         setExplanations({ lime: null, gradcam: null, ig: null });
//         setLoadingExplanation(false);
//         setIsLoading(false);
//         setPredictionResult({ predicted_label: '', confidence: 0, actual_label: 'Unknown', all_confidences: {} });
//     };
//     // --- End Handlers ---

//     useEffect(() => {
//         // Cleanup function for Object URL
//         return () => {
//             if (uploadedImage) {
//                 URL.revokeObjectURL(uploadedImage);
//             }
//         };
//     }, [uploadedImage]); // Dependency ensures cleanup on change/unmount


//     const renderBarChart = (data) => {
//         if (!data || !data.all_confidences || Object.keys(data.all_confidences).length === 0) {
//             return <p>No confidence data available.</p>;
//         }

//         // Ensure confidence values are numbers
//         const chartData = Object.entries(data.all_confidences).map(([stage, confidence]) => ({
//             stage,
//             confidence: Number(confidence) || 0,
//         }));

//         return (
//             <ResponsiveContainer width="100%" height={300}>
//                 <BarChart data={chartData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
//                     <CartesianGrid strokeDasharray="3 3" />
//                     <XAxis dataKey="stage" />
//                     <YAxis tickFormatter={(value) => `${value}%`} domain={[0, 100]} />
//                     <Tooltip formatter={(value) => `${value.toFixed(2)}%`} />
//                     <Legend />
//                     <Bar dataKey="confidence" fill="#1e90ff" />
//                 </BarChart>
//             </ResponsiveContainer>
//         );
//     };


//     const renderStepContent = () => {
//         switch (step) {
//             case 1:
//                 return (
//                     <>
//                         <div className="file-upload-container">
//                             <div className="file-input-container">
//                                 {/* Added ref to the input */}
//                                 <input type="file" onChange={handleFileSelect} className="file-input" accept=".jpg,.jpeg,.png" ref={fileInputRef}/>
//                                 {fileTypeError && <p className="error-message">Invalid file type.</p>}
//                             </div>
//                             {/* Conditionally render Upload button only if file is selected */}
//                              {file && !fileTypeError && (
//                                 <button className="btn" onClick={handleUpload} disabled={uploadProgress > 0 && uploadProgress < 100}>
//                                     {uploadProgress === 100 ? "Uploaded ✓" : "Upload"}
//                                 </button>
//                             )}
//                         </div>
//                         {file && !fileTypeError && (
//                             <div className="upload-info-container">
//                                 <div className="upload-info">
//                                     <span className="file-name">{fileName}</span>
//                                     <progress value={uploadProgress} max="100"></progress>
//                                     <span className="upload-percentage">{uploadProgress}%</span>
//                                 </div>
//                             </div>
//                         )}
//                         {/* --- Updated Step 1 Actions --- */}
//                         {/* Show actions only when a file is selected and valid */}
//                         {file && !fileTypeError && (
//                              <div className="step-actions step-actions-step1"> {/* Added specific class for styling */}
//                                 {/* Cancel Button (Left) */}
//                                 <button className="btn btn-cancel" onClick={handleCancelUpload} disabled={isLoading}> {/* Ensure isLoading disables it too */}
//                                     Cancel
//                                 </button>
//                                 {/* Preview Button (Right) */}
//                                 <button className="btn btn-preview" onClick={handlePreview} disabled={uploadProgress !== 100 || isLoading}>
//                                     Preview
//                                 </button>
//                             </div>
//                         )}
//                         {/* --- End Updated Step 1 Actions --- */}
//                     </>
//                 );
//             case 2:
//                 return (
//                     <>
//                         <div className="preview-container">
//                             {uploadedImage ? <img src={uploadedImage} alt="Preview" className="preview-image" /> : <p>No image uploaded.</p>}
//                             <div className="file-details">
//                                 <p><strong>File:</strong> {fileName}</p>
//                                 <p><strong>Type:</strong> {fileType}</p>
//                                 <p><strong>Size:</strong> {fileSize > 0 ? `${(fileSize / 1024).toFixed(2)} KB` : 'N/A'}</p>
//                             </div>
//                         </div>
//                         <div className="step-actions">
//                             <button className="btn btn-back" onClick={() => setStep(1)}>Back</button>
//                             <button className="btn btn-predict" onClick={handlePredict} disabled={isLoading}>Predict</button>
//                         </div>
//                     </>
//                 );
//             case 3:
//                 return (
//                     <>
//                         <div className="result-container" ref={resultRef}>
//                             <div className="result-top-section">
//                                 <div className="result-details">
//                                     <div className="detail-item"><span className="detail-label">File</span>: {fileName}</div>
//                                     <div className="detail-item"><span className="detail-label">Stage</span>: {predictionResult.predicted_label || 'N/A'}</div>
//                                     <div className="detail-item"><span className="detail-label">Confidence</span>: {predictionResult.confidence?.toFixed(2) ?? '0.00'}%</div>
//                                 </div>
//                                 {uploadedImage && <img src={uploadedImage} alt="Result" className="result-image" />}
//                             </div>
//                             <div className="confidence-bars">
//                                 <h3 className="confidence-bar-heading">Results Summary</h3>
//                                 {renderBarChart(predictionResult)}
//                             </div>
//                         </div>
//                         <div className="step-actions">
//                             <button className="btn btn-back" onClick={resetState}>New Scan</button>
//                             <button className="btn btn-interpretations" onClick={handleVisualInterpretations} disabled={isLoading || !predictionResult.predicted_label}>Visual Interpretations</button>
//                         </div>
//                     </>
//                 );
//             case 4:
//                  const hasAnyExplanation = Object.values(explanations).some(method => method && Object.keys(method).length > 0);
//                 return (
//                     <>
//                         <div className="interpretations-container" ref={interpretationsRef}>
//                             {loadingExplanation ? (
//                                 <div className="loading-container">
//                                     <div className="spinner"></div>
//                                     <p>Loading Explanations...</p>
//                                 </div>
//                              ) : !hasAnyExplanation ? (
//                                  <p className="placeholder-text">No explanation data available.</p>
//                              ) : (
//                                 <div className="models-row">
//                                     {Object.keys(explanations.lime || explanations.gradcam || explanations.ig || {}).map((modelKey) => (
//                                          (explanations.lime?.[modelKey] || explanations.gradcam?.[modelKey] || explanations.ig?.[modelKey]) && (
//                                             <div className="model-column" key={modelKey}>
//                                                 <h2>Model: {modelKey.charAt(0).toUpperCase() + modelKey.slice(1)}</h2>
//                                                 {explanations.lime?.[modelKey] && (
//                                                      <div className="explanation-item">
//                                                         <h4>LIME</h4>
//                                                         <img src={`data:image/png;base64,${explanations.lime[modelKey]}`} alt={`${modelKey} LIME`} className="interpretation-image" />
//                                                     </div>
//                                                 )}
//                                                 {explanations.gradcam?.[modelKey] && (
//                                                      <div className="explanation-item">
//                                                         <h4>Grad-CAM</h4>
//                                                         <img src={`data:image/png;base64,${explanations.gradcam[modelKey]}`} alt={`${modelKey} GradCAM`} className="interpretation-image" />
//                                                     </div>
//                                                 )}
//                                                 {explanations.ig?.[modelKey] && (
//                                                      <div className="explanation-item">
//                                                         <h4>Integrated Gradients</h4>
//                                                         <img src={`data:image/png;base64,${explanations.ig[modelKey]}`} alt={`${modelKey} IG`} className="interpretation-image" />
//                                                     </div>
//                                                 )}
//                                             </div>
//                                         )
//                                     ))}
//                                 </div>
//                             )}
//                         </div>
//                         {/* --- Step 4 Actions --- */}
//                         <div className="step-actions step-actions-step4"> {/* Added specific class for styling */}
//                              {/* Button 1: New Scan (Aligned Left) */}
//                             <button className="btn btn-back" onClick={resetState}>New Scan</button>

//                              {/* Button 2: Download (Centered) */}
//                             <button
//                                 className="btn btn-secondary btn-download"
//                                 onClick={handleDownloadVisualInterpretations}
//                                 disabled={isLoading || loadingExplanation || !hasAnyExplanation}
//                             >
//                                 Download Interpretations
//                             </button>

//                              {/* Button 3: Go Back (Aligned Right) */}
//                             <button className="btn btn-preview" onClick={() => setStep(3)}>Go Back</button>
//                         </div>
//                          {/* --- END: Step 4 Actions --- */}
//                     </>
//                 );
//             default:
//                 return <p>An error occurred. Please refresh.</p>;
//         }
//     };

//     // --- Main Component Return ---
//     return (
//         <div className="predict-container">
//             <Navbar />
//             <div className="steps-container">
//                 <div className="step-indicator-container">
//                     <div className={`step ${step === 1 ? 'active' : ''}`}>1. Upload MRI</div>
//                     <div className={`step ${step === 2 ? 'active' : ''}`}>2. Preview MRI</div>
//                     <div className={`step ${step === 3 ? 'active' : ''}`}>3. Detection Result</div>
//                     <div className={`step ${step === 4 ? 'active' : ''}`}>4. Visual Interpretations</div>
//                 </div>
//                 <div className="step-content">
//                     {renderStepContent()}
//                 </div>
//             </div>

//             {(isLoading || loadingExplanation) && (
//                 <div className="loading-screen">
//                     <div className="loading-content">
//                         <div className="spinner"></div>
//                         <p>Loading...</p>
//                     </div>
//                 </div>
//             )}
//         </div>
//     );
// }

// export default PredictPage;


import React, { useState, useRef, useEffect } from 'react';
import './PredictPage.css'; // Ensure this CSS file exists and is styled
import Navbar from '../Navbar/Navbar'; // Ensure this path is correct
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import jsPDF from 'jspdf';
import html2canvas from 'html2canvas';

function PredictPage() {
    const backendBaseUrl = 'http://127.0.0.1:5000';
    const [step, setStep] = useState(1);
    const [file, setFile] = useState(null);
    const [fileName, setFileName] = useState('');
    const [uploadedImage, setUploadedImage] = useState(null);
    const [uploadProgress, setUploadProgress] = useState(0);
    const [fileType, setFileType] = useState('');
    const [fileSize, setFileSize] = useState(0);
    const [fileTypeError, setFileTypeError] = useState(false);
    const [loadingExplanation, setLoadingExplanation] = useState(false);
    const [explanations, setExplanations] = useState({ lime: null, gradcam: null, ig: null });
    const [isLoading, setIsLoading] = useState(false);
    const [predictionResult, setPredictionResult] = useState({
        predicted_label: '',
        confidence: 0,
        actual_label: 'Unknown',
        all_confidences: {},
    });

    const resultRef = useRef(null);
    const interpretationsRef = useRef(null);
    const fileInputRef = useRef(null); // Ref for the file input element

    // --- START: Updated handleDownloadVisualInterpretations ---
    const handleDownloadVisualInterpretations = async () => {
        const elementToCapture = interpretationsRef.current;
        if (!elementToCapture) {
            alert("Could not find the interpretations content to download.");
            console.error("interpretationsRef is not attached or the component is not rendered.");
            return;
        }

        const hasExplanations = explanations.lime || explanations.gradcam || explanations.ig;
         // Check if explanations have actual data (e.g., at least one model has one type)
         const hasExplanationData = hasExplanations && Object.values(hasExplanations).some(method => method && Object.keys(method).length > 0);
         if (!hasExplanationData) {
             alert("No visual interpretations are available to download.");
             return;
         }

         // Check if prediction result is available
         if (!predictionResult.predicted_label) {
            alert("Prediction results are not available to include in the PDF.");
            // Optionally proceed without prediction info or stop here
            // return; // You might uncomment this if prediction data is mandatory
         }

        setIsLoading(true); // Use general loading state to indicate activity
        try {
            const canvas = await html2canvas(elementToCapture, {
                 useCORS: true, // Handle base64 images correctly
                 scale: 2, // Improve resolution
                 logging: false,
                 scrollX: 0,
                 scrollY: -window.scrollY, // Account for page scroll if element isn't fully visible
                 windowWidth: elementToCapture.scrollWidth,
                 windowHeight: elementToCapture.scrollHeight
            });

            const imgData = canvas.toDataURL('image/png');
            const imgWidth = canvas.width;
            const imgHeight = canvas.height;

            // A4 dimensions in points: 595.28 x 841.89 pt
            const pdfWidth = 595.28;
            const pdfHeight = 841.89;

            const pdf = new jsPDF({
                 orientation: imgWidth > imgHeight ? 'landscape' : 'portrait',
                 unit: 'pt',
                 format: 'a4'
            });

            // --- START: Add Prediction Info Text to PDF ---
            let currentY = 30; // Initial Y position for text

            pdf.setFontSize(16);
            pdf.setFont(undefined, 'bold'); // Use bold for title
            pdf.text('Visual Interpretations Report', pdfWidth / 2, currentY, { align: 'center' });
            currentY += 20; // Increase Y position

            pdf.setFontSize(11);
            pdf.setFont(undefined, 'normal'); // Reset font style
            pdf.text(`Source File: ${fileName || 'N/A'}`, pdfWidth / 2, currentY, { align: 'center' });
            currentY += 25; // Increase Y position

            // Add Predicted Case and Confidence
            if (predictionResult.predicted_label) {
                pdf.setFont(undefined, 'bold');
                pdf.text('Prediction Result:', 40, currentY); // Use a fixed X margin (e.g., 40pt)
                currentY += 15;

                pdf.setFont(undefined, 'normal');
                pdf.text(`- Predicted Stage: ${predictionResult.predicted_label}`, 55, currentY); // Indent slightly
                currentY += 15;
                pdf.text(`- Confidence Score: ${predictionResult.confidence?.toFixed(2) ?? 'N/A'}%`, 55, currentY);
                currentY += 25; // Add more space before the image section title
            } else {
                 // Optionally add a placeholder if prediction data isn't available but you still proceed
                 pdf.setFont(undefined, 'italic');
                 pdf.text('Prediction results were not available for this report.', 40, currentY);
                 currentY += 25;
            }

            pdf.setFontSize(14);
            pdf.setFont(undefined, 'bold');
            pdf.text('Explanation Visualizations:', pdfWidth / 2, currentY, { align: 'center' });
            currentY += 20; // Space after the section title, before the image
            // --- END: Add Prediction Info Text to PDF ---


            // Calculate the aspect ratio to fit the image on the page below the text
            // Adjust available height for the image
            const availableHeight = pdfHeight - currentY - 20; // Subtract current Y and bottom margin
            // Adjust available width slightly for margins (e.g., 40pt each side)
            const ratio = Math.min((pdfWidth - 80) / imgWidth, availableHeight / imgHeight);
            const pdfImageWidth = imgWidth * ratio;
            const pdfImageHeight = imgHeight * ratio;

            // Center the image (optional) below the text
            const xOffset = (pdfWidth - pdfImageWidth) / 2;
            const yOffset = currentY; // Start image after the text

            // Ensure image doesn't overflow page (redundant check if ratio is calculated correctly)
            if (yOffset + pdfImageHeight > pdfHeight - 20) {
                console.warn("Calculated image height might exceed page limits after adding text.");
                // Optional: further reduce scale or handle multi-page
            }

            pdf.addImage(imgData, 'PNG', xOffset, yOffset, pdfImageWidth, pdfImageHeight);

            pdf.save(`visual_interpretations_${fileName.split('.')[0] || 'report'}.pdf`);

        } catch (error) {
            console.error("Error generating PDF:", error);
            alert("Failed to generate PDF for visual interpretations. See console for details.");
        } finally {
            setIsLoading(false); // Stop loading indicator
        }
    };
    // --- END: Updated handleDownloadVisualInterpretations ---


    const handleFileSelect = async (event) => {
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
    
            // --- Start: Validate MRI Image using Backend Validator ---
            try {
                const formData = new FormData();
                formData.append('file', selectedFile);
    
                const response = await fetch('http://127.0.0.1:5000/validate_mri', {
                    method: 'POST',
                    body: formData,
                });
    
                if (!response.ok) {
                    throw new Error(`HTTP error! Status: ${response.status}`);
                }
    
                const result = await response.json();
    
                if (!result.is_valid) {
                    alert(' Uploaded image is not recognised as a valid MRI scan.');
                    event.target.value = null;
                    resetFileStates();
                    return;
                }
            } catch (error) {
                console.error("Validation error:", error);
                alert('Failed to validate the MRI scan. Please try again.');
                event.target.value = null;
                resetFileStates();
                return;
            }
            // --- End MRI Validation ---
    
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

    const handleUpload = () => { // Simulate upload progress
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

    const handlePredict = async () => { // Predict using single file
        if (!file) return;
        setIsLoading(true);
        try {
            const formData = new FormData();
            formData.append('file', file); // Use 'file' key for backend

            const response = await fetch(`${backendBaseUrl}/predict_mri`, { // Use single MRI endpoint
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

    const handleVisualInterpretations = async () => { // Explain single file
         if (!file) {
             alert("Cannot request interpretations without an uploaded image context.");
             return;
         }
         if (!predictionResult.predicted_label) {
             alert("Please run the prediction first to get context for interpretations.");
             return;
         }
        setIsLoading(true);
        setLoadingExplanation(true);
        try {
            const response = await fetch(`${backendBaseUrl}/explain_mri`);
             if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            const data = await response.json();

             const hasData = data && (data.lime || data.gradcam || data.ig);
             const hasContent = hasData && (Object.keys(data.lime || {}).length > 0 || Object.keys(data.gradcam || {}).length > 0 || Object.keys(data.ig || {}).length > 0);

             if (!hasContent) {
                 console.warn("Received empty or invalid explanation data:", data);
                 alert("Explanations received from the server were empty or unreadable.");
                 setExplanations({ lime: null, gradcam: null, ig: null });
             } else {
                  setExplanations({
                      lime: data.lime || null,
                      gradcam: data.gradcam || null,
                      ig: data.ig || null,
                  });
                  setStep(4);
             }
        } catch (error) {
            console.error("Explanation fetch failed:", error);
            alert(`Failed to load explanations: ${error.message}`);
            setExplanations({ lime: null, gradcam: null, ig: null });
        } finally {
            setIsLoading(false);
            setLoadingExplanation(false);
        }
    };

    const resetState = () => { // Reset state for single file
        setStep(1);
        // Use the cancel handler logic to reset file-related state
        handleCancelUpload();
        // Reset states not covered by handleCancelUpload
        setExplanations({ lime: null, gradcam: null, ig: null });
        setLoadingExplanation(false);
        setIsLoading(false);
        setPredictionResult({ predicted_label: '', confidence: 0, actual_label: 'Unknown', all_confidences: {} });
    };
    // --- End Handlers ---

    useEffect(() => {
        // Cleanup function for Object URL
        return () => {
            if (uploadedImage) {
                URL.revokeObjectURL(uploadedImage);
            }
        };
    }, [uploadedImage]); // Dependency ensures cleanup on change/unmount


    const renderBarChart = (data) => {
        if (!data || !data.all_confidences || Object.keys(data.all_confidences).length === 0) {
            return <p>No confidence data available.</p>;
        }

        // Ensure confidence values are numbers
        const chartData = Object.entries(data.all_confidences).map(([stage, confidence]) => ({
            stage,
            confidence: Number(confidence) || 0,
        }));

        return (
            <ResponsiveContainer width="100%" height={300}>
                <BarChart data={chartData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="stage" />
                    <YAxis tickFormatter={(value) => `${value}%`} domain={[0, 100]} />
                    <Tooltip formatter={(value) => `${value.toFixed(2)}%`} />
                    <Legend />
                    <Bar dataKey="confidence" fill="#1e90ff" />
                </BarChart>
            </ResponsiveContainer>
        );
    };


    const renderStepContent = () => {
        switch (step) {
            case 1:
                return (
                    <>
                        <div className="file-upload-container">
                            <div className="file-input-container">
                                {/* Added ref to the input */}
                                <input type="file" onChange={handleFileSelect} className="file-input" accept=".jpg,.jpeg,.png" ref={fileInputRef}/>
                                {fileTypeError && <p className="error-message">Invalid file type.</p>}
                            </div>
                            {/* Conditionally render Upload button only if file is selected */}
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
                             <div className="step-actions step-actions-step1"> {/* Added specific class for styling */}
                                {/* Cancel Button (Left) */}
                                <button className="btn btn-cancel" onClick={handleCancelUpload} disabled={isLoading}> {/* Ensure isLoading disables it too */}
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
                        <div className="result-container" ref={resultRef}>
                            <div className="result-top-section">
                                <div className="result-details">
                                    <div className="detail-item"><span className="detail-label">File</span>: {fileName}</div>
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
                            <button className="btn btn-interpretations" onClick={handleVisualInterpretations} disabled={isLoading || !predictionResult.predicted_label}>Visual Interpretations</button>
                        </div>
                    </>
                );
            case 4:
                 const hasAnyExplanation = Object.values(explanations).some(method => method && Object.keys(method).length > 0);
                return (
                    <>
                        <div className="interpretations-container" ref={interpretationsRef}>
                            {loadingExplanation ? (
                                <div className="loading-container">
                                    <div className="spinner"></div>
                                    <p>Loading Explanations...</p>
                                </div>
                             ) : !hasAnyExplanation ? (
                                 <p className="placeholder-text">No explanation data available.</p>
                             ) : (
                                <div className="models-row">
                                    {Object.keys(explanations.lime || explanations.gradcam || explanations.ig || {}).map((modelKey) => (
                                         (explanations.lime?.[modelKey] || explanations.gradcam?.[modelKey] || explanations.ig?.[modelKey]) && (
                                            <div className="model-column" key={modelKey}>
                                                <h2>Model: {modelKey.charAt(0).toUpperCase() + modelKey.slice(1)}</h2>
                                                {explanations.lime?.[modelKey] && (
                                                     <div className="explanation-item">
                                                        <h4>LIME</h4>
                                                        <img src={`data:image/png;base64,${explanations.lime[modelKey]}`} alt={`${modelKey} LIME`} className="interpretation-image" />
                                                    </div>
                                                )}
                                                {explanations.gradcam?.[modelKey] && (
                                                     <div className="explanation-item">
                                                        <h4>Grad-CAM</h4>
                                                        <img src={`data:image/png;base64,${explanations.gradcam[modelKey]}`} alt={`${modelKey} GradCAM`} className="interpretation-image" />
                                                    </div>
                                                )}
                                                {explanations.ig?.[modelKey] && (
                                                     <div className="explanation-item">
                                                        <h4>Integrated Gradients</h4>
                                                        <img src={`data:image/png;base64,${explanations.ig[modelKey]}`} alt={`${modelKey} IG`} className="interpretation-image" />
                                                    </div>
                                                )}
                                            </div>
                                        )
                                    ))}
                                </div>
                            )}
                        </div>
                        {/* --- Step 4 Actions --- */}
                        <div className="step-actions step-actions-step4"> {/* Added specific class for styling */}
                             {/* Button 1: New Scan (Aligned Left) */}
                            <button className="btn btn-back" onClick={resetState}>New Scan</button>

                             {/* Button 2: Download (Centered) */}
                            <button
                                className="btn btn-secondary btn-download"
                                onClick={handleDownloadVisualInterpretations}
                                disabled={isLoading || loadingExplanation || !hasAnyExplanation}
                            >
                                Download Interpretations
                            </button>

                             {/* Button 3: Go Back (Aligned Right) */}
                            <button className="btn btn-preview" onClick={() => setStep(3)}>Go Back</button>
                        </div>
                         {/* --- END: Step 4 Actions --- */}
                    </>
                );
            default:
                return <p>An error occurred. Please refresh.</p>;
        }
    };

    // --- Main Component Return ---
    return (
        <div className="predict-container">
            <Navbar />
            <div className="steps-container">
                <div className="step-indicator-container">
                    <div className={`step ${step === 1 ? 'active' : ''}`}>1. Upload MRI</div>
                    <div className={`step ${step === 2 ? 'active' : ''}`}>2. Preview MRI</div>
                    <div className={`step ${step === 3 ? 'active' : ''}`}>3. Detection Result</div>
                    <div className={`step ${step === 4 ? 'active' : ''}`}>4. Visual Interpretations</div>
                </div>
                <div className="step-content">
                    {renderStepContent()}
                </div>
            </div>

            {(isLoading || loadingExplanation) && (
                <div className="loading-screen">
                    <div className="loading-content">
                        <div className="spinner"></div>
                        <p>Loading...</p>
                    </div>
                </div>
            )}
        </div>
    );
}

export default PredictPage;