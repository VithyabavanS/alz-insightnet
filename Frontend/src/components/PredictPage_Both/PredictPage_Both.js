// import React, { useState, useRef } from 'react';
// import './PredictPage_Both.css';
// import Navbar from '../Navbar/Navbar';
// import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
// import jsPDF from 'jspdf';
// import html2canvas from 'html2canvas';

// function PredictPageBoth() {
//   const backendBaseUrl = 'http://127.0.0.1:5000';
//   const [step, setStep] = useState(1);
//   const [mriFile, setMriFile] = useState(null);
//   const [petFile, setPetFile] = useState(null);
//   const [mriPreview, setMriPreview] = useState(null);
//   const [petPreview, setPetPreview] = useState(null);
//   const [mriProgress, setMriProgress] = useState(0);
//   const [petProgress, setPetProgress] = useState(0);
//   const [isMriUploaded, setIsMriUploaded] = useState(false);
//   const [isPetUploaded, setIsPetUploaded] = useState(false);
//   const [explanations, setExplanations] = useState({});
//   const [prediction, setPrediction] = useState({ mri: null, pet: null, fused: null });
//   const [isLoading, setIsLoading] = useState(false);

//   const interpretationsRef = useRef(null); // 🆕 Added ref for PDF generation

//   const handleFileSelect = (event, type) => {
//     const file = event.target.files[0];
//     if (!file || !/\.(jpg|jpeg|png)$/i.test(file.name)) {
//       alert('Only JPG/PNG files allowed.');
//       return;
//     }
//     const preview = URL.createObjectURL(file);
//     if (type === 'mri') {
//       setMriFile(file);
//       setMriPreview(preview);
//       setMriProgress(0);
//       setIsMriUploaded(false);
//     } else {
//       setPetFile(file);
//       setPetPreview(preview);
//       setPetProgress(0);
//       setIsPetUploaded(false);
//     }
//   };

//   const simulateUpload = (type) => {
//     let setProgress = type === 'mri' ? setMriProgress : setPetProgress;
//     let setUploaded = type === 'mri' ? setIsMriUploaded : setIsPetUploaded;
//     let progress = 0;
//     const interval = setInterval(() => {
//       progress += 10;
//       setProgress(progress);
//       if (progress >= 100) {
//         clearInterval(interval);
//         setUploaded(true);
//       }
//     }, 100);
//   };

//   const handlePredict = async () => {
//     setIsLoading(true);
//     try {
//       const formData = new FormData();
//       formData.append('mri', mriFile);
//       formData.append('pet', petFile);
//       const res = await fetch(`${backendBaseUrl}/predict_both`, { method: 'POST', body: formData });
//       const data = await res.json();
//       setPrediction({ mri: data.mri, pet: data.pet, fused: data.fused });
//       setStep(3);
//     } catch (error) {
//       alert('Failed to get prediction.');
//     } finally {
//       setIsLoading(false);
//     }
//   };

//   const handleVisuals = async () => {
//     setIsLoading(true);
//     try {
//       const res = await fetch(`${backendBaseUrl}/explain_both`);
//       const data = await res.json();
//       setExplanations(data);
//       setStep(4);
//     } catch (error) {
//       alert('Failed to load explanations.');
//     } finally {
//       setIsLoading(false);
//     }
//   };

//   // const handleDownloadVisualsPdf = async () => {
//   //   if (!interpretationsRef.current) {
//   //     alert("No visual interpretations to download.");
//   //     return;
//   //   }

//   //   try {
//   //     const canvas = await html2canvas(interpretationsRef.current, {
//   //       useCORS: true,
//   //       scale: 2,
//   //     });

//   //     const imgData = canvas.toDataURL('image/png');
//   //     const pdf = new jsPDF('landscape', 'pt', 'a4');
//   //     const pdfWidth = pdf.internal.pageSize.getWidth();
//   //     const pdfHeight = pdf.internal.pageSize.getHeight();

//   //     const ratio = Math.min(pdfWidth / canvas.width, pdfHeight / canvas.height);
//   //     const x = (pdfWidth - canvas.width * ratio) / 2;
//   //     const y = 30;

//   //     pdf.setFontSize(16);
//   //     pdf.setFont(undefined, 'bold');
//   //     pdf.text("Visual Interpretations Report", pdfWidth / 2, 20, { align: 'center' });
//   //     pdf.addImage(imgData, 'PNG', x, y, canvas.width * ratio, canvas.height * ratio);
//   //     pdf.save('interpretations_report.pdf');
//   //   } catch (err) {
//   //     console.error("PDF generation failed", err);
//   //     alert("Failed to generate PDF.");
//   //   }
//   // };

//   const handleDownloadVisualsPdf = async () => {
//     if (!interpretationsRef.current) {
//       alert("No visual interpretations to download.");
//       return;
//     }
  
//     const fused = prediction.fused || {};
//     const mriFileName = mriFile?.name || 'N/A';
//     const petFileName = petFile?.name || 'N/A';
  
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
  
//       // Portrait A4 PDF (595.28pt x 841.89pt)
//       const pdf = new jsPDF({
//         orientation: 'portrait',
//         unit: 'pt',
//         format: 'a4',
//       });
  
//       const pdfWidth = 595.28;
//       const pdfHeight = 841.89;
//       const margin = 40;
//       let currentY = margin;
  
//       // 1. Title
//       pdf.setFontSize(16);
//       pdf.setFont(undefined, 'bold');
//       pdf.text('Visual Interpretations Report', pdfWidth / 2, currentY, { align: 'center' });
//       currentY += 25;
  
//       // 2. File Info
//       pdf.setFontSize(11);
//       pdf.setFont(undefined, 'normal');
//       pdf.text(`Source Files: MRI - ${mriFileName}, PET - ${petFileName}`, pdfWidth / 2, currentY, { align: 'center' });
//       currentY += 25;
  
//       // 3. Prediction Results
//       if (fused.predicted_label) {
//         pdf.setFont(undefined, 'bold');
//         pdf.text('Prediction Result:', margin, currentY);
//         currentY += 15;
  
//         pdf.setFont(undefined, 'normal');
//         pdf.text(`- Predicted Stage: ${fused.predicted_label}`, margin + 15, currentY);
//         currentY += 15;
//         pdf.text(`- Confidence Score: ${fused.confidence?.toFixed(2) ?? 'N/A'}%`, margin + 15, currentY);
//         currentY += 25;
//       } else {
//         pdf.setFont(undefined, 'italic');
//         pdf.text('Prediction results were not available for this report.', margin, currentY);
//         currentY += 25;
//       }
  
//       // 4. Visual Header
//       pdf.setFontSize(14);
//       pdf.setFont(undefined, 'bold');
//       pdf.text('Explanation Visualizations:', pdfWidth / 2, currentY, { align: 'center' });
//       currentY += 20;
  
//       // 5. Calculate scaled image size
//       const availableWidth = pdfWidth - margin * 2;
//       const availableHeight = pdfHeight - currentY - margin;
  
//       const scale = Math.min(
//         availableWidth / imgWidth,
//         availableHeight / imgHeight
//       );
  
//       const finalImageWidth = imgWidth * scale;
//       const finalImageHeight = imgHeight * scale;
//       const xOffset = (pdfWidth - finalImageWidth) / 2;
  
//       // 6. Add image
//       pdf.addImage(
//         imgData,
//         'PNG',
//         xOffset,
//         currentY,
//         finalImageWidth,
//         finalImageHeight
//       );
  
//       pdf.save(`visual_interpretations_${mriFileName.split('.')[0] || 'report'}.pdf`);
//     } catch (err) {
//       console.error("PDF generation failed", err);
//       alert("Failed to generate PDF.");
//     }
//   };
  

//   const resetAll = () => {
//     setStep(1);
//     setMriFile(null);
//     setPetFile(null);
//     setMriPreview(null);
//     setPetPreview(null);
//     setMriProgress(0);
//     setPetProgress(0);
//     setIsMriUploaded(false);
//     setIsPetUploaded(false);
//     setExplanations({});
//     setPrediction({ mri: null, pet: null, fused: null });
//     setIsLoading(false);
//   };

//   const renderBarChart = (data) => {
//     if (!data || !data.all_confidences) {
//       return <p>No confidence data available.</p>;
//     }

//     const chartData = Object.entries(data.all_confidences).map(([stage, confidence]) => ({
//       stage,
//       confidence: confidence || 0,
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

//   return (
//     <div className="predict-container">
//       <Navbar />
//       <div className="steps-container">
//         <div className="step-indicator-container">
//           <div className={`step ${step === 1 ? 'active' : ''}`}>1. Upload MRI and PET</div>
//           <div className={`step ${step === 2 ? 'active' : ''}`}>2. Preview Images</div>
//           <div className={`step ${step === 3 ? 'active' : ''}`}>3. Detection Result</div>
//           <div className={`step ${step === 4 ? 'active' : ''}`}>4. Visual Interpretations</div>
//         </div>

//         {/* Upload Step */}
//         {step === 1 && (
//           <>
//             {/* MRI Upload */}
//             <div className="upload-section">
//               <h3>MRI Upload</h3>
//               <div className="file-upload-container">
//                 <input type="file" onChange={(e) => handleFileSelect(e, 'mri')} />
//                 <button className="btn" onClick={() => simulateUpload('mri')} disabled={!mriFile}>Upload</button>
//                 <progress value={mriProgress} max="100"></progress>
//               </div>
//             </div>

//             {/* PET Upload */}
//             <div className="upload-section">
//               <h3>PET Upload</h3>
//               <div className="file-upload-container">
//                 <input type="file" onChange={(e) => handleFileSelect(e, 'pet')} />
//                 <button className="btn" onClick={() => simulateUpload('pet')} disabled={!petFile}>Upload</button>
//                 <progress value={petProgress} max="100"></progress>
//               </div>
//             </div>

//             {isMriUploaded && isPetUploaded && (
//               <div className="step-actions">
//                 <button className="btn btn-preview" onClick={() => setStep(2)}>Preview</button>
//               </div>
//             )}
//           </>
//         )}

//         {/* Preview Step */}
//         {step === 2 && (
//           <>
//             {[{ label: 'MRI', src: mriPreview, file: mriFile }, { label: 'PET', src: petPreview, file: petFile }].map(({ label, src, file }) => (
//               <div className="preview-container" key={label}>
//                 {src ? (
//                   <>
//                     <img src={src} alt={label} className="preview-image" />
//                     <div className="file-details">
//                       <p><strong>{label} File:</strong> {file?.name}</p>
//                       <p><strong>Type:</strong> {file?.type}</p>
//                       <p><strong>Size:</strong> {file?.size?.toLocaleString()} bytes</p>
//                     </div>
//                   </>
//                 ) : (
//                   <p>No {label} image uploaded.</p>
//                 )}
//               </div>
//             ))}
//             <div className="step-actions">
//               <button className="btn btn-back" onClick={() => setStep(1)}>Back</button>
//               <button className="btn btn-predict" onClick={handlePredict}>Predict</button>
//             </div>
//           </>
//         )}

//         {/* Results Step */}
//         {step === 3 && (
//           <>
//             {[{ label: 'MRI Results', data: prediction.mri, file: mriFile, preview: mriPreview },
//               { label: 'PET Results', data: prediction.pet, file: petFile, preview: petPreview },
//               { label: 'Combined Results', data: prediction.fused, file: null, preview: null }]
//               .map(({ label, data, file, preview }) => (
//                 <div className="result-container" key={label}>
//                   <h3 style={{ marginBottom: '1rem', color: 'black' }}>{label}</h3>

//                   <div className="result-top-section">
//                     <div className="result-details">
//                       {file && (
//                         <div className="detail-item">
//                           <span className="detail-label">File</span>: {file.name}
//                         </div>
//                       )}
//                       <div className="detail-item">
//                         <span className="detail-label">Stage</span>: {data?.predicted_label}
//                       </div>
//                       <div className="detail-item">
//                         <span className="detail-label">Confidence</span>: {data?.confidence?.toFixed(2)}%
//                       </div>
//                     </div>
//                     {preview && <img src={preview} alt={`${label} preview`} className="result-image" />}
//                   </div>

//                   <div className="confidence-bars">
//                     <h3 className="confidence-bar-heading">Results Summary</h3>
//                     {renderBarChart(data)}
//                   </div>
//                 </div>
//               ))}
//             <div className="step-actions">
//               <button className="btn btn-back" onClick={() => setStep(2)}>Back</button>
//               <button className="btn" onClick={handleVisuals}>Visual Interpretations</button>
//             </div>
//           </>
//         )}

//         {/* Visual Interpretations Step */}
//         {step === 4 && (
//           <>
//             <div className="interpretations-row" ref={interpretationsRef}>
//               {[{ label: 'MRI Model 01', data: explanations?.mri?.densenet },
//                 { label: 'MRI Model 02', data: explanations?.mri?.resnet },
//                 { label: 'PET Model', data: explanations?.pet }]
//                 .map(({ label, data }) => (
//                   <div className="model-column" key={label}>
//                     <h2>{label}</h2>
//                     {['lime', 'gradcam', 'ig'].map(type => (
//                       <img
//                         key={type}
//                         src={`data:image/png;base64,${data?.[type]}`}
//                         className="interpretation-image"
//                         alt={`${label} ${type}`}
//                       />
//                     ))}
//                   </div>
//               ))}
//             </div>
//             <div className="step-actions">
//   {/* Button 1: New Scan (Left) */}
//   <button className="btn btn-back" onClick={resetAll}>New Scan</button>

//   {/* Button 2: Download (Center) */}
//   <button
//     className="btn btn-secondary btn-download"
//     onClick={handleDownloadVisualsPdf}
//     disabled={!explanations || Object.keys(explanations).length === 0}
//   >
//     Download Interpretations
//   </button>

//   {/* Button 3: Go Back (Right) */}
//   <button className="btn btn-preview" onClick={() => setStep(3)}>Go Back</button>
// </div>

//           </>
//         )}
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

// export default PredictPageBoth;



import React, { useState, useRef } from 'react';
import './PredictPage_Both.css';
import Navbar from '../Navbar/Navbar';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import jsPDF from 'jspdf';
import html2canvas from 'html2canvas';

function PredictPageBoth() {
  const backendBaseUrl = 'http://127.0.0.1:5000';
  const [step, setStep] = useState(1);
  const [mriFile, setMriFile] = useState(null);
  const [petFile, setPetFile] = useState(null);
  const [mriPreview, setMriPreview] = useState(null);
  const [petPreview, setPetPreview] = useState(null);
  const [mriProgress, setMriProgress] = useState(0);
  const [petProgress, setPetProgress] = useState(0);
  const [isMriUploaded, setIsMriUploaded] = useState(false);
  const [isPetUploaded, setIsPetUploaded] = useState(false);
  const [explanations, setExplanations] = useState({});
  const [prediction, setPrediction] = useState({ mri: null, pet: null, fused: null });
  const [isLoading, setIsLoading] = useState(false);

  const interpretationsRef = useRef(null);

  const handleFileSelect = async (event, type) => {
    const file = event.target.files[0];
    if (!file || !/\.(jpg|jpeg|png)$/i.test(file.name)) {
        alert('Only JPG/PNG files allowed.');
        return;
    }

    try {
        const formData = new FormData();
        formData.append('file', file);

        const validationEndpoint = type === 'mri' 
            ? `${backendBaseUrl}/validate_mri` 
            : `${backendBaseUrl}/validate_pet`;

        const response = await fetch(validationEndpoint, {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            throw new Error(`Validation HTTP error! Status: ${response.status}`);
        }

        const result = await response.json();

        if (!result.is_valid) {
            alert(` Uploaded ${type.toUpperCase()} image is NOT recognised as a valid scan.`);
            return; // Stop if not valid
        }

        const preview = URL.createObjectURL(file);

        if (type === 'mri') {
            if (mriPreview) URL.revokeObjectURL(mriPreview); // Revoke previous
            setMriFile(file);
            setMriPreview(preview);
            setMriProgress(0);
            setIsMriUploaded(false);
        } else {
            if (petPreview) URL.revokeObjectURL(petPreview); // Revoke previous
            setPetFile(file);
            setPetPreview(preview);
            setPetProgress(0);
            setIsPetUploaded(false);
        }
    } catch (error) {
        console.error(`Validation failed for ${type.toUpperCase()} scan:`, error);
        alert(`Failed to validate the ${type.toUpperCase()} scan. Please try again.`);
    }
};

  const simulateUpload = (type) => {
    let setProgress = type === 'mri' ? setMriProgress : setPetProgress;
    let setUploaded = type === 'mri' ? setIsMriUploaded : setIsPetUploaded;
    let progress = 0;
    const interval = setInterval(() => {
      progress += 10;
      setProgress(progress);
      if (progress >= 100) {
        clearInterval(interval);
        setUploaded(true);
      }
    }, 100);
  };

  const handleCancelUpload = (type) => {
    if (type === 'mri') {
      setMriFile(null);
      setMriPreview(null);
      setMriProgress(0);
      setIsMriUploaded(false);
    } else {
      setPetFile(null);
      setPetPreview(null);
      setPetProgress(0);
      setIsPetUploaded(false);
    }
  };

  const handlePredict = async () => {
    setIsLoading(true);
    try {
      const formData = new FormData();
      formData.append('mri', mriFile);
      formData.append('pet', petFile);
      const res = await fetch(`${backendBaseUrl}/predict_both`, { method: 'POST', body: formData });
      const data = await res.json();
      setPrediction({ mri: data.mri, pet: data.pet, fused: data.fused });
      setStep(3);
    } catch (error) {
      alert('Failed to get prediction.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleVisuals = async () => {
    setIsLoading(true);
    try {
      const res = await fetch(`${backendBaseUrl}/explain_both`);
      const data = await res.json();
      setExplanations(data);
      setStep(4);
    } catch (error) {
      alert('Failed to load explanations.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleDownloadVisualsPdf = async () => {
    if (!interpretationsRef.current) {
      alert("No visual interpretations to download.");
      return;
    }

    const fused = prediction.fused || {};
    const mriFileName = mriFile?.name || 'N/A';
    const petFileName = petFile?.name || 'N/A';

    try {
      const canvas = await html2canvas(interpretationsRef.current, {
        useCORS: true,
        scale: 2,
        scrollX: 0,
        scrollY: -window.scrollY,
        windowWidth: interpretationsRef.current.scrollWidth,
        windowHeight: interpretationsRef.current.scrollHeight,
      });

      const imgData = canvas.toDataURL('image/png');
      const imgWidth = canvas.width;
      const imgHeight = canvas.height;

      const pdf = new jsPDF({
        orientation: 'portrait',
        unit: 'pt',
        format: 'a4',
      });

      const pdfWidth = 595.28;
      const pdfHeight = 841.89;
      const margin = 40;
      let currentY = margin;

      pdf.setFontSize(16);
      pdf.setFont(undefined, 'bold');
      pdf.text('Visual Interpretations Report', pdfWidth / 2, currentY, { align: 'center' });
      currentY += 25;

      pdf.setFontSize(11);
      pdf.setFont(undefined, 'normal');
      pdf.text(`Source Files: MRI - ${mriFileName}, PET - ${petFileName}`, pdfWidth / 2, currentY, { align: 'center' });
      currentY += 25;

      if (fused.predicted_label) {
        pdf.setFont(undefined, 'bold');
        pdf.text('Prediction Result:', margin, currentY);
        currentY += 15;

        pdf.setFont(undefined, 'normal');
        pdf.text(`- Predicted Stage: ${fused.predicted_label}`, margin + 15, currentY);
        currentY += 15;
        pdf.text(`- Confidence Score: ${fused.confidence?.toFixed(2) ?? 'N/A'}%`, margin + 15, currentY);
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

      const availableWidth = pdfWidth - margin * 2;
      const availableHeight = pdfHeight - currentY - margin;

      const scale = Math.min(
        availableWidth / imgWidth,
        availableHeight / imgHeight
      );

      const finalImageWidth = imgWidth * scale;
      const finalImageHeight = imgHeight * scale;
      const xOffset = (pdfWidth - finalImageWidth) / 2;

      pdf.addImage(
        imgData,
        'PNG',
        xOffset,
        currentY,
        finalImageWidth,
        finalImageHeight
      );

      pdf.save(`visual_interpretations_${mriFileName.split('.')[0] || 'report'}.pdf`);
    } catch (err) {
      console.error("PDF generation failed", err);
      alert("Failed to generate PDF.");
    }
  };

  const resetAll = () => {
    setStep(1);
    setMriFile(null);
    setPetFile(null);
    setMriPreview(null);
    setPetPreview(null);
    setMriProgress(0);
    setPetProgress(0);
    setIsMriUploaded(false);
    setIsPetUploaded(false);
    setExplanations({});
    setPrediction({ mri: null, pet: null, fused: null });
    setIsLoading(false);
  };

  const renderBarChart = (data) => {
    if (!data || !data.all_confidences) {
      return <p>No confidence data available.</p>;
    }

    const chartData = Object.entries(data.all_confidences).map(([stage, confidence]) => ({
      stage,
      confidence: confidence || 0,
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

  return (
    <div className="predict-container">
      <Navbar />
      <div className="steps-container">
        <div className="step-indicator-container">
          <div className={`step ${step === 1 ? 'active' : ''}`}>1. Upload MRI and PET</div>
          <div className={`step ${step === 2 ? 'active' : ''}`}>2. Preview Images</div>
          <div className={`step ${step === 3 ? 'active' : ''}`}>3. Detection Result</div>
          <div className={`step ${step === 4 ? 'active' : ''}`}>4. Visual Interpretations</div>
        </div>

        {step === 1 && (
          <>
            <div className="upload-section">
              <h3>MRI Upload</h3>
              <div className="file-upload-container">
                <input type="file" onChange={(e) => handleFileSelect(e, 'mri')} />
                <div className="upload-buttons" style={{ display: 'flex' }}>
                  <button
                    className="btn"
                    onClick={() => simulateUpload('mri')}
                    disabled={!mriFile}
                  >
                    Upload
                  </button>
                  <button
                    className="btn btn-cancel"
                    onClick={() => handleCancelUpload('mri')}
                    disabled={!mriFile}
                    style={{ marginLeft: 'auto' }} // Push to the right
                  >
                    Cancel
                  </button>
                </div>
                <progress value={mriProgress} max="100"></progress>
              </div>
            </div>

            <div className="upload-section">
              <h3>PET Upload</h3>
              <div className="file-upload-container">
                <input type="file" onChange={(e) => handleFileSelect(e, 'pet')} />
                <div className="upload-buttons" style={{ display: 'flex' }}>
                  <button
                    className="btn"
                    onClick={() => simulateUpload('pet')}
                    disabled={!petFile}
                  >
                    Upload
                  </button>
                  <button
                    className="btn btn-cancel"
                    onClick={() => handleCancelUpload('pet')}
                    disabled={!petFile}
                    style={{ marginLeft: 'auto' }} // Push to the right
                  >
                    Cancel
                  </button>
                </div>
                <progress value={petProgress} max="100"></progress>
              </div>
            </div>

            {isMriUploaded && isPetUploaded && (
              <div className="step-actions">
                <button className="btn btn-preview" onClick={() => setStep(2)}>Preview</button>
              </div>
            )}
          </>
        )}

        {step === 2 && (
          <>
            {[
              { label: 'MRI', src: mriPreview, file: mriFile },
              { label: 'PET', src: petPreview, file: petFile },
            ].map(({ label, src, file }) => (
              <div className="preview-container" key={label}>
                {src ? (
                  <>
                    <img src={src} alt={label} className="preview-image" />
                    <div className="file-details">
                      <p>
                        <strong>{label} File:</strong> {file?.name}
                      </p>
                      <p>
                        <strong>Type:</strong> {file?.type}
                      </p>
                      <p>
                        <strong>Size:</strong> {file?.size?.toLocaleString()} bytes
                      </p>
                    </div>
                  </>
                ) : (
                  <p>No {label} image uploaded.</p>
                )}
              </div>
            ))}
            <div className="step-actions">
              <button className="btn btn-back" onClick={() => setStep(1)}>
                Back
              </button>
              <button className="btn btn-predict" onClick={handlePredict}>
                Predict
              </button>
            </div>
          </>
        )}

        {step === 3 && (
          <>
            {[
              { label: 'MRI Results', data: prediction.mri, file: mriFile, preview: mriPreview },
              { label: 'PET Results', data: prediction.pet, file: petFile, preview: petPreview },
              { label: 'Combined Results', data: prediction.fused, file: null, preview: null },
            ].map(({ label, data, file, preview }) => (
              <div className="result-container" key={label}>
                <h3 style={{ marginBottom: '1rem', color: 'black' }}>{label}</h3>

                <div className="result-top-section">
                  <div className="result-details">
                    {file && (
                      <div className="detail-item">
                        <span className="detail-label">File</span>: {file.name}
                      </div>
                    )}
                    <div className="detail-item">
                      <span className="detail-label">Stage</span>: {data?.predicted_label}
                    </div>
                    <div className="detail-item">
                      <span className="detail-label">Confidence</span>: {data?.confidence?.toFixed(2)}%
                    </div>
                  </div>
                  {preview && <img src={preview} alt={`${label} preview`} className="result-image" />}
                </div>

                <div className="confidence-bars">
                  <h3 className="confidence-bar-heading">Results Summary</h3>
                  {renderBarChart(data)}
                </div>
              </div>
            ))}
            <div className="step-actions">
              <button className="btn btn-back" onClick={() => setStep(2)}>
                Back
              </button>
              <button className="btn" onClick={handleVisuals}>
                Visual Interpretations
              </button>
            </div>
          </>
        )}

        {step === 4 && (
          <>
            <div className="interpretations-row" ref={interpretationsRef}>
              {[
                { label: 'MRI Model 01', data: explanations?.mri?.densenet },
                { label: 'MRI Model 02', data: explanations?.mri?.resnet },
                { label: 'PET Model', data: explanations?.pet },
              ].map(({ label, data }) => (
                <div className="model-column" key={label}>
                  <h2>{label}</h2>
                  {['lime', 'gradcam', 'ig'].map((type) => (
                    <img
                      key={type}
                      src={`data:image/png;base64,${data?.[type]}`}
                      className="interpretation-image"
                      alt={`${label} ${type}`}
                    />
                  ))}
                </div>
              ))}
            </div>
            <div className="step-actions">
              <button className="btn btn-back" onClick={resetAll}>
                New Scan
              </button>
              <button
                className="btn btn-secondary btn-download"
                onClick={handleDownloadVisualsPdf}
                disabled={!explanations || Object.keys(explanations).length === 0}
              >
                Download Interpretations
              </button>
              <button className="btn btn-preview" onClick={() => setStep(3)}>
                Go Back
              </button>
            </div>
          </>
        )}
      </div>

      {isLoading && (
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

export default PredictPageBoth;