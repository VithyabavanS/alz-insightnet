import React from 'react';
import './InstructionsPage.css';
import Navbar from '../Navbar/Navbar';

function InstructionsPage() {
  return (
    <div className="instructions-container">
      <Navbar />
      <div className="instructions-content">
        <h1>How to use Alz-InsightNet?</h1>

        <h2>Predict - MRI and PET Based Detection</h2>
        <ul className="instructions-list">
          <li className="list">
            Alzh-InsightNet offers stage prediction of Alzheimer’s Disease using both MRI and PET brain scans, along with visual interpretations.
          </li>
          <li className="list">
            Please upload 2D brain MRI and PET images of the <b>same patient</b>. Uploading mismatched or unrelated images may affect prediction accuracy.
          </li>
          <li className="list">On the '<b>Predict - MRI and PET</b>' page:</li>
          <ul className="predict-steps">
            <li className="list">1 - Upload the 2D brain <b>MRI</b> image.</li>
            <li className="list">2 - Upload the corresponding 2D brain <b>PET</b> image.</li>
            <li className="list">3 - Click ‘<b>Preview</b>’ to confirm the uploaded images.</li>
            <li className="list">4 - Click ‘<b>Predict Stage</b>’ to detect the Alzheimer’s stage.</li>
            <li className="list">
              5 - To understand the prediction, click ‘<b>View Visual Interpretations</b>’. This will show highlighted regions that contributed to the diagnosis.
            </li>
          </ul>
        </ul>

        <h2>Predict - MRI Only Detection</h2>
        <ul className="instructions-list">
          <li className="list">
            You can also use Alzh-InsightNet for Alzheimer's detection using only MRI scans, ideal when PET images are not available.
          </li>
          <li className="list">
            Please upload a clear 2D brain MRI image. Other image types may lead to incorrect results.
          </li>
          <li className="list">On the '<b>Predict - MRI</b>' page:</li>
          <ul className="predict-steps">
            <li className="list">1 - Upload the 2D brain <b>MRI</b> image.</li>
            <li className="list">2 - Click ‘<b>Preview</b>’ to view the image.</li>
            <li className="list">3 - Click ‘<b>Predict Stage</b>’ to get the Alzheimer’s prediction.</li>
            <li className="list">
              4 - To explore how the result was derived, click ‘<b>View Visual Interpretations</b>’.
            </li>
          </ul>
        </ul>

        <h2>Predict - PET Only Detection</h2>
        <ul className="instructions-list">
          <li className="list">
            Alzh-InsightNet also supports Alzheimer's detection using only PET scans.
          </li>
          <li className="list">
            Please upload a clear 2D brain PET image. Other image types may lead to incorrect results.
          </li>
          <li className="list">On the '<b>Predict - PET</b>' page:</li>
          <ul className="predict-steps">
            <li className="list">1 - Upload the 2D brain <b>PET</b> image.</li>
            <li className="list">2 - Click ‘<b>Preview</b>’ to view the image.</li>
            <li className="list">3 - Click ‘<b>Predict Stage</b>’ to get the Alzheimer’s prediction.</li>
            <li className="list">
              4 - To explore how the result was derived, click ‘<b>View Visual Interpretations</b>’.
            </li>
          </ul>
        </ul>
      </div>
    </div>
  );
}

export default InstructionsPage;