// HomePage.js
import React from 'react';
import './HomePage.css';
import Navbar from '../Navbar/Navbar';
import { useNavigate } from 'react-router-dom';
import brainImage from '../../assets/brain_3.png'; // Import the brain image

function HomePage() {
    const navigate = useNavigate();

    const handleGetStartedBtnClick = () => {
        navigate('/instructions');
    };

    return (
        <div className="home-container">
            <Navbar/>
            <div className="content-container">
                <div className="image-container">
                    <img src={brainImage} alt="Brain" className="brain-image" />
                </div>
                <div className="divider"></div>
                <div className="text-content">
                    <p className="wordings animated-text">Empowering You with AI-Driven Brain Health Insights.</p>
                    <p className="wordings animated-text">Precise Alzheimer's staging and AI-powered visualization for enhanced diagnostic confidence.</p>
                    <p className="wordings animated-text ready-text animated-ready">Ready to begin?</p>
                </div>
                <div className="button-container">
                    <button className="get-started-btn animated-button" onClick={handleGetStartedBtnClick}>Get Started</button>
                </div>
            </div>
        </div>
    );
}

export default HomePage;












/**
 * Originally developed by Mr. Savin Madapatha for the dissertation project:
 * "AlzhiScan: A Novel Explainable AI-Focused Approach for Alzheimer’s Disease Detection Using MRI Images"
 *
 * This file has been adapted and updated for use in my individual project.
 * Modifications include design changes, feature enhancements, and integration with custom backend components.
 */
