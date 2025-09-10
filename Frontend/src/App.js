
import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import LoginPage from './components/LoginPage/LoginPage';
import HomePage from './components/HomePage/HomePage';
import InstructionsPage from './components/InstructionsPage/InstructionsPage';
import PredictPage from './components/PredictPage/PredictPage';
import PredictPage_PET from './components/PredictPage_PET/PredictPage_PET';
import PredictPage_Both from './components/PredictPage_Both/PredictPage_Both';
import PrivateRoute from './components/LoginPage/PrivateRoute';

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<LoginPage />} />
        <Route path="/home" element={<PrivateRoute><HomePage /></PrivateRoute>} />
        <Route path="/instructions" element={<PrivateRoute><InstructionsPage /></PrivateRoute>} />
        <Route path="/predict_mri" element={<PrivateRoute><PredictPage /></PrivateRoute>} />
        <Route path="/predict_pet" element={<PrivateRoute><PredictPage_PET /></PrivateRoute>} />
        <Route path="/predict_both" element={<PrivateRoute><PredictPage_Both /></PrivateRoute>} />
      </Routes>
    </Router>
  );
}

export default App;
