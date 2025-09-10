// Navbar.js
import React from 'react';
import { NavLink } from 'react-router-dom';
import './Navbar.css';

function Navbar() {
  return (
    <nav className="navbar">
      <div className="navbar-container">
        <a href="/" className="site-name">Alz-InsightNet.</a>
        <div className="nav-links">
          <NavLink end to="/home" className={({ isActive }) => isActive ? "nav-item active links" : "nav-item links"}>
            Home
          </NavLink>
          <NavLink to="/instructions" className={({ isActive }) => isActive ? "nav-item active links" : "nav-item links"}>
            Instructions
          </NavLink>
          <NavLink to="/predict_both" className={({ isActive }) => isActive ? "nav-item active links" : "nav-item links"}>
            Predict - MRI and PET
          </NavLink>
          <NavLink to="/predict_mri" className={({ isActive }) => isActive ? "nav-item active links" : "nav-item links"}>
            Predict - MRI
          </NavLink>
          <NavLink to="/predict_pet" className={({ isActive }) => isActive ? "nav-item active links" : "nav-item links"}>
            Predict - PET
          </NavLink>
        </div>
      </div>
    </nav>
  );
}

export default Navbar;